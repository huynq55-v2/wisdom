use burn::backend::wgpu::WgpuDevice;
use burn::module::AutodiffModule;
use burn::record::NamedMpkFileRecorder;
use burn::{
    backend::{Autodiff, Wgpu},
    data::dataloader::batcher::Batcher,
    optim::{AdamConfig, GradientsParams, Optimizer, decay::WeightDecayConfig},
    prelude::*,
    record::Recorder,
};
use rand::RngExt;

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;
use wisdom::board::{Board, Color};
use wisdom::nn::board_to_tensor;
use wisdom::nn::{
    BOARD_H, BOARD_W, NUM_PLANES, TENSOR_SIZE, XiangqiNetConfig, XiangqiTrainingBatch,
};

// ==========================================================
// 1. Data Structures
// ==========================================================

#[derive(Clone, Debug)]
pub struct TrainItem {
    pub fen: String,
    pub value: f32,
    pub policy: usize,
}

// ==========================================================
// 2. Batcher (giống selfplay.rs, có data augmentation)
// ==========================================================

#[derive(Clone)]
pub struct XiangqiBatcher<B: Backend> {
    device: B::Device,
    is_training: bool,
}

impl<B: Backend> XiangqiBatcher<B> {
    pub fn new(device: B::Device, is_training: bool) -> Self {
        Self {
            device,
            is_training,
        }
    }
}

impl<B: Backend> Batcher<B, TrainItem, XiangqiTrainingBatch<B>> for XiangqiBatcher<B> {
    fn batch(&self, items: Vec<TrainItem>, device: &B::Device) -> XiangqiTrainingBatch<B> {
        let batch_size = items.len();

        let mut inputs_flat = Vec::with_capacity(batch_size * TENSOR_SIZE);
        let mut targets_v_flat = Vec::with_capacity(batch_size);
        let mut targets_p_flat = Vec::with_capacity(batch_size);

        let mut rng = rand::rng();

        for item in items {
            let mut board = Board::new();
            wisdom::ucci::parse_fen(&mut board, &item.fen);
            let mut tensor = board_to_tensor(&board);

            let from_dense = item.policy / 90;
            let to_dense = item.policy % 90;

            let mut from_r = from_dense / 9;
            let mut from_c = from_dense % 9;
            let mut to_r = to_dense / 9;
            let mut to_c = to_dense % 9;

            // LẬT PERSPECTIVE NẾU LÀ ĐEN
            if board.side_to_move == Color::Black {
                from_r = 9 - from_r;
                from_c = 8 - from_c;
                to_r = 9 - to_r;
                to_c = 8 - to_c;
            }

            // DATA AUGMENTATION: LẬT GƯƠNG NGANG (50%)
            if self.is_training && rng.random_bool(0.5) {
                for plane in 0..14 {
                    for r in 0..10 {
                        for c in 0..4 {
                            let idx1 = plane * 90 + r * 9 + c;
                            let idx2 = plane * 90 + r * 9 + (8 - c);
                            tensor.swap(idx1, idx2);
                        }
                    }
                }
                from_c = 8 - from_c;
                to_c = 8 - to_c;
            }

            let policy_idx = (from_r * 9 + from_c) * 90 + (to_r * 9 + to_c);

            inputs_flat.extend_from_slice(&tensor);
            targets_v_flat.push(item.value.clamp(-1.0, 1.0));
            targets_p_flat.push(policy_idx as i32);
        }

        let inputs = Tensor::<B, 1>::from_data(inputs_flat.as_slice(), device)
            .reshape([batch_size, NUM_PLANES, BOARD_H, BOARD_W]);

        let targets_v =
            Tensor::<B, 1>::from_data(targets_v_flat.as_slice(), device).reshape([batch_size, 1]);

        let targets_p =
            Tensor::<B, 1, burn::tensor::Int>::from_data(targets_p_flat.as_slice(), device);

        XiangqiTrainingBatch {
            inputs,
            targets_v,
            targets_p,
        }
    }
}

// ==========================================================
// 3. Đọc replay_buffer.csv vào RAM
// ==========================================================

fn load_replay_buffer(csv_path: &str) -> Vec<TrainItem> {
    println!("📂 Đang nạp dữ liệu từ {}...", csv_path);
    let file =
        fs::File::open(csv_path).unwrap_or_else(|_| panic!("❌ Không tìm thấy file {}", csv_path));
    let reader = BufReader::new(file);

    let mut items = Vec::new();
    let mut invalid_count = 0;

    for line in reader.lines().map_while(|l| l.ok()) {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() == 3 {
            if let (Ok(value), Ok(policy)) = (parts[1].parse::<f32>(), parts[2].parse::<usize>()) {
                if policy < 8100 {
                    items.push(TrainItem {
                        fen: parts[0].to_string(),
                        value,
                        policy,
                    });
                } else {
                    invalid_count += 1;
                }
            }
        }
    }

    if invalid_count > 0 {
        println!(
            "⚠️  Bỏ qua {} vị trí có action index không hợp lệ.",
            invalid_count
        );
    }
    println!("✅ Đã nạp {} positions hợp lệ.", items.len());
    items
}

// ==========================================================
// 4. Cosine Annealing LR Schedule
// ==========================================================

/// Cosine Annealing: lr giảm mượt từ lr_max → eta_min
fn cosine_lr(step: usize, total_steps: usize, lr_max: f64, eta_min: f64) -> f64 {
    eta_min
        + (lr_max - eta_min)
            * 0.5
            * (1.0 + (std::f64::consts::PI * step as f64 / total_steps as f64).cos())
}

// ==========================================================
// 5. Hàm tách riêng để chạy Validation (Khắc phục lỗi .inner())
// ==========================================================
fn run_validation<B: Backend>(
    model: &wisdom::nn::XiangqiNet<B>,
    val_data: &[TrainItem],
    batch_size: usize,
    device: &B::Device,
) -> (f64, f64, f64) {
    let batcher_val = XiangqiBatcher::<B>::new(device.clone(), false);
    let val_batches = (val_data.len() + batch_size - 1) / batch_size;

    let mut val_loss_v = 0.0f64;
    let mut val_loss_p = 0.0f64;
    let mut val_correct: usize = 0;
    let mut val_total: usize = 0;

    for batch_idx in 0..val_batches {
        let start = batch_idx * batch_size;
        let end = std::cmp::min(start + batch_size, val_data.len());
        let batch_items = val_data[start..end].to_vec();
        let actual_batch_size = batch_items.len();

        // Batch này sinh ra Tensor thuần (Không Autodiff), không cần gọi .inner()
        let batch = batcher_val.batch(batch_items, device);

        let (pred_value, pred_policy) = model.forward(batch.inputs);

        let loss_v = burn::nn::loss::MseLoss::new().forward(
            pred_value,
            batch.targets_v,
            burn::nn::loss::Reduction::Mean,
        );

        let loss_p = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&batch.targets_p.device())
            .forward(pred_policy.clone(), batch.targets_p.clone());

        val_loss_v += loss_v.into_scalar().to_f64() * actual_batch_size as f64;
        val_loss_p += loss_p.into_scalar().to_f64() * actual_batch_size as f64;

        let predicted = pred_policy.argmax(1).reshape([actual_batch_size]);
        let targets = batch.targets_p.reshape([actual_batch_size]);
        let is_correct = predicted.equal(targets);
        let correct_count: f32 = is_correct.into_data().convert::<f32>().iter::<f32>().sum();

        val_correct += correct_count as usize;
        val_total += actual_batch_size;
    }

    let avg_val_v = val_loss_v / val_total as f64;
    let avg_val_p = val_loss_p / val_total as f64;
    let val_accuracy = (val_correct as f64 / val_total as f64) * 100.0;

    (avg_val_v, avg_val_p, val_accuracy)
}

// ==========================================================
// 6. Main Training Loop
// ==========================================================

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    let config = XiangqiNetConfig::new();
    let model_dir = "./wisdom_models";

    // --- HYPERPARAMETERS ---
    let num_epochs = 10;
    let batch_size: usize = 256;
    let lr_max = 2e-3;
    let eta_min = 1e-5;
    let weight_decay = 1e-4;

    // TẦN SUẤT VALIDATION (Đổi thành 1 nếu muốn chạy Valid sau MỖI batch)
    let valid_interval = 5;

    // --- NẠP DỮ LIỆU ---
    let csv_path = format!("{}/replay_buffer.csv", model_dir);
    let all_data = load_replay_buffer(&csv_path);

    if all_data.is_empty() {
        println!("❌ Không có dữ liệu để train!");
        return;
    }

    use rand::seq::SliceRandom;
    let mut shuffled = all_data;
    shuffled.shuffle(&mut rand::rng());

    let val_size = (shuffled.len() as f32 * 0.001) as usize;
    let train_size = shuffled.len() - val_size;
    let train_data = shuffled[..train_size].to_vec();
    let val_data = shuffled[train_size..].to_vec();

    println!(
        "\n📊 Train: {} samples | Validation: {} samples",
        train_size, val_size
    );

    let base_model_path = format!("{}/xiangqi_net_base", model_dir);
    let temp_transfer_path = format!("{}/temp_train_transfer", model_dir);

    let base_model: wisdom::nn::XiangqiNet<MyBackend>;
    if std::path::Path::new(&format!("{}.mpk", base_model_path)).exists() {
        println!("📥 Đang nạp base model từ {}.mpk...", base_model_path);
        let record = NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default()
            .load(base_model_path.into(), &device)
            .expect("❌ Không thể nạp base model!");
        base_model = config.init::<MyBackend>(&device).load_record(record);
    } else {
        println!("🆕 Không tìm thấy base model, khởi tạo model mới...");
        base_model = config.init::<MyBackend>(&device);
    }

    base_model
        .save_file(
            &temp_transfer_path,
            &NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default(),
        )
        .unwrap();

    let autodiff_record = NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default()
        .load(temp_transfer_path.clone().into(), &device)
        .unwrap();

    let current_model = config
        .init::<MyAutodiffBackend>(&device)
        .load_record(autodiff_record);

    use burn::nn::loss::{CrossEntropyLossConfig, MseLoss};

    let optimizer_config =
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(weight_decay)));
    let mut optimizer =
        optimizer_config.init::<MyAutodiffBackend, wisdom::nn::XiangqiNet<MyAutodiffBackend>>();

    let mut current_model = current_model;

    let steps_per_epoch = (train_size + batch_size - 1) / batch_size;
    let total_steps = steps_per_epoch * num_epochs;

    let separator = "=".repeat(80);

    println!(
        "\n🚀 Bắt đầu Training: {} epochs, {} steps/epoch",
        num_epochs, steps_per_epoch
    );
    println!("{}\n", separator);

    let mut global_step: usize = 0;

    for epoch in 0..num_epochs {
        let epoch_start = Instant::now();
        let mut epoch_train = train_data.clone();
        epoch_train.shuffle(&mut rand::rng());

        let num_batches = (epoch_train.len() + batch_size - 1) / batch_size;

        let mut total_loss_v = 0.0f64;
        let mut total_loss_p = 0.0f64;
        let mut train_correct: usize = 0;
        let mut train_samples: usize = 0;

        let batcher_train = XiangqiBatcher::<MyAutodiffBackend>::new(device.clone(), true);

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = std::cmp::min(start + batch_size, epoch_train.len());
            let batch_items = epoch_train[start..end].to_vec();
            let actual_batch_size = batch_items.len();

            let lr = cosine_lr(global_step, total_steps, lr_max, eta_min);
            let batch = batcher_train.batch(batch_items, &device);

            let (pred_value, pred_policy) = current_model.forward(batch.inputs);

            let loss_v = MseLoss::new().forward(
                pred_value.clone(),
                batch.targets_v.clone(),
                burn::nn::loss::Reduction::Mean,
            );

            let loss_p = CrossEntropyLossConfig::new()
                .init(&batch.targets_p.device())
                .forward(pred_policy.clone(), batch.targets_p.clone());

            let loss = loss_v.clone() + loss_p.clone();

            let predicted_1d = pred_policy.inner().argmax(1).reshape([actual_batch_size]);
            let targets_1d = batch.targets_p.inner().reshape([actual_batch_size]);
            let is_correct = predicted_1d.equal(targets_1d);

            train_correct += is_correct
                .into_data()
                .convert::<f32>()
                .iter::<f32>()
                .sum::<f32>() as usize;
            train_samples += actual_batch_size;

            total_loss_v += loss_v.into_scalar().to_f64();
            total_loss_p += loss_p.into_scalar().to_f64();

            let grads = GradientsParams::from_grads(loss.backward(), &current_model);
            current_model = optimizer.step(lr, current_model, grads);
            global_step += 1;

            let is_last_batch = batch_idx == num_batches - 1;

            // ======================================================
            // LOG VÀ CHẠY VALIDATION TRỰC TIẾP TRONG VÒNG LẶP
            // ======================================================
            if (batch_idx + 1) % valid_interval == 0 || is_last_batch {
                // Rút ruột model sang chế độ Valid (bỏ Autodiff)
                let val_model = current_model.valid();

                // Gọi hàm Validation
                let (val_v, val_p, val_acc) =
                    run_validation(&val_model, &val_data, batch_size, &device);

                let avg_loss_v = total_loss_v / (batch_idx + 1) as f64;
                let avg_loss_p = total_loss_p / (batch_idx + 1) as f64;
                let running_acc = (train_correct as f64 / train_samples as f64) * 100.0;

                // Dùng \x1B[2K để xóa sạch dòng trước khi in đè
                print!(
                    "\x1B[2K\r🔄 Ep {}/{} | Batch {}/{} | LR: {:.6}\n   ↳ Train [Acc: {:05.2}% | P: {:.4} | V: {:.4}]\n   ↳ Valid [Acc: {:05.2}% | P: {:.4} | V: {:.4}]",
                    epoch + 1,
                    num_epochs,
                    batch_idx + 1,
                    num_batches,
                    lr,
                    running_acc,
                    avg_loss_p,
                    avg_loss_v,
                    val_acc,
                    val_p,
                    val_v
                );
                let _ = std::io::stdout().flush();

                // Trả lại vị trí con trỏ lên 2 dòng để lần sau in đè mượt mà (chỉ khi không phải batch cuối)
                if !is_last_batch {
                    print!("\x1B[2A");
                }
            }
        }

        println!(
            "\n\n📋 EPOCH {} HOÀN TẤT ({:.1}s)",
            epoch + 1,
            epoch_start.elapsed().as_secs_f64()
        );
        println!("{}\n", separator);

        let save_path = format!("{}/xiangqi_net_epoch_{}", model_dir, epoch + 1);
        current_model
            .clone()
            .save_file(
                &save_path,
                &NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default(),
            )
            .unwrap();
    }

    let final_path = format!("{}/xiangqi_net_trained", model_dir);
    current_model
        .save_file(
            &final_path,
            &NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default(),
        )
        .unwrap();

    let _ = fs::remove_file(format!("{}.mpk", temp_transfer_path));
    println!("🎉 Hoàn tất quá trình huấn luyện!");
}
