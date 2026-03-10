use burn::backend::wgpu::WgpuDevice;
use burn::module::AutodiffModule; // Thêm dòng này để dùng .valid()
use burn::record::NamedMpkFileRecorder;
use burn::{
    backend::{Autodiff, Wgpu},
    data::{
        dataloader::batcher::Batcher, // Đã bỏ DataLoader và Learner
        dataset::Dataset,
    },
    optim::{AdamConfig, GradientsParams, Optimizer, decay::WeightDecayConfig}, // Thêm Optimizer
    prelude::*,
    record::Recorder,
};
use crossbeam_channel::Sender;
use rand::RngExt;

use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use wisdom::board::{Board, Color, HistoryEntry, PieceType};
use wisdom::eval_queue::{EvalQueue, EvalRequest};
use wisdom::mcts::MCTS;
use wisdom::r#move::Move;
use wisdom::nn::board_to_tensor;
use wisdom::nn::{
    BOARD_H, BOARD_W, NUM_PLANES, TENSOR_SIZE, XiangqiNetConfig, XiangqiTrainingBatch,
};
use wisdom::tt::TranspositionTable;

// ==========================================================
// 1. Data Structures and Batcher
// ==========================================================

#[derive(Clone, Debug)]
pub struct SelfPlayItem {
    pub fen: String,
    pub value: f32,
    pub policy: usize,
}

pub struct RAMDataset {
    pub items: Vec<SelfPlayItem>,
}

impl Dataset<SelfPlayItem> for RAMDataset {
    fn get(&self, index: usize) -> Option<SelfPlayItem> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

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

impl<B: Backend> Batcher<B, SelfPlayItem, XiangqiTrainingBatch<B>> for XiangqiBatcher<B> {
    fn batch(&self, items: Vec<SelfPlayItem>, device: &B::Device) -> XiangqiTrainingBatch<B> {
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

            if board.side_to_move == Color::Black {
                from_r = 9 - from_r;
                from_c = 8 - from_c;
                to_r = 9 - to_r;
                to_c = 8 - to_c;
            }

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
            targets_v_flat.push(item.value);
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
// Hàm hỗ trợ Validation trực tiếp (Giống train.rs)
// ==========================================================
fn run_validation<B: Backend>(
    model: &wisdom::nn::XiangqiNet<B>,
    val_data: &[SelfPlayItem],
    batch_size: usize,
    device: &B::Device,
) -> (f64, f64, f64) {
    let batcher_val = XiangqiBatcher::<B>::new(device.clone(), false);

    // Vì val_data được set đúng 256 mẫu, nó sẽ chỉ chạy đúng 1 Batch
    let batch = batcher_val.batch(val_data.to_vec(), device);
    let actual_batch_size = val_data.len();

    let (pred_value, pred_policy) = model.forward(batch.inputs);

    let loss_v = burn::nn::loss::MseLoss::new().forward(
        pred_value,
        batch.targets_v,
        burn::nn::loss::Reduction::Mean,
    );

    let loss_p = burn::nn::loss::CrossEntropyLossConfig::new()
        .init(&batch.targets_p.device())
        .forward(pred_policy.clone(), batch.targets_p.clone());

    let val_loss_v = loss_v.into_scalar().to_f64();
    let val_loss_p = loss_p.into_scalar().to_f64();

    let predicted = pred_policy.argmax(1).reshape([actual_batch_size]);
    let targets = batch.targets_p.reshape([actual_batch_size]);
    let is_correct = predicted.equal(targets);
    let correct_count: f32 = is_correct.into_data().convert::<f32>().iter::<f32>().sum();

    let val_accuracy = (correct_count as f64 / actual_batch_size as f64) * 100.0;

    (val_loss_v, val_loss_p, val_accuracy)
}

// ==========================================================
// 2. Self-Play Logic (Giữ nguyên)
// ==========================================================
fn get_all_legal_moves(board: &mut Board) -> Vec<wisdom::r#move::Move> {
    let mut all_moves = board.generate_captures();
    all_moves.append(&mut board.generate_quiets());
    let mut legal_moves = Vec::new();
    let moving_side = board.side_to_move;

    for m in all_moves {
        let undo = board.make_move(m);
        let legal = !board.kings_facing() && !board.is_in_check(moving_side);
        board.unmake_move(m, undo);
        if legal {
            legal_moves.push(m);
        }
    }
    legal_moves
}

fn play_game(eval_tx: &Sender<EvalRequest>, tt: &Arc<TranspositionTable>) -> Vec<SelfPlayItem> {
    let mut board = Board::new();
    board.set_initial_position();
    let mut history = Vec::new();

    let mut game_records: Vec<(String, f32, Color, usize)> = Vec::new();
    let mut move_count = 0;
    let winner: Option<Color>;

    let mcts = MCTS::new(50_000);

    loop {
        let legal_moves = get_all_legal_moves(&mut board);
        if legal_moves.is_empty() {
            if board.is_in_check(board.side_to_move) {
                winner = Some(board.side_to_move.opposite());
            } else {
                winner = None;
            }
            break;
        }

        if move_count > 400 {
            winner = None;
            break;
        }

        let rep = board.judge_repetition(&history, history.len(), 1);
        match rep {
            wisdom::board::RepetitionResult::Draw => {
                winner = None;
                break;
            }
            wisdom::board::RepetitionResult::Loss => {
                winner = Some(board.side_to_move.opposite());
                break;
            }
            wisdom::board::RepetitionResult::Win => {
                winner = Some(board.side_to_move);
                break;
            }
            _ => {}
        }

        let simulations = 400;
        let (best_move, metrics) =
            mcts.search_best_move(&board, &history, simulations, eval_tx, &tt, 1, true);

        print!(".");
        let _ = std::io::stdout().flush();

        let normalized_score = (metrics.win_pct / 50.0) - 1.0;
        let current_side = board.side_to_move;

        game_records.push((
            board.to_fen(),
            normalized_score,
            current_side,
            Move::move_to_nn_index(best_move, current_side == Color::Black),
        ));

        let chosen_move = best_move;
        let is_capture = board.piece_at(chosen_move.to_sq()).is_some();
        let piece = board.piece_at(chosen_move.from_sq()).unwrap();

        let is_reversible = !is_capture
            && (piece.piece_type != PieceType::Pawn || {
                let (from_row, _) = Board::square_to_coord(chosen_move.from_sq());
                let (to_row, _) = Board::square_to_coord(chosen_move.to_sq());
                from_row == to_row
            });

        let pre_threats = if is_reversible {
            board.get_unprotected_threats(current_side)
        } else {
            0
        };

        board.make_move(chosen_move);
        let gives_check = board.is_in_check(board.side_to_move);

        let chased_set = if is_reversible && !gives_check {
            let post_threats = board.get_unprotected_threats(current_side);
            post_threats & !pre_threats
        } else {
            0
        };

        history.push(HistoryEntry {
            hash: board.zobrist_key,
            is_check: gives_check,
            chased_set,
            is_reversible,
        });

        move_count += 1;
    }

    let alpha = 0.85;

    let final_items: Vec<SelfPlayItem> = match winner {
        None => game_records
            .into_iter()
            .map(|(fen, search_val, _side, policy)| SelfPlayItem {
                fen,
                value: search_val * alpha,
                policy,
            })
            .collect(),
        Some(winning_color) => game_records
            .into_iter()
            .map(|(fen, search_val, side, policy)| {
                let z = if side == winning_color { 1.0 } else { -1.0 };
                SelfPlayItem {
                    fen,
                    value: search_val * alpha + z * (1.0 - alpha),
                    policy,
                }
            })
            .collect(),
    };

    final_items
}

fn load_all_shards(buffers_dir: &str, max_size: usize) -> Vec<SelfPlayItem> {
    let mut items = Vec::new();
    let mut shard_files: Vec<PathBuf> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(buffers_dir) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                if let Ok(sub_entries) = std::fs::read_dir(entry.path()) {
                    for sub_entry in sub_entries.flatten() {
                        if sub_entry.path().extension().and_then(|s| s.to_str()) == Some("csv") {
                            shard_files.push(sub_entry.path());
                        }
                    }
                }
            }
        }
    }

    shard_files.sort_by_key(|path| {
        std::fs::metadata(path)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });

    for path in shard_files {
        if let Ok(file) = std::fs::File::open(&path) {
            let reader = BufReader::new(file);
            for line in reader.lines().map_while(|l| l.ok()) {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() == 3 {
                    if let (Ok(value), Ok(policy)) =
                        (parts[1].parse::<f32>(), parts[2].parse::<usize>())
                    {
                        items.push(SelfPlayItem {
                            fen: parts[0].to_string(),
                            value,
                            policy,
                        });
                    }
                }
            }
        }
    }

    if items.len() > max_size {
        let excess = items.len() - max_size;
        items.drain(0..excess);
    }
    items
}

// ==========================================================
// 4. Main Unified Pipeline Loop
// ==========================================================

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = WgpuDevice::default();
    let config = XiangqiNetConfig::new();
    let model_dir = "./wisdom_models";
    let buffers_dir = format!("{}/buffers", model_dir);

    fs::create_dir_all(&buffers_dir).expect("Failed to create dirs");

    let start_version = fs::read_dir(model_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter_map(|name| {
            name.strip_prefix("xiangqi_net_version_")?
                .strip_suffix(".mpk")?
                .parse::<usize>()
                .ok()
        })
        .max()
        .unwrap_or(0);

    let checkpoint_path = if start_version > 0 {
        format!("{}/xiangqi_net_version_{}.mpk", model_dir, start_version)
    } else {
        format!("{}/xiangqi_net_base.mpk", model_dir)
    };

    println!("📥 Đang nạp model từ '{}'...", checkpoint_path);
    let record = NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default()
        .load(checkpoint_path.clone().into(), &device)
        .expect("❌ Không tìm thấy base model!");

    let mut model = config.init::<MyBackend>(&device).load_record(record);

    let num_iterations = 500;
    let games_per_iteration = 128;
    let concurrent_games = 128;
    let max_buffer_size = 300_000;

    // --- THÔNG SỐ TRAINING TÙY CHỈNH CỦA BÁC ---
    let batch_size = 256;
    let valid_interval = 20; // 🎯 BÁC SỬA SỐ n Ở ĐÂY (Valid sau mỗi n batch)
    let lr_max = 1e-5; // LR ổn định cho Self-Play Fine-tuning

    let shared_tt = Arc::new(TranspositionTable::new(1024));

    println!("Đang nạp dữ liệu từ Shard...");
    let initial_buffer = load_all_shards(&buffers_dir, max_buffer_size);
    println!("✅ Đã nạp {} FENs vào RAM.", initial_buffer.len());

    let replay_buffer_arc = Arc::new(Mutex::new(initial_buffer));
    let mut current_playing_version = start_version;

    for iter in 1..=num_iterations {
        let version = start_version + iter;
        let initial_buf_size = replay_buffer_arc.lock().unwrap().len();
        let skip_gen = iter == 1 && initial_buf_size >= 40_000;

        let iter_records = Arc::new(Mutex::new(Vec::new()));

        if skip_gen {
            println!(
                "✅ Có sẵn {} FENs. Bỏ qua MCTS, train version {}...",
                initial_buf_size, version
            );
        } else {
            println!("\n============================================================");
            println!(
                " Version {} / {} - Generating Data (CPU-MCTS + GPU-NN)",
                version, num_iterations
            );
            println!("============================================================");

            // Khởi tạo Model
            let onnx_model = wisdom::nn::XiangqiOnnx::new("./wisdom_models/xiangqi_model.onnx");

            // Khởi tạo Queue (Bỏ cái device đi)
            let eval_queue = EvalQueue::new(onnx_model, batch_size, 1);
            let total_batches = (games_per_iteration + concurrent_games - 1) / concurrent_games;

            for batch_idx in 0..total_batches {
                let games_in_batch = std::cmp::min(
                    concurrent_games,
                    games_per_iteration - batch_idx * concurrent_games,
                );
                std::thread::scope(|s| {
                    for _ in 0..games_in_batch {
                        let tx = &eval_queue.tx;
                        let rb_clone = Arc::clone(&replay_buffer_arc);
                        let tt_clone = Arc::clone(&shared_tt);
                        let ir_clone = Arc::clone(&iter_records);

                        s.spawn(move || {
                            let records = play_game(tx, &tt_clone);
                            rb_clone.lock().unwrap().extend(records.clone());
                            ir_clone.lock().unwrap().extend(records);
                        });
                    }
                });
            }

            let ir_snapshot = iter_records.lock().unwrap().clone();
            if !ir_snapshot.is_empty() {
                let current_v_dir = format!("{}/v{}", buffers_dir, current_playing_version);
                fs::create_dir_all(&current_v_dir).unwrap();
                let shard_path = format!(
                    "{}/shard_iter{}_{}.csv",
                    current_v_dir,
                    iter,
                    std::time::UNIX_EPOCH.elapsed().unwrap().as_secs()
                );

                if let Ok(file) = fs::File::create(&shard_path) {
                    let mut writer = BufWriter::new(file);
                    for item in &ir_snapshot {
                        let _ = writeln!(writer, "{},{},{}", item.fen, item.value, item.policy);
                    }
                }
            }

            let mut rb = replay_buffer_arc.lock().unwrap();
            if rb.len() > max_buffer_size {
                let excess = rb.len() - max_buffer_size;
                rb.drain(0..excess);
            }
        }

        let current_buffer_size = replay_buffer_arc.lock().unwrap().len();
        if current_buffer_size < 40_000 {
            println!(
                "⏳ Replay Buffer có {} FENs. Chờ đạt 40_000...",
                current_buffer_size
            );
            continue;
        }

        // ==========================================================
        // KHÂU CHUẨN BỊ DỮ LIỆU & TRAINING LOOP (VIẾT LẠI MỚI)
        // ==========================================================
        use rand::seq::SliceRandom;
        let mut train_dataset = iter_records.lock().unwrap().clone();
        let mut rb_snapshot = replay_buffer_arc.lock().unwrap().clone();
        rb_snapshot.shuffle(&mut rand::rng());
        train_dataset.extend_from_slice(&rb_snapshot[0..50_000.min(rb_snapshot.len())]);
        train_dataset.shuffle(&mut rand::rng());

        // 🎯 Cắt đúng 256 FEN cuối cùng làm Validation Set
        let val_size = 256.min(train_dataset.len());
        let train_size = train_dataset.len() - val_size;

        let mut train_data = train_dataset[..train_size].to_vec();
        let val_data = train_dataset[train_size..].to_vec();

        println!(
            "\n🔥 Version {} - Training trên {} FENs | Validation: {} FENs",
            version, train_size, val_size
        );

        // Chuyển Model qua nhánh Autodiff
        let temp_transfer_path = format!("{}/temp_transfer", model_dir);
        model
            .clone()
            .save_file(
                &temp_transfer_path,
                &NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default(),
            )
            .unwrap();
        let autodiff_record =
            NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default()
                .load(temp_transfer_path.clone().into(), &device)
                .unwrap();
        let mut autodiff_model = config
            .init::<MyAutodiffBackend>(&device)
            .load_record(autodiff_record);

        // Setup Optimizer
        use burn::nn::loss::{CrossEntropyLossConfig, MseLoss};
        let optimizer_config =
            AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1e-4)));
        let mut optimizer =
            optimizer_config.init::<MyAutodiffBackend, wisdom::nn::XiangqiNet<MyAutodiffBackend>>();

        let num_batches = (train_data.len() + batch_size - 1) / batch_size;
        let batcher_train = XiangqiBatcher::<MyAutodiffBackend>::new(device.clone(), true);

        let mut total_loss_v = 0.0f64;
        let mut total_loss_p = 0.0f64;
        let mut train_correct: usize = 0;
        let mut train_samples: usize = 0;

        println!("\n============================================================");
        println!("▶️ BẮT ĐẦU TRAINING (Manual Loop)");
        println!("============================================================\n");

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = std::cmp::min(start + batch_size, train_data.len());
            let batch_items = train_data[start..end].to_vec();
            let actual_batch_size = batch_items.len();

            let batch = batcher_train.batch(batch_items, &device);
            let (pred_value, pred_policy) = autodiff_model.forward(batch.inputs);

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

            // Backward & Step
            let grads = GradientsParams::from_grads(loss.backward(), &autodiff_model);
            autodiff_model = optimizer.step(lr_max, autodiff_model, grads);

            let is_last_batch = batch_idx == num_batches - 1;

            // 🎯 KIỂM TRA VALIDATION SAU N BATCH
            if (batch_idx + 1) % valid_interval == 0 || is_last_batch {
                let val_model = autodiff_model.valid(); // Bóc Autodiff để chạy Valid
                let (val_v, val_p, val_acc) =
                    run_validation(&val_model, &val_data, batch_size, &device);

                let avg_loss_v = total_loss_v / (batch_idx + 1) as f64;
                let avg_loss_p = total_loss_p / (batch_idx + 1) as f64;
                let running_acc = (train_correct as f64 / train_samples as f64) * 100.0;

                print!(
                    "\x1B[2K\r🔄 Model v{} | Batch {}/{} | LR: {:.6}\n   ↳ Train [Acc: {:05.2}% | P: {:.4} | V: {:.4}]\n   ↳ Valid [Acc: {:05.2}% | P: {:.4} | V: {:.4}]",
                    version,
                    batch_idx + 1,
                    num_batches,
                    lr_max,
                    running_acc,
                    avg_loss_p,
                    avg_loss_v,
                    val_acc,
                    val_p,
                    val_v
                );
                let _ = std::io::stdout().flush();

                if !is_last_batch {
                    print!("\x1B[2A"); // Giật lùi con trỏ lên 2 dòng
                }
            }
        }

        println!("\n\n⏹️ KẾT THÚC TRAIN.\n");

        // CẬP NHẬT TRỌNG SỐ CHO BASE MODEL ĐỂ CHUẨN BỊ LẶP LẠI SELF-PLAY
        autodiff_model
            .save_file(
                &temp_transfer_path,
                &NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default(),
            )
            .unwrap();
        let new_record = NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default()
            .load(temp_transfer_path.into(), &device)
            .unwrap();

        model = model.load_record(new_record);
        current_playing_version = version;

        let final_mpk_path = format!("{}/xiangqi_net_version_{}", model_dir, version);
        model
            .clone()
            .save_file(
                &final_mpk_path,
                &NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default(),
            )
            .unwrap();
        println!("✅ Đã lưu Model Version {} thành công!\n", version);
    }
}
