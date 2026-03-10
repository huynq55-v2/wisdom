use burn::backend::wgpu::WgpuDevice;
use burn::record::NamedMpkFileRecorder;
use burn::{
    backend::{Autodiff, Wgpu},
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    optim::{AdamConfig, decay::WeightDecayConfig},
    prelude::*,
    record::Recorder, // BẮT BUỘC PHẢI IMPORT ĐỂ DÙNG HÀM .load()
    train::{Learner, SupervisedTraining, renderer::CliMetricsRenderer},
};
use crossbeam_channel::Sender;
use rand::RngExt;

use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
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

            // Lấy tọa độ gốc từ policy_idx trong Shard
            let from_dense = item.policy / 90;
            let to_dense = item.policy % 90;

            let mut from_r = from_dense / 9;
            let mut from_c = from_dense % 9;
            let mut to_r = to_dense / 9;
            let mut to_c = to_dense % 9;

            // 1. LẬT PERSPECTIVE NẾU LÀ ĐEN
            if board.side_to_move == Color::Black {
                from_r = 9 - from_r;
                from_c = 8 - from_c;
                to_r = 9 - to_r;
                to_c = 8 - to_c;
            }

            // 2. DATA AUGMENTATION: LẬT GƯƠNG NGANG
            if self.is_training && rng.random_bool(0.5) {
                // Lật gương Tensor ngang
                for plane in 0..14 {
                    for r in 0..10 {
                        for c in 0..4 {
                            let idx1 = plane * 90 + r * 9 + c;
                            let idx2 = plane * 90 + r * 9 + (8 - c);
                            tensor.swap(idx1, idx2);
                        }
                    }
                }
                // Lật gương tọa độ Policy ngang
                from_c = 8 - from_c;
                to_c = 8 - to_c;
            }

            // 3. ĐÓNG GÓI LẠI INDEX 8100
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
// 3. Self-Play Logic
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

    let alpha = 0.85; // Tỷ lệ giữ lại giá trị Search

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

// ==========================================================
// Hàm hỗ trợ nạp Shard từ ổ cứng vào RAM
// ==========================================================
fn load_all_shards(buffers_dir: &str, max_size: usize) -> Vec<SelfPlayItem> {
    let mut items = Vec::new();
    let mut shard_files: Vec<PathBuf> = Vec::new();

    // Thu thập tất cả file shard từ các thư mục con v...
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

    // Sắp xếp file theo thời gian sửa đổi (cũ đến mới)
    shard_files.sort_by_key(|path| {
        std::fs::metadata(path)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });

    // Nạp data
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

    // Cắt bớt nếu vượt quá max_size (chỉ giữ lại phần mới nhất ở cuối)
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

    // 1. TÌM VERSION MỚI NHẤT MỘT CÁCH GỌN GÀNG TỪ TÊN FILE
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

    // Base Model (Dùng cho MCTS) - Đã mang sẵn giá trị BatchNorm
    let mut model = config.init::<MyBackend>(&device).load_record(record);

    let num_iterations = 500;
    let games_per_iteration = 128;
    let concurrent_games = 128;
    let batch_size = 128;
    let max_buffer_size = 300_000;

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

            let eval_queue = EvalQueue::new(model.clone(), device.clone(), batch_size, 1);
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

            // Lưu Shard
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

            // Cắt tỉa RAM
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

        // --- CHUẨN BỊ DỮ LIỆU TRAIN & VALID ---
        use rand::seq::SliceRandom;
        let mut train_dataset = iter_records.lock().unwrap().clone();
        let mut rb_snapshot = replay_buffer_arc.lock().unwrap().clone();
        rb_snapshot.shuffle(&mut rand::rng());
        train_dataset.extend_from_slice(&rb_snapshot[0..50_000.min(rb_snapshot.len())]);
        train_dataset.shuffle(&mut rand::rng());

        println!(
            "\n🔥 Version {} - Training Model trên {} FENs",
            version,
            train_dataset.len()
        );

        let split_idx = (train_dataset.len() as f32 * 0.9) as usize;
        let dataloader_train = DataLoaderBuilder::new(XiangqiBatcher::new(device.clone(), true))
            .batch_size(1024)
            .shuffle(42)
            .num_workers(2)
            .set_device(device.clone())
            .build(RAMDataset {
                items: train_dataset[0..split_idx].to_vec(),
            });

        let dataloader_valid = DataLoaderBuilder::new(XiangqiBatcher::new(device.clone(), false))
            .batch_size(1024)
            .shuffle(42)
            .num_workers(2)
            .set_device(device.clone())
            .build(RAMDataset {
                items: train_dataset[split_idx..].to_vec(),
            });

        // 🔥 TRẢ LẠI LOGIC "CÂY CẦU" TEMP FILE CỦA BÁC (Chuẩn và An toàn nhất)
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

        let autodiff_model = config
            .init::<MyAutodiffBackend>(&device)
            .load_record(autodiff_record);

        let iter_learner_dir = format!("{}/learner_current", model_dir);

        use burn::grad_clipping::GradientClippingConfig;
        use burn::optim::lr_scheduler::constant::ConstantLr;

        // 🔥 KHỞI TẠO ADAM TƯƠI MỚI & SỬA LEARNING RATE
        let learner = Learner::new(
            autodiff_model,
            AdamConfig::new()
                .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
                .with_grad_clipping(Some(GradientClippingConfig::Value(1.0)))
                .init(),
            ConstantLr::new(5e-5),
        );

        // XÓA SẠCH CÁC METRIC CỦA BURN, CHỈ ĐỂ LẠI BỘ KHUNG TRAINING
        let training =
            SupervisedTraining::new(&iter_learner_dir, dataloader_train, dataloader_valid)
                .num_epochs(1)
                // Vẫn giữ Renderer để nó in ra thông báo Epoch
                .renderer(CliMetricsRenderer::new());

        // CHẠY TRAIN: Lúc này trên màn hình sẽ liên tục in ra các dòng "🚀 [Train Batch]" và "🎯 [Valid Batch]" do code tự viết ở nn.rs
        println!("\n▶️ BẮT ĐẦU TRAIN (TỰ TÍNH LOG CHUẨN)...");
        let trained_autodiff_model = training.launch(learner).model;
        println!("⏹️ KẾT THÚC TRAIN.\n");

        // 🔥 CẬP NHẬT TRỌNG SỐ MỚI QUA CÂY CẦU TEMP FILE
        trained_autodiff_model
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

        // XÓA TOÀN BỘ ĐOẠN read_log_metric VÀ IN TỔNG KẾT, VÌ CHÚNG TA ĐÃ IN TỪNG BATCH RỒI!

        let final_mpk_path = format!("{}/xiangqi_net_version_{}", model_dir, version);
        model
            .clone()
            .save_file(
                &final_mpk_path,
                &NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default(),
            )
            .unwrap();
        println!("✅ Đã lưu Model Version {} thành công!", version);
    }
}
