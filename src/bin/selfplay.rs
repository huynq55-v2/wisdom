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

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use wisdom::board::{Board, Color, HistoryEntry, PieceType};
use wisdom::eval_queue::{EvalQueue, EvalRequest};
use wisdom::mcts::MCTS;
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
}

impl<B: Backend> XiangqiBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
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

            let absolute_idx = item.policy; // 0..8099
            let is_black = board.side_to_move == Color::Black;

            let from_dense = absolute_idx / 90;
            let to_dense = absolute_idx % 90;

            let mut from_r = from_dense / 9;
            let mut from_c = from_dense % 9;
            let mut to_r = to_dense / 9;
            let mut to_c = to_dense % 9;

            // 1. LẬT PERSPECTIVE NẾU LÀ ĐEN
            if is_black {
                from_r = 9 - from_r;
                from_c = 8 - from_c;
                to_r = 9 - to_r;
                to_c = 8 - to_c;
            }

            // 2. DATA AUGMENTATION: LẬT GƯƠNG NGANG
            if rng.random_bool(0.5) {
                // Sửa .gen_bool() thành .random_bool() cho tương thích rand 0.9
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

            // 3. TÍNH ACTION SPACE 8100
            let from_sq90 = from_r * 9 + from_c;
            let to_sq90 = to_r * 9 + to_c;
            let policy_idx = from_sq90 * 90 + to_sq90;

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
            wisdom::nn::move_to_index(best_move), // Giữ nguyên Index tuyệt đối
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
    std::fs::create_dir_all(model_dir).expect("Failed to create models directory");

    let mut start_version = 0;
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let fname = entry.file_name().into_string().unwrap_or_default();
            if fname.starts_with("xiangqi_net_version_") && fname.ends_with(".mpk") {
                if let Some(num_str) = fname
                    .strip_prefix("xiangqi_net_version_")
                    .and_then(|s| s.strip_suffix(".mpk"))
                {
                    if let Ok(num) = num_str.parse::<usize>() {
                        start_version = start_version.max(num);
                    }
                }
            }
        }
    }

    let checkpoint_path = if start_version > 0 {
        format!("{}/xiangqi_net_version_{}.mpk", model_dir, start_version)
    } else {
        format!("{}/xiangqi_net_base.mpk", model_dir)
    };

    println!("📥 Loading base model from '{}'...", checkpoint_path);
    // SỬA LỖI RECORD LOAD Ở ĐÂY
    let record = NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default()
        .load(checkpoint_path.clone().into(), &device)
        .unwrap_or_else(|_| panic!("❌ Failed to load model from {}. Vui lòng đổi tên file Kaggle thành xiangqi_net_base.mpk", checkpoint_path));

    let mut model = config.init::<MyBackend>(&device).load_record(record);

    let num_iterations = 500;
    let games_per_iteration = 128;
    // TỐI ƯU TỐC ĐỘ: BẠN ĐÃ YÊU CẦU KO ĐỔI THÀNH 16, GIỮ NGUYÊN 128
    let concurrent_games = 128;
    let batch_size = 128;

    let shared_tt = Arc::new(TranspositionTable::new(1024));

    // --- SETUP SHARD DIRECTORY VÀ NẠP DATA ---
    let max_buffer_size = 300_000;
    let buffers_dir = format!("{}/buffers", model_dir);
    std::fs::create_dir_all(&buffers_dir).expect("Failed to create buffers directory");

    println!("Đang quét và nạp dữ liệu từ các file Shard...");
    let initial_buffer = load_all_shards(&buffers_dir, max_buffer_size);
    println!(
        "✅ Đã nạp xong {} bản ghi FEN vào Replay Buffer RAM.",
        initial_buffer.len()
    );

    let replay_buffer_arc = Arc::new(Mutex::new(initial_buffer));

    // Biến này theo dõi data được sinh ra bởi model version mấy
    let mut current_playing_version = start_version;

    for iter in 1..=num_iterations {
        let version = start_version + iter;

        // --- ĐOẠN CODE KIỂM TRA WARM-UP LƯỢT ĐẦU ---
        let initial_buf_size = replay_buffer_arc.lock().unwrap().len();
        let skip_gen = iter == 1 && initial_buf_size >= 40_000;

        if skip_gen {
            println!(
                "✅ Đã có sẵn {} FENs. Trực tiếp dùng data này để train version {}, thay vì gen data version 0...",
                initial_buf_size, version
            );
        }

        let iter_records = Arc::new(Mutex::new(Vec::new()));

        if !skip_gen {
            println!("============================================================");
            println!(
                " Version {} / {} - Generating Data (CPU-MCTS + GPU-NN Batched)",
                version, num_iterations
            );
            println!("============================================================");

            let eval_queue = EvalQueue::new(model.clone(), device.clone(), batch_size, 1);
            let eval_tx = eval_queue.tx.clone();

            let total_batches = (games_per_iteration + concurrent_games - 1) / concurrent_games;

            for batch_idx in 0..total_batches {
                let games_in_batch = std::cmp::min(
                    concurrent_games,
                    games_per_iteration - batch_idx * concurrent_games,
                );

                std::thread::scope(|s| {
                    for _ in 0..games_in_batch {
                        let tx = &eval_tx;
                        let rb_clone = Arc::clone(&replay_buffer_arc);
                        let tt_clone = Arc::clone(&shared_tt);
                        let iter_records_clone = Arc::clone(&iter_records);

                        s.spawn(move || {
                            let records = play_game(tx, &tt_clone);

                            // Đẩy vào RAM buffer để lấy mẫu Train
                            {
                                let mut rb = rb_clone.lock().unwrap();
                                rb.extend(records.clone());
                            }

                            // Gom vào iter_records để lát nữa đóng gói thành file Shard
                            {
                                let mut ir = iter_records_clone.lock().unwrap();
                                ir.extend(records);
                            }
                        });
                    }
                });
                println!();
            }

            drop(eval_tx);
            drop(eval_queue);
            std::thread::sleep(std::time::Duration::from_millis(1000));

            // --- TỰ ĐỘNG LƯU SHARD CHO ITERATION NÀY ---
            let ir_snapshot = {
                let ir = iter_records.lock().unwrap();
                ir.clone()
            };

            if !ir_snapshot.is_empty() {
                // Tạo thư mục ứng với version của Model ĐÃ CHƠI
                let current_v_dir = format!("{}/v{}", buffers_dir, current_playing_version);
                std::fs::create_dir_all(&current_v_dir).unwrap();

                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                let shard_path = format!("{}/shard_iter{}_{}.csv", current_v_dir, iter, timestamp);

                if let Ok(file) = std::fs::File::create(&shard_path) {
                    let mut writer = BufWriter::new(file);
                    for item in ir_snapshot.iter() {
                        let _ = writeln!(writer, "{},{},{}", item.fen, item.value, item.policy);
                    }
                    let _ = writer.flush();
                    println!(
                        "💾 Đã đóng gói và lưu {} FENs mới vào shard: {}",
                        ir_snapshot.len(),
                        shard_path
                    );
                }
            }

            // --- CẮT TỈA BUFFER RAM (KHÔNG ĐỤNG TỚI Ổ CỨNG NỮA) ---
            {
                let mut rb = replay_buffer_arc.lock().unwrap();
                if rb.len() > max_buffer_size {
                    let excess = rb.len() - max_buffer_size;
                    rb.drain(0..excess);
                    println!("Đã cắt tỉa {} bản ghi cũ khỏi RAM Replay Buffer.", excess);
                }
            }
        }

        // --- ĐOẠN CODE KIỂM TRA NGƯỠNG WARM-UP ---
        let current_buffer_size = {
            let rb = replay_buffer_arc.lock().unwrap();
            rb.len()
        };

        if current_buffer_size < 40_000 {
            println!(
                "⏳ Replay Buffer hiện có {} FENs. Đang tích lũy chờ đạt mốc 40,000 FENs mới bắt đầu Train...",
                current_buffer_size
            );
            continue;
        }
        // ----------------------------------------

        let mut train_dataset = {
            let ir = iter_records.lock().unwrap();
            ir.clone()
        };

        use rand::seq::SliceRandom;
        let mut rb_snapshot = {
            let rb = replay_buffer_arc.lock().unwrap();
            rb.clone()
        };
        rb_snapshot.shuffle(&mut rand::rng());

        let samples_from_buffer = 50_000.min(rb_snapshot.len());
        train_dataset.extend_from_slice(&rb_snapshot[0..samples_from_buffer]);

        train_dataset.shuffle(&mut rand::rng());

        println!("============================================================");
        println!(
            " Version {} - Training Model on {} sampled positions",
            version,
            train_dataset.len()
        );
        println!("============================================================");

        let split_idx = (train_dataset.len() as f32 * 0.9) as usize;
        let train_data = train_dataset[0..split_idx].to_vec();
        let valid_data = train_dataset[split_idx..].to_vec();

        let batcher_train = XiangqiBatcher::<MyAutodiffBackend>::new(device.clone());
        let batcher_valid = XiangqiBatcher::<MyBackend>::new(device.clone());

        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(256)
            .shuffle(42)
            .num_workers(2)
            .set_device(device.clone())
            .build(RAMDataset { items: train_data });

        let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
            .batch_size(256)
            .shuffle(42)
            .num_workers(2)
            .set_device(device.clone())
            .build(RAMDataset { items: valid_data });

        let temp_transfer_path = format!("{}/temp_transfer", model_dir);
        model
            .clone()
            .save_file(
                &temp_transfer_path,
                &NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default(),
            )
            .expect("Failed to save temp transfer model");

        // SỬA LỖI RECORD LOAD LẦN 2
        let record = NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default()
            .load(temp_transfer_path.clone().into(), &device)
            .expect("Failed to load temp transfer model");

        let autodiff_model = config
            .init::<MyAutodiffBackend>(&device)
            .load_record(record);

        let iter_learner_dir = format!("{}/learner_current", model_dir);

        use burn::optim::lr_scheduler::constant::ConstantLr;
        let learner = Learner::new(
            autodiff_model,
            AdamConfig::new()
                .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
                .init(),
            ConstantLr::new(1e-5),
        );

        let training =
            SupervisedTraining::new(&iter_learner_dir, dataloader_train, dataloader_valid)
                .metric_train_numeric(burn::train::metric::AccuracyMetric::new())
                .metric_valid_numeric(burn::train::metric::AccuracyMetric::new())
                .metric_train_numeric(burn::train::metric::LossMetric::new())
                .metric_valid_numeric(burn::train::metric::LossMetric::new())
                .num_epochs(1)
                .renderer(CliMetricsRenderer::new());

        let result = training.launch(learner);

        model = result.model;

        // BÂY GIỜ UPDATE CURRENT PLAYING VERSION THÀNH BẢN MỚI NHẤT
        current_playing_version = version;

        // --- ĐOẠN CODE MỚI THÊM: TRÍCH XUẤT KẾT QUẢ VALIDATION ---
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let loss_file = format!("{}/valid-loss.jsonl", iter_learner_dir);
        let acc_file = format!("{}/valid-accuracy.jsonl", iter_learner_dir);

        // Hàm đọc dòng cuối cùng của file jsonl để lấy value
        let get_last_value = |path: &str| -> Option<String> {
            let file = File::open(path).ok()?;
            let last_line = BufReader::new(file).lines().filter_map(|l| l.ok()).last()?;
            let parts: Vec<&str> = last_line.split("\"value\":").collect();
            if parts.len() > 1 {
                // Tách ngay tại dấu phẩy hoặc ngoặc nhọn để vứt bỏ các key JSON thừa phía sau
                let val_str = parts[1].split(|c| c == ',' || c == '}').next()?.trim();

                // Nếu là Accuracy, Burn lưu dưới dạng thập phân (ví dụ 0.1171), ta nhân 100 để in %
                if path.contains("accuracy") {
                    if let Ok(num) = val_str.parse::<f32>() {
                        return Some(format!("{:.2}%", num * 100.0));
                    }
                }

                Some(val_str.to_string())
            } else {
                None
            }
        };

        let final_loss = get_last_value(&loss_file).unwrap_or_else(|| "N/A".to_string());
        let final_acc = get_last_value(&acc_file).unwrap_or_else(|| "N/A".to_string());

        println!(
            "\n📊 TỔNG KẾT VERSION {}: Val Loss = {} | Val Acc = {}\n",
            version, final_loss, final_acc
        );
        // ----------------------------------------------------------

        let final_mpk_path = format!("{}/xiangqi_net_version_{}", model_dir, version);
        model
            .clone()
            .save_file(
                &final_mpk_path,
                &NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default(),
            )
            .expect("Failed to save mpk model");

        println!(
            "✅ Version {} completed. Model saved to {}.mpk",
            version, final_mpk_path
        );
    }

    println!("Unified Pipeline completely finished.");
}
