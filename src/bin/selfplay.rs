use burn::record::NamedMpkFileRecorder;
use burn::{
    backend::{Autodiff, Wgpu},
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    module::AutodiffModule,
    nn::loss::{CrossEntropyLossConfig, MseLoss},
    optim::{AdamConfig, decay::WeightDecayConfig},
    prelude::*,
    record::Recorder,
    train::{LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use crossbeam_channel::Sender;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::{Arc, Mutex};
use wisdom::board::{Board, Color, HistoryEntry, PieceType};
use wisdom::eval_queue::{EvalQueue, EvalRequest};
use wisdom::mcts::MCTS;
use wisdom::nn::board_to_tensor;
use wisdom::nn::{BOARD_H, BOARD_W, NUM_PLANES, TENSOR_SIZE, XiangqiNet, XiangqiNetConfig};
use wisdom::tt::TranspositionTable;

// ==========================================================
// 1. Data Structures and Batcher
// ==========================================================

#[derive(Clone, Debug)]
pub struct SelfPlayItem {
    pub fen: String,
    pub value: f32,    // -1.0 to 1.0
    pub policy: usize, // best move index
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

#[derive(Debug)]
pub struct XiangqiBatch<B: Backend> {
    pub inputs: Tensor<B, 4>,
    pub targets_v: Tensor<B, 2>,
    pub targets_p: Tensor<B, 1, burn::tensor::Int>,
}

impl<B: Backend> Clone for XiangqiBatch<B> {
    fn clone(&self) -> Self {
        Self {
            inputs: self.inputs.clone(),
            targets_v: self.targets_v.clone(),
            targets_p: self.targets_p.clone(),
        }
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

impl<B: Backend> Batcher<SelfPlayItem, XiangqiBatch<B>> for XiangqiBatcher<B> {
    fn batch(&self, items: Vec<SelfPlayItem>) -> XiangqiBatch<B> {
        let batch_size = items.len();

        let mut inputs_flat = Vec::with_capacity(batch_size * TENSOR_SIZE);
        let mut targets_v_flat = Vec::with_capacity(batch_size);
        let mut targets_p_flat = Vec::with_capacity(batch_size);

        let mut rng = rand::thread_rng();

        for item in items {
            let mut board = Board::new();
            wisdom::ucci::parse_fen(&mut board, &item.fen);
            let mut tensor = board_to_tensor(&board);

            let mut policy_idx = item.policy; // Index thô tuyệt đối từ CSV
            let is_black = board.side_to_move == Color::Black;

            // ==========================================================
            // BƯỚC 1: CHUẨN HÓA GÓC NHÌN (CANONICAL) NẾU PHE ĐEN ĐI
            // ==========================================================
            if is_black {
                let from_dense = policy_idx / 90;
                let to_dense = policy_idx % 90;

                let f_row = from_dense / 9;
                let f_col = from_dense % 9;
                let t_row = to_dense / 9;
                let t_col = to_dense % 9;

                // Lật 180 độ (xoay mâm): hàng = 9 - hàng, cột = 8 - cột
                let new_from = (9 - f_row) * 9 + (8 - f_col);
                let new_to = (9 - t_row) * 9 + (8 - t_col);

                policy_idx = new_from * 90 + new_to;
            }

            // ==========================================================
            // BƯỚC 2: DATA AUGMENTATION (LẬT GƯƠNG NGANG 50-50)
            // Lật đối xứng trái-phải cho cả Tensor và Policy
            // ==========================================================
            use rand::Rng;
            if rng.gen_bool(0.5) {
                // A. Lật Tensor Bàn Cờ theo chiều ngang
                for plane in 0..14 {
                    for r in 0..10 {
                        for c in 0..4 {
                            // Chỉ chạy c đến 4 (một nửa bàn) để swap
                            let idx1 = plane * 90 + r * 9 + c;
                            let idx2 = plane * 90 + r * 9 + (8 - c);
                            tensor.swap(idx1, idx2);
                        }
                    }
                }

                // B. Lật Policy Index theo chiều ngang
                let from_dense = policy_idx / 90;
                let to_dense = policy_idx % 90;

                let f_row = from_dense / 9;
                let f_col = from_dense % 9;
                let t_row = to_dense / 9;
                let t_col = to_dense % 9;

                // Lật gương ngang: Chỉ lật cột (cột = 8 - cột), giữ nguyên hàng
                let flip_from = f_row * 9 + (8 - f_col);
                let flip_to = t_row * 9 + (8 - t_col);

                policy_idx = flip_from * 90 + flip_to;
            }

            inputs_flat.extend_from_slice(&tensor);
            targets_v_flat.push(item.value);
            targets_p_flat.push(policy_idx as i32);
        }

        let inputs = Tensor::<B, 1>::from_data(inputs_flat.as_slice(), &self.device)
            .reshape([batch_size, NUM_PLANES, BOARD_H, BOARD_W]);

        let targets_v = Tensor::<B, 1>::from_data(targets_v_flat.as_slice(), &self.device)
            .reshape([batch_size, 1]);

        let targets_p =
            Tensor::<B, 1, burn::tensor::Int>::from_data(targets_p_flat.as_slice(), &self.device);

        XiangqiBatch {
            inputs,
            targets_v,
            targets_p,
        }
    }
}

// ==========================================================
// 2. Training Steps Implementation
// ==========================================================

impl<B: burn::tensor::backend::AutodiffBackend> TrainStep<XiangqiBatch<B>, RegressionOutput<B>>
    for XiangqiNet<B>
{
    fn step(&self, batch: XiangqiBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let (pred_value, pred_policy) = self.forward(batch.inputs);

        let loss_v = MseLoss::new().forward(
            pred_value.clone(),
            batch.targets_v.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        let loss_p = CrossEntropyLossConfig::new()
            .init(&batch.targets_p.device())
            .forward(pred_policy.clone(), batch.targets_p.clone());

        let loss = loss_v + loss_p;

        TrainOutput::new(
            self,
            loss.backward(),
            RegressionOutput {
                loss,
                output: pred_value,
                targets: batch.targets_v,
            },
        )
    }
}

impl<B: Backend> ValidStep<XiangqiBatch<B>, RegressionOutput<B>> for XiangqiNet<B> {
    fn step(&self, batch: XiangqiBatch<B>) -> RegressionOutput<B> {
        let (pred_value, pred_policy) = self.forward(batch.inputs);

        let loss_v = MseLoss::new().forward(
            pred_value.clone(),
            batch.targets_v.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        let loss_p = CrossEntropyLossConfig::new()
            .init(&batch.targets_p.device())
            .forward(pred_policy.clone(), batch.targets_p.clone());

        let loss = loss_v + loss_p;

        RegressionOutput {
            loss,
            output: pred_value,
            targets: batch.targets_v,
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

/// Play a single self-play game. Returns Vec<(FEN, value)> with ground-truth-blended values.
fn play_game(eval_tx: &Sender<EvalRequest>, tt: &Arc<TranspositionTable>) -> Vec<SelfPlayItem> {
    let mut board = Board::new();
    board.set_initial_position();
    let mut history = Vec::new();

    // Store (fen, search_value, side_to_move_at_that_position, policy_index)
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

        // Check repetition before searching
        let rep = board.judge_repetition(&history, move_count, 1);
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

        // Chạy MCTS với 400 simulations và BẬT DIRICHLET NOISE (true)
        let simulations = 400;
        let (best_move, metrics) =
            mcts.search_best_move(&board, simulations, eval_tx, &tt, 1, true);

        // ==========================================
        // THÊM 2 DÒNG NÀY ĐỂ BÁO HIỆU ĐÃ TÌM XONG 1 NƯỚC
        print!(".");
        let _ = std::io::stdout().flush();
        // ==========================================

        // Quy đổi win_pct [0..100] về dải [-1.0 .. 1.0] làm Search Value
        let normalized_score = (metrics.win_pct / 50.0) - 1.0;
        let current_side = board.side_to_move;

        game_records.push((
            board.to_fen(),
            normalized_score,
            current_side,
            wisdom::nn::move_to_index(best_move), // Lưu chính xác Index tuyệt đối của nước đi!
        ));

        let chosen_move = best_move;

        // ===== TỐI ƯU LOGIC LUẬT CỜ =====
        // Rút gọn logic để tính HistoryEntry phục vụ luật Repetition cờ tướng
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

    // ===== BUG FIX 2: Backpropagate ground truth to all positions =====
    let final_items: Vec<SelfPlayItem> = match winner {
        None => {
            // TRƯỜNG HỢP HÒA (Draw)
            game_records
                .into_iter()
                .map(|(fen, search_val, _side, policy)| SelfPlayItem {
                    fen,
                    value: search_val * 0.5, // blend search với điểm hòa (0.0)
                    policy,
                })
                .collect()
        }
        Some(winning_color) => {
            // TRƯỜNG HỢP CÓ NGƯỜI CHIẾN THẮNG
            game_records
                .into_iter()
                .map(|(fen, search_val, side, policy)| {
                    // Nếu phe ở trạng thái này trùng với phe chiến thắng -> Z = +1.0 (Thắng)
                    // Ngược lại -> Z = -1.0 (Thua)
                    let z = if side == winning_color { 1.0 } else { -1.0 };

                    SelfPlayItem {
                        fen,
                        value: search_val * 0.5 + z * 0.5, // blend 50% search + 50% ground truth
                        policy,
                    }
                })
                .collect()
        }
    };

    final_items
}

// ==========================================================
// 4. Main Unified Pipeline Loop
// ==========================================================

fn main() {
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let config = XiangqiNetConfig::new();

    let model_dir = "./wisdom_models";
    std::fs::create_dir_all(model_dir).expect("Failed to create models directory");

    // TÌM CHECKPOINT MỚI NHẤT
    let mut start_iteration = 0;
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let fname = entry.file_name().into_string().unwrap_or_default();
            if fname.starts_with("xiangqi_net_ckpt_") {
                if let Some(num_str) = fname.strip_prefix("xiangqi_net_ckpt_") {
                    let num_part = num_str.split('.').next().unwrap_or("");
                    if let Ok(num) = num_part.parse::<usize>() {
                        start_iteration = start_iteration.max(num);
                    }
                }
            }
        }
    }

    let checkpoint_path = if start_iteration > 0 {
        format!("{}/xiangqi_net_ckpt_{}", model_dir, start_iteration)
    } else {
        format!("{}/xiangqi_net_latest", model_dir)
    };

    let temp_transfer_path = format!("{}/temp_transfer", model_dir);

    // CỐ GẮNG LOAD MODEL CŨ
    let record_result =
        burn::record::CompactRecorder::new().load(checkpoint_path.clone().into(), &device);

    let mut model = match record_result {
        Ok(record) => {
            println!(
                "✅ Found existing checkpoint! Loading model from '{}'...",
                checkpoint_path
            );
            config.init::<MyBackend>(&device).load_record(record)
        }
        Err(_) => {
            println!(
                "⚠️ No checkpoint found or failed to load. Initializing a NEW random model..."
            );
            config.init::<MyBackend>(&device)
        }
    };

    let num_iterations = 500;
    let games_per_iteration = 128;
    let concurrent_games = 128; // Tối ưu: Bằng đúng batch_size của EvalQueue

    // Khởi tạo TT 1 lần duy nhất, cấp 1024 MB (1 GB) dùng chung cho cả 128 luồng
    let shared_tt = Arc::new(TranspositionTable::new(1024));

    // CẤU TRÚC LƯU TRỮ CHIA SẺ (Thread-safe)
    let max_buffer_size = 2_000_000;
    let mut initial_buffer: Vec<SelfPlayItem> = Vec::new();
    let buffer_path = format!("{}/replay_buffer.csv", model_dir);

    // Nạp lại dữ liệu cũ nếu có
    if let Ok(file) = std::fs::File::open(&buffer_path) {
        println!("Đang nạp data từ {}...", buffer_path);
        let reader = BufReader::new(file);
        for line in reader.lines().map_while(|l| l.ok()) {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() == 3 {
                if let (Ok(value), Ok(policy)) =
                    (parts[1].parse::<f32>(), parts[2].parse::<usize>())
                {
                    initial_buffer.push(SelfPlayItem {
                        fen: parts[0].to_string(),
                        value,
                        policy,
                    });
                }
            }
        }
        println!(
            "Đã nạp xong {} bản ghi vào Replay Buffer.",
            initial_buffer.len()
        );
    }

    // Đưa Buffer vào Arc<Mutex> để các thread có thể ghi trực tiếp
    let replay_buffer_arc = Arc::new(Mutex::new(initial_buffer));

    // Mở file ở chế độ Ghi nối tiếp (Append) và đưa vào Arc<Mutex>
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&buffer_path)
        .expect("Không thể mở replay buffer file");
    let file_arc = Arc::new(Mutex::new(BufWriter::new(file)));

    for iter in 1..=num_iterations {
        let iteration = start_iteration + iter;
        println!("============================================================");
        println!(
            " Iteration {} / {} - Generating Data (CPU-MCTS + GPU-NN Batched)",
            iteration, num_iterations
        );
        println!("============================================================");

        // GENERATION PHASE
        let eval_queue = EvalQueue::new(model.clone(), device.clone(), 32, 1); // GPU chạy infer batch 64
        let eval_tx = eval_queue.tx.clone();

        let total_batches = (games_per_iteration + concurrent_games - 1) / concurrent_games;

        for batch_idx in 0..total_batches {
            let games_in_batch = std::cmp::min(
                concurrent_games,
                games_per_iteration - batch_idx * concurrent_games,
            );

            let start_game = batch_idx * concurrent_games + 1;
            println!(
                "  Batch {}/{}: games {}-{}... (Spawning {} CPU threads)",
                batch_idx + 1,
                total_batches,
                start_game,
                start_game + games_in_batch - 1,
                games_in_batch
            );
            std::io::stdout().flush().unwrap();

            std::thread::scope(|s| {
                for game_i in 0..games_in_batch {
                    let tx = &eval_tx;
                    let rb_clone = Arc::clone(&replay_buffer_arc);
                    let file_clone = Arc::clone(&file_arc);
                    let tt_clone = Arc::clone(&shared_tt);

                    s.spawn(move || {
                        // Chạy 1 ván cờ (Tốn thời gian)
                        let records = play_game(tx, &tt_clone);
                        let len = records.len();

                        // 1. CẬP NHẬT NGAY VÀO RAM BUFFER
                        {
                            let mut rb = rb_clone.lock().unwrap();
                            rb.extend(records.clone());
                        }

                        // 2. GHI NỐI TIẾP NGAY VÀO FILE Ổ CỨNG (Append)
                        {
                            let mut f = file_clone.lock().unwrap();
                            for item in &records {
                                let _ = writeln!(f, "{},{},{}", item.fen, item.value, item.policy);
                            }
                        }

                        print!("g{}({}) ", game_i + 1, len);
                        let _ = std::io::stdout().flush();
                    });
                }
            });
            println!();

            // XẢ TOÀN BỘ DATA CỦA BATCH XUỐNG Ổ CỨNG TRONG 1 LẦN DUY NHẤT
            {
                let mut f = file_arc.lock().unwrap();
                let _ = f.flush();
            }
        }

        drop(eval_tx);
        drop(eval_queue);
        std::thread::sleep(std::time::Duration::from_millis(1000));

        // KIỂM TRA VÀ CẮT TỈA (TRIM) BUFFER NẾU VƯỢT QUÁ GIỚI HẠN
        {
            let mut rb = replay_buffer_arc.lock().unwrap();
            if rb.len() > max_buffer_size {
                let excess = rb.len() - max_buffer_size;
                rb.drain(0..excess); // Xóa data cũ nhất ở đầu
                println!("Đã cắt tỉa {} bản ghi cũ khỏi Replay Buffer.", excess);

                // Vì đã xóa data cũ, ta phải Ghi đè (Overwrite) lại toàn bộ file CSV
                let mut f = file_arc.lock().unwrap();
                *f = BufWriter::new(std::fs::File::create(&buffer_path).unwrap());
                for item in rb.iter() {
                    let _ = writeln!(f, "{},{},{}", item.fen, item.value, item.policy);
                }
                let _ = f.flush();
            }
        }

        // TRAINING PHASE
        let dataset_snapshot = {
            let rb = replay_buffer_arc.lock().unwrap();
            rb.clone() // Clone ra để nhả Lock, giúp train không khóa mất biến
        };

        println!("============================================================");
        println!(
            " Iteration {} / {} - Training Model on {} positions",
            iteration,
            num_iterations,
            dataset_snapshot.len()
        );
        println!("============================================================");

        use rand::seq::SliceRandom;
        let mut train_dataset = dataset_snapshot;
        train_dataset.shuffle(&mut rand::thread_rng());
        let split_idx = (train_dataset.len() as f32 * 0.9) as usize;
        let train_data = train_dataset[0..split_idx].to_vec();
        let valid_data = train_dataset[split_idx..].to_vec();

        let batcher_train = XiangqiBatcher::<MyAutodiffBackend>::new(device.clone());
        let batcher_valid = XiangqiBatcher::<MyBackend>::new(device.clone());

        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(256) // Tùy chỉnh: có thể tăng lên 128/256 khi train
            .shuffle(42)
            .num_workers(2)
            .build(RAMDataset { items: train_data });

        let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
            .batch_size(256)
            .shuffle(42)
            .num_workers(2)
            .build(RAMDataset { items: valid_data });

        let optim = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1e-4)));

        // Lưu tạm Model
        model
            .clone()
            .save_file(&temp_transfer_path, &burn::record::CompactRecorder::new())
            .expect("Failed to save temp transfer model");

        let record = burn::record::CompactRecorder::new()
            .load(temp_transfer_path.clone().into(), &device)
            .expect("Failed to load temp transfer model");

        let autodiff_model = config
            .init::<MyAutodiffBackend>(&device)
            .load_record(record);

        // ĐÃ FIX BUG LEARNER: Tạo thư mục learner RIÊNG BIỆT cho từng iteration
        let iter_learner_dir = format!("{}/learner_iter_{}", model_dir, iteration);

        let learner = LearnerBuilder::new(&iter_learner_dir)
            .with_file_checkpointer(burn::record::CompactRecorder::new())
            .devices(vec![device.clone()])
            .num_epochs(1) // Chạy 1 Epoch cho mỗi Iteration
            .build(autodiff_model, optim.init(), 1e-4);

        let trained_autodiff_model = learner.fit(dataloader_train, dataloader_valid);

        model = trained_autodiff_model.valid();

        // Xóa thư mục Learner tạm để giải phóng ổ cứng
        let _ = std::fs::remove_dir_all(&iter_learner_dir);

        // 4. CHECKPOINTING CHUẨN ĐƯỜNG DẪN TÁCH RỜI TỪNG ITERATION
        let new_ckpt_path = format!("{}/xiangqi_net_ckpt_{}", model_dir, iteration);
        model
            .clone()
            .save_file(&new_ckpt_path, &burn::record::CompactRecorder::new())
            .expect("Failed to save model");

        // 5. LƯU MODEL DƯỚI DẠNG NATIVE MPK CHO ENGINE/GUI
        let final_mpk_path = format!("{}/xiangqi_net_{}", model_dir, iteration);
        model
            .clone()
            .save_file(
                &final_mpk_path,
                &NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default(),
            )
            .expect("Failed to save mpk model");

        println!(
            "Iteration {} completed. Model saved to {}.mpk",
            iteration, final_mpk_path
        );
    }

    println!("Unified Pipeline completely finished.");
}
