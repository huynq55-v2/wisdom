use burn::record::NamedMpkFileRecorder;
use burn::{
    backend::{Autodiff, Wgpu},
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    module::AutodiffModule,
    nn::loss::{CrossEntropyLossConfig, MseLoss},
    optim::AdamConfig,
    prelude::*,
    record::Recorder,
    train::{LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use crossbeam_channel::Sender;
use std::io::Write;
use std::sync::Mutex;
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

        for item in items {
            let mut board = Board::new();
            wisdom::ucci::parse_fen(&mut board, &item.fen);
            let tensor = board_to_tensor(&board);

            inputs_flat.extend_from_slice(&tensor);
            targets_v_flat.push(item.value);
            targets_p_flat.push(item.policy as i32);
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

/// Enum to track game outcome from the perspective of the side that just played.
#[derive(Debug, Clone, Copy)]
enum GameResult {
    /// The side to move has no legal moves and IS in check => they lost.
    /// This means the side that played the LAST move won.
    Win,
    Draw,
}

/// Play a single self-play game. Returns Vec<(FEN, value)> with ground-truth-blended values.
fn play_game(eval_tx: &Sender<EvalRequest>) -> Vec<SelfPlayItem> {
    let mut board = Board::new();
    board.set_initial_position();
    let tt = TranspositionTable::new(32);
    let mut history = Vec::new();

    // Store (fen, search_value, side_to_move_at_that_position, policy_index)
    let mut game_records: Vec<(String, f32, Color, usize)> = Vec::new();
    let mut move_count = 0;
    let mut result = GameResult::Draw;

    loop {
        let legal_moves = get_all_legal_moves(&mut board);
        if legal_moves.is_empty() {
            if board.is_in_check(board.side_to_move) {
                // Current side to move is mated => the side that played the last move won
                result = GameResult::Win;
            } else {
                // Stalemate => Draw
                result = GameResult::Draw;
            }
            break;
        }

        if move_count > 400 {
            result = GameResult::Draw;
            break;
        }

        // Check repetition before searching
        let rep = board.judge_repetition(&history, move_count, 1);
        match rep {
            wisdom::board::RepetitionResult::Draw
            | wisdom::board::RepetitionResult::Win
            | wisdom::board::RepetitionResult::Loss => {
                result = GameResult::Draw;
                break;
            }
            _ => {}
        }

        // Khởi tạo MCTS (Cây nhỏ 5000 node là đủ cho Self-play)
        let mcts = MCTS::new(5_000);

        // Chạy MCTS với 400 simulations và BẬT DIRICHLET NOISE (true)
        let simulations = 400;
        let (best_move, metrics) =
            mcts.search_best_move(&board, simulations, eval_tx, &tt, 1, true);

        // Quy đổi win_pct [0..100] về dải [-1.0 .. 1.0] làm Search Value
        let normalized_score = (metrics.win_pct / 50.0) - 1.0;
        let current_side = board.side_to_move;

        game_records.push((
            board.to_fen(),
            normalized_score,
            current_side,
            wisdom::nn::move_to_index(best_move),
        ));

        // ==== ĐÃ XÓA BỎ LOGIC EPSILON-GREEDY RANDOM ====
        let chosen_move = best_move;

        // ===== BUG FIX 1: Compute proper HistoryEntry =====
        let is_capture = board.piece_at(chosen_move.to_sq()).is_some();

        if is_capture {
            // Captures are never reversible in repetition logic
            board.make_move(chosen_move);
            let gives_check = board.is_in_check(board.side_to_move);

            history.push(HistoryEntry {
                hash: board.zobrist_key, // <-- Lấy hash sau khi make_move
                is_check: gives_check,
                chased_set: 0,
                is_reversible: false,
            });
        } else {
            // Quiet move: compute is_reversible, pre_threats, chased_set
            let piece = board.piece_at(chosen_move.from_sq()).unwrap();
            let is_reversible = piece.piece_type != PieceType::Pawn || {
                let (from_row, _) = Board::square_to_coord(chosen_move.from_sq());
                let (to_row, _) = Board::square_to_coord(chosen_move.to_sq());
                from_row == to_row
            };

            let pre_threats = if is_reversible {
                board.get_unprotected_threats(board.side_to_move)
            } else {
                0
            };

            board.make_move(chosen_move);

            let gives_check = board.is_in_check(board.side_to_move);

            let chased_set = if is_reversible && !gives_check {
                let post_threats = board.get_unprotected_threats(board.side_to_move.opposite());
                post_threats & !pre_threats
            } else {
                0
            };

            history.push(HistoryEntry {
                hash: board.zobrist_key, // <-- Lấy hash sau khi make_move
                is_check: gives_check,
                chased_set,
                is_reversible,
            });
        }

        move_count += 1;
    }

    // ===== BUG FIX 2: Backpropagate ground truth to all positions =====
    // Determine Z from the perspective of the side that played the LAST move
    // game_records stores (fen, search_val, side_to_move_at_that_position)
    // The last entry's side_to_move is the side that was about to play when the game ended.

    let final_items: Vec<SelfPlayItem> = match result {
        GameResult::Draw => {
            // All positions get blended value: 0.5 * search + 0.5 * 0.0
            game_records
                .into_iter()
                .map(|(fen, search_val, _side, policy)| SelfPlayItem {
                    fen,
                    value: search_val * 0.5, // blend search with draw (0.0)
                    policy,
                })
                .collect()
        }
        GameResult::Win => {
            // The side to move at the end of the game LOST (was checkmated).
            // So the losing side is board.side_to_move (current, after the game loop).
            let losing_side = board.side_to_move;

            game_records
                .into_iter()
                .map(|(fen, search_val, side, policy)| {
                    // Z from this position's perspective:
                    // If side == losing_side => Z = -1.0 (this position was losing)
                    // If side != losing_side => Z = +1.0 (this position was winning)
                    let z = if side == losing_side { -1.0 } else { 1.0 };

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

    // 2. SỬA BUG ĐƯỜNG DẪN: Lưu tất cả vào thư mục cục bộ thay vì /tmp
    let model_dir = "./wisdom_models";
    std::fs::create_dir_all(model_dir).expect("Failed to create models directory");

    let checkpoint_path = format!("{}/xiangqi_net_latest", model_dir);
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

    let num_iterations = 50;
    let games_per_iteration = 100;

    // 3. TỐI ƯU SỐ LUỒNG CHO CPU
    // Kaggle có khoảng 30 cores CPU. Đặt concurrent_games = 32 là lý tưởng nhất.
    // Nếu để 128 như trước, các thread sẽ tranh giành CPU gây ra hiệu ứng Context Switch cực kỳ chậm.
    let concurrent_games = 64; // GPU Iris Xe: tạo đủ áp lực nhồi đầy Batch Size 64

    // Khởi tạo Replay Buffer lưu tối đa khoảng 100,000 positions
    let mut replay_buffer: Vec<SelfPlayItem> = Vec::new();
    let max_buffer_size = 100_000;

    for iteration in 1..=num_iterations {
        println!("============================================================");
        println!(
            " Iteration {} / {} - Generating Data (CPU Mode)",
            iteration, num_iterations
        );
        println!("============================================================");

        let iteration_data = Mutex::new(Vec::new());

        // GENERATION PHASE
        // batch_size của CPU có thể để nhỏ (ví dụ 16 hoặc 32) để giảm đỗ trễ
        let eval_queue = EvalQueue::new(model.clone(), device.clone(), 64, 1); // Batch 64 cho GPU
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

            // Nhờ std::thread::scope, hàm này đã tự động chạy song song N ván cờ trên N nhân CPU!
            std::thread::scope(|s| {
                for game_i in 0..games_in_batch {
                    let tx = &eval_tx;
                    let data = &iteration_data;
                    s.spawn(move || {
                        let records = play_game(tx);
                        let len = records.len();
                        data.lock().unwrap().extend(records);
                        print!("g{}({}) ", game_i + 1, len);
                        let _ = std::io::stdout().flush();
                    });
                }
            });
            println!();
        }

        drop(eval_tx);
        drop(eval_queue);
        std::thread::sleep(std::time::Duration::from_millis(1000));

        let mut iteration_data = iteration_data.into_inner().unwrap();
        replay_buffer.append(&mut iteration_data);

        if replay_buffer.len() > max_buffer_size {
            let excess = replay_buffer.len() - max_buffer_size;
            replay_buffer.drain(0..excess);
        }

        // TRAINING PHASE
        println!("============================================================");
        println!(
            " Iteration {} / {} - Training Model on {} positions (Replay Buffer)",
            iteration,
            num_iterations,
            replay_buffer.len()
        );
        println!("============================================================");

        use rand::seq::SliceRandom;
        let mut train_dataset = replay_buffer.clone();
        train_dataset.shuffle(&mut rand::thread_rng());
        let split_idx = (train_dataset.len() as f32 * 0.9) as usize;
        let train_data = train_dataset[0..split_idx].to_vec();
        let valid_data = train_dataset[split_idx..].to_vec();

        let batcher_train = XiangqiBatcher::<MyAutodiffBackend>::new(device.clone());
        let batcher_valid = XiangqiBatcher::<MyBackend>::new(device.clone());

        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(32)
            .shuffle(42)
            .num_workers(1)
            .build(RAMDataset { items: train_data });

        let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
            .batch_size(32)
            .shuffle(42)
            .num_workers(1)
            .build(RAMDataset { items: valid_data });

        let optim = AdamConfig::new();

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

        // Đổi đường dẫn Learner về model_dir
        let learner = LearnerBuilder::new(model_dir)
            .with_file_checkpointer(burn::record::CompactRecorder::new())
            .devices(vec![device.clone()])
            .num_epochs(1)
            .build(autodiff_model, optim.init(), 1e-4);

        let trained_autodiff_model = learner.fit(dataloader_train, dataloader_valid);

        model = trained_autodiff_model.valid();

        // 4. CHECKPOINTING CHUẨN ĐƯỜNG DẪN
        model
            .clone()
            .save_file(&checkpoint_path, &burn::record::CompactRecorder::new())
            .expect("Failed to save model");

        // 5. LƯU MODEL DƯỚI DẠNG NATIVE MPK CHO ENGINE/GUI
        let final_mpk_path = format!("{}/xiangqi_net_weights", model_dir);
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
