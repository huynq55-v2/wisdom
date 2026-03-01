use burn::{
    backend::{Autodiff, Wgpu},
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    module::AutodiffModule,
    nn::loss::MseLoss,
    optim::AdamConfig,
    prelude::*,
    record::Recorder,
    train::{LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use crossbeam_channel::Sender;
use rand::Rng;
use std::io::Write;
use wisdom::board::{Board, HistoryEntry};
use wisdom::eval_queue::{EvalQueue, EvalRequest};
use wisdom::nn::board_to_tensor;
use wisdom::nn::{BOARD_H, BOARD_W, NUM_PLANES, TENSOR_SIZE, XiangqiNet, XiangqiNetConfig};
use wisdom::search::{MATE_VALUE, search_best_move_parallel};
use wisdom::tt::TranspositionTable;

// ==========================================================
// 1. Data Structures and Batcher
// ==========================================================

#[derive(Clone, Debug)]
pub struct SelfPlayItem {
    pub fen: String,
    pub value: f32, // -1.0 to 1.0
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
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> Clone for XiangqiBatch<B> {
    fn clone(&self) -> Self {
        Self {
            inputs: self.inputs.clone(),
            targets: self.targets.clone(),
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
        let mut targets_flat = Vec::with_capacity(batch_size);

        for item in items {
            let mut board = Board::new();
            wisdom::ucci::parse_fen(&mut board, &item.fen);
            let tensor = board_to_tensor(&board);

            inputs_flat.extend_from_slice(&tensor);
            targets_flat.push(item.value);
        }

        let inputs = Tensor::<B, 1>::from_data(inputs_flat.as_slice(), &self.device)
            .reshape([batch_size, NUM_PLANES, BOARD_H, BOARD_W]);

        let targets = Tensor::<B, 1>::from_data(targets_flat.as_slice(), &self.device)
            .reshape([batch_size, 1]);

        XiangqiBatch { inputs, targets }
    }
}

// ==========================================================
// 2. Training Steps Implementation
// ==========================================================

impl<B: burn::tensor::backend::AutodiffBackend> TrainStep<XiangqiBatch<B>, RegressionOutput<B>>
    for XiangqiNet<B>
{
    fn step(&self, batch: XiangqiBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let predictions = self.forward(batch.inputs);

        let loss = MseLoss::new().forward(
            predictions.clone(),
            batch.targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        TrainOutput::new(
            self,
            loss.backward(),
            RegressionOutput {
                loss,
                output: predictions,
                targets: batch.targets,
            },
        )
    }
}

impl<B: Backend> ValidStep<XiangqiBatch<B>, RegressionOutput<B>> for XiangqiNet<B> {
    fn step(&self, batch: XiangqiBatch<B>) -> RegressionOutput<B> {
        let predictions = self.forward(batch.inputs);

        let loss = MseLoss::new().forward(
            predictions.clone(),
            batch.targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        RegressionOutput {
            loss,
            output: predictions,
            targets: batch.targets,
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

fn play_game(eval_tx: &Sender<EvalRequest>) -> Vec<SelfPlayItem> {
    let mut board = Board::new();
    board.set_initial_position();
    let tt = TranspositionTable::new(32);
    let mut history = Vec::new();
    let mut rng = rand::thread_rng();

    let mut game_records: Vec<SelfPlayItem> = Vec::new();
    let mut move_count = 0;

    loop {
        let legal_moves = get_all_legal_moves(&mut board);
        if legal_moves.is_empty() {
            if !board.is_in_check(board.side_to_move) {
                // Stalemate/Draw. Override last position's score to 0.0
                if let Some(last) = game_records.last_mut() {
                    last.value = 0.0;
                }
            }
            break;
        }

        if move_count > 400 {
            // Draw via move limit
            if let Some(last) = game_records.last_mut() {
                last.value = 0.0;
            }
            break;
        }

        // 1. Search for best move and score
        let depth = 4;
        let (best_move, score) =
            search_best_move_parallel(&board, depth, &tt, &history, eval_tx, 8);

        // Clamping automatically handles checkmate bounding (+/- 1.0)
        let normalized_score = (score as f32 / 10000.0).clamp(-1.0, 1.0);

        game_records.push(SelfPlayItem {
            fen: board.to_fen(),
            value: normalized_score,
        });

        // 2. Epsilon-greedy: Choose move
        let mut chosen_move = best_move;
        let is_mate = score > MATE_VALUE - 100 || score < -MATE_VALUE + 100;

        // 90% best move, 10% random legal move
        if !is_mate && rng.gen_bool(0.10) {
            let random_idx = rng.gen_range(0..legal_moves.len());
            chosen_move = legal_moves[random_idx];
        }

        // Apply move
        history.push(HistoryEntry {
            hash: board.zobrist_key,
            is_check: false,
            chased_set: 0,
            is_reversible: false,
        });

        board.make_move(chosen_move);
        move_count += 1;
    }

    game_records
}

// ==========================================================
// 4. Main Unified Pipeline Loop
// ==========================================================

fn main() {
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let config = XiangqiNetConfig::new();
    let mut model = config.init::<MyBackend>(&device);

    let num_iterations = 10;
    let games_per_iteration = 100; // Adjust to 10000 later for heavy cloud compute

    for iteration in 1..=num_iterations {
        println!("============================================================");
        println!(
            " Iteration {} / {} - Generating Data",
            iteration, num_iterations
        );
        println!("============================================================");

        let mut iteration_data = Vec::new();

        // 1. GENERATION PHASE
        // Create an active EvalQueue taking ownership of the model (GPU listener)
        let eval_queue = EvalQueue::new(model.clone(), device.clone(), 32, 5);
        let eval_tx = eval_queue.tx.clone();

        for game_idx in 1..=games_per_iteration {
            print!("Game {}/{}... ", game_idx, games_per_iteration);
            std::io::stdout().flush().unwrap();
            let mut records = play_game(&eval_tx);
            println!("Length: {} plies", records.len());
            iteration_data.append(&mut records);
        }

        // Destroy the eval queue thread to free the GPU for Burn's Learner
        drop(eval_tx);
        drop(eval_queue);
        std::thread::sleep(std::time::Duration::from_millis(500)); // wait for crossbeam thread cleanly exit

        // 2. TRAINING PHASE
        println!("============================================================");
        println!(
            " Iteration {} / {} - Training Model on {} positions",
            iteration,
            num_iterations,
            iteration_data.len()
        );
        println!("============================================================");

        // Shuffle and split train/valid sets (90% train, 10% valid)
        use rand::seq::SliceRandom;
        iteration_data.shuffle(&mut rand::thread_rng());
        let split_idx = (iteration_data.len() as f32 * 0.9) as usize;
        let train_data = iteration_data[0..split_idx].to_vec();
        let valid_data = iteration_data[split_idx..].to_vec();

        let batcher_train = XiangqiBatcher::<MyAutodiffBackend>::new(device.clone());
        let batcher_valid = XiangqiBatcher::<MyBackend>::new(device.clone());

        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(256)
            .shuffle(42)
            .num_workers(2)
            .build(RAMDataset { items: train_data });

        let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
            .batch_size(256)
            .shuffle(42)
            .num_workers(2)
            .build(RAMDataset { items: valid_data });

        let optim = AdamConfig::new();

        // Convert Wgpu model to Autodiff<Wgpu> for training module via File Save/Load
        model
            .clone()
            .save_file(
                "/tmp/wisdom_models/temp_transfer",
                &burn::record::CompactRecorder::new(),
            )
            .expect("Failed to save temp transfer model");

        let record = burn::record::CompactRecorder::new()
            .load("/tmp/wisdom_models/temp_transfer".into(), &device)
            .expect("Failed to load temp transfer model");

        let autodiff_model = config
            .init::<MyAutodiffBackend>(&device)
            .load_record(record);

        let learner = LearnerBuilder::new("/tmp/wisdom_models")
            .with_file_checkpointer(burn::record::CompactRecorder::new())
            .devices(vec![device.clone()])
            .num_epochs(1) // Run 1 epoch over the newly generated dataset per iteration
            .build(autodiff_model, optim.init(), 1e-4);

        // Re-train the model directly inside memory
        let trained_autodiff_model = learner.fit(dataloader_train, dataloader_valid);

        // Extract the raw backend model back to use in EvalQueue for the next iteration!
        model = trained_autodiff_model.valid();

        // 3. CHECKPOINTING
        model
            .clone()
            .save_file(
                "/tmp/wisdom_models/xiangqi_net_latest",
                &burn::record::CompactRecorder::new(),
            )
            .expect("Failed to save model");

        println!(
            "Iteration {} completed perfectly. Model checkpointed.",
            iteration
        );
    }

    println!("Unified Pipeline completely finished.");
}
