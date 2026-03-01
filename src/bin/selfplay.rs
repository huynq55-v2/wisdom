use burn::prelude::*;
use crossbeam_channel::Sender;
use rand::Rng;
use std::io::Write;
use wisdom::board::{Board, HistoryEntry};
use wisdom::eval_queue::{EvalQueue, EvalRequest};
use wisdom::nn::{TENSOR_SIZE, XiangqiNetConfig, board_to_tensor};
use wisdom::search::{MATE_VALUE, search_best_move_parallel};
use wisdom::tt::TranspositionTable;

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

fn play_game(eval_tx: &Sender<EvalRequest>) {
    let mut board = Board::new();
    let tt = TranspositionTable::new(32);
    let mut history = Vec::new(); // Store history of the game for repetitions
    let mut rng = rand::thread_rng();

    // Open a file to append our self-play data
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open("/tmp/selfplay_data.bin")
        .unwrap();

    let mut move_count = 0;

    loop {
        let legal_moves = get_all_legal_moves(&mut board);
        if legal_moves.is_empty() {
            println!("Game over: Checkmate or Stalemate. Length: {}", move_count);
            break;
        }

        // Max moves to prevent infinite loops if they just randomly shuffle
        if move_count > 400 {
            println!("Game over: Maximum moves reached.");
            break;
        }

        // 1. Search for best move and score
        let depth = 4;
        let (best_move, score) =
            search_best_move_parallel(&board, depth, &tt, &history, eval_tx, 8);

        // 2. Save the tensor and score
        let tensor = board_to_tensor(&board);
        // Normalize score to roughly -1.0 to 1.0 (clipping at 10000 material)
        let normalized_score = (score as f32 / 10000.0).clamp(-1.0, 1.0);

        let mut byte_buffer = Vec::with_capacity(TENSOR_SIZE * 4 + 4);
        for val in tensor.iter() {
            byte_buffer.extend_from_slice(&val.to_le_bytes());
        }
        byte_buffer.extend_from_slice(&normalized_score.to_le_bytes());
        file.write_all(&byte_buffer).unwrap();

        // 3. Epsilon-greedy: Choose move
        let mut chosen_move = best_move;

        // 100% certainty for checkmate moves, otherwise 90%/10% epsilon
        let is_mate = score > MATE_VALUE - 100 || score < -MATE_VALUE + 100;

        if !is_mate && rng.gen_bool(0.10) {
            // Pick a random move!
            let random_idx = rng.gen_range(0..legal_moves.len());
            chosen_move = legal_moves[random_idx];
            println!(
                "Turn {}: Evaluated {}, playing RANDOM move",
                move_count, score
            );
        } else {
            println!(
                "Turn {}: Evaluated {}, playing BEST move",
                move_count, score
            );
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
}

fn main() {
    type MyBackend = burn::backend::Wgpu;
    let device = burn::backend::wgpu::WgpuDevice::default();

    let config = XiangqiNetConfig::new();
    let model = config.init::<MyBackend>(&device);

    // Spawn evaluation thread
    let eval_queue = EvalQueue::new(model, device, 32, 5);
    let eval_tx = eval_queue.tx;

    println!("Starting self-play generation...");
    for game in 1..=5 {
        println!("--- Starting Game {} ---", game);
        play_game(&eval_tx);
    }
    println!("Self-play complete.");
}
