use burn::backend::wgpu::WgpuDevice;
use burn::record::NamedMpkFileRecorder;
use burn::record::Recorder;
use burn::prelude::*;
use burn::backend::Wgpu;
use crossbeam_channel::Sender;
use std::env;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::sync::{Arc, Mutex};

use wisdom::board::{Board, Color, HistoryEntry, PieceType, RepetitionResult};
use wisdom::mcts::MCTS;
use wisdom::eval_queue::{EvalQueue, EvalRequest};
use wisdom::nn::XiangqiNetConfig;
use wisdom::tt::TranspositionTable;
use wisdom::ucci::move_to_ucci_string;

type MyBackend = Wgpu<f32, i32>;

#[derive(Clone)]
pub struct ArenaConfig {
    pub num_games: usize,
    pub simulations: usize,
    pub pgn_out: String,
}

#[derive(Clone, Copy, PartialEq)]
enum GameResult {
    RedWin,
    BlackWin,
    Draw,
}

fn result_to_string(res: &GameResult) -> &'static str {
    match res {
        GameResult::RedWin => "1-0",
        GameResult::BlackWin => "0-1",
        GameResult::Draw => "1/2-1/2",
    }
}

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

fn play_arena_game(
    red_tx: &Sender<EvalRequest>,
    black_tx: &Sender<EvalRequest>,
    sims: usize,
) -> (GameResult, String) {
    let mut board = Board::new();
    board.set_initial_position();
    let mut history = Vec::new();
    let mut pgn_moves = String::new();
    let mcts = MCTS::new(100_000); 
    
    // Independent TT for each game!
    let tt = Arc::new(TranspositionTable::new(1024));

    let mut move_count = 0;
    let mut move_number = 1;

    loop {
        let legal_moves = get_all_legal_moves(&mut board);
        if legal_moves.is_empty() {
            if board.is_in_check(board.side_to_move) {
                return (
                    if board.side_to_move == Color::Red { GameResult::BlackWin } else { GameResult::RedWin },
                    pgn_moves
                );
            } else {
                return (GameResult::Draw, pgn_moves);
            }
        }

        if move_count > 400 {
            return (GameResult::Draw, pgn_moves);
        }

        let rep = board.judge_repetition(&history, history.len(), 1);
        match rep {
            RepetitionResult::Draw => return (GameResult::Draw, pgn_moves),
            RepetitionResult::Loss => return (
                if board.side_to_move == Color::Red { GameResult::BlackWin } else { GameResult::RedWin },
                pgn_moves
            ),
            RepetitionResult::Win => return (
                if board.side_to_move == Color::Red { GameResult::RedWin } else { GameResult::BlackWin },
                pgn_moves
            ),
            _ => {}
        }

        let current_tx = if board.side_to_move == Color::Red { red_tx } else { black_tx };
        
        let (best_move, _) = mcts.search_best_move(&board, &history, sims, current_tx, &tt, 1, false);

        if board.side_to_move == Color::Red {
            pgn_moves.push_str(&format!("{}. {} ", move_number, move_to_ucci_string(best_move)));
            move_number += 1;
        } else {
            pgn_moves.push_str(&format!("{} ", move_to_ucci_string(best_move)));
        }

        let is_capture = board.piece_at(best_move.to_sq()).is_some();
        let piece = board.piece_at(best_move.from_sq()).unwrap();
        let is_reversible = !is_capture && (piece.piece_type != PieceType::Pawn || {
            let (from_row, _) = Board::square_to_coord(best_move.from_sq());
            let (to_row, _) = Board::square_to_coord(best_move.to_sq());
            from_row == to_row
        });

        let current_side = board.side_to_move;
        let pre_threats = if is_reversible { board.get_unprotected_threats(current_side) } else { 0 };

        board.make_move(best_move);
        let gives_check = board.is_in_check(board.side_to_move);
        
        let chased_set = if is_reversible && !gives_check {
            let post_threats = board.get_unprotected_threats(current_side);
            post_threats & !pre_threats
        } else { 0 };

        history.push(HistoryEntry {
            hash: board.zobrist_key,
            is_check: gives_check,
            chased_set,
            is_reversible,
        });

        move_count += 1;
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Cách dùng: cargo run --bin arena --release <MODEL_V0_PATH> <MODEL_V1_PATH> <NUM_GAMES> [PGN_OUT_FILE]");
        std::process::exit(1);
    }

    let model_v0_path = &args[1];
    let model_v1_path = &args[2];
    let num_games: usize = args[3].parse().expect("NUM_GAMES phải là một số nguyên dương");
    let pgn_out = if args.len() > 4 { args[4].clone() } else { "arena_results.pgn".to_string() };

    println!("⚔️  Bắt đầu trận đấu: {} vs {}", model_v0_path, model_v1_path);
    println!("🎮 Tổng số ván đấu: {}", num_games);

    let device = WgpuDevice::default();
    let config = XiangqiNetConfig::new();

    println!("📥 Loading Model 0 from '{}'...", model_v0_path);
    let recorder_v0 = NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default();
    let record_v0 = Recorder::load(&recorder_v0, model_v0_path.into(), &device)
        .expect("Failed to load Model 0");
    let model_v0 = config.init::<MyBackend>(&device).load_record(record_v0);

    println!("📥 Loading Model 1 from '{}'...", model_v1_path);
    let recorder_v1 = NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::default();
    let record_v1 = Recorder::load(&recorder_v1, model_v1_path.into(), &device)
        .expect("Failed to load Model 1");
    let model_v1 = config.init::<MyBackend>(&device).load_record(record_v1);

    let cfg = ArenaConfig {
        num_games,
        simulations: 400,
        pgn_out: pgn_out.clone(),
    };

    let pgn_file = File::create(&cfg.pgn_out).expect("Không tạo được file PGN");
    let pgn_writer = Arc::new(Mutex::new(BufWriter::new(pgn_file)));

    let v0_wins = Arc::new(Mutex::new(0));
    let v1_wins = Arc::new(Mutex::new(0));
    let draws = Arc::new(Mutex::new(0));

    let queue_v0 = EvalQueue::new(model_v0.clone(), device.clone(), 64, 1);
    let queue_v1 = EvalQueue::new(model_v1.clone(), device.clone(), 64, 1);
    
    let concurrent_games = 16.min(num_games);
    let total_batches = (num_games + concurrent_games - 1) / concurrent_games;
    
    for batch_idx in 0..total_batches {
        let games_in_batch = std::cmp::min(
            concurrent_games,
            num_games - batch_idx * concurrent_games,
        );

        std::thread::scope(|s| {
            for inner_idx in 0..games_in_batch {
                let game_id = batch_idx * concurrent_games + inner_idx;
                let tx_v0 = &queue_v0.tx;
                let tx_v1 = &queue_v1.tx;
                let pgn_writer_clone = Arc::clone(&pgn_writer);
                
                let v0_w = Arc::clone(&v0_wins);
                let v1_w = Arc::clone(&v1_wins);
                let dr = Arc::clone(&draws);

                let cfg_clone = cfg.clone();

                s.spawn(move || {
                    let is_v1_red = game_id % 2 == 0;
                    let (red_tx, black_tx) = if is_v1_red {
                        (tx_v1, tx_v0)
                    } else {
                        (tx_v0, tx_v1)
                    };

                    println!("🎮 Trận {}/{} - V1 cầm {}: Đang thi đấu...", game_id + 1, cfg_clone.num_games, if is_v1_red {"Đỏ"} else {"Đen"});
                    
                    let (result, pgn_content) = play_arena_game(red_tx, black_tx, cfg_clone.simulations);

                    {
                        let mut f = pgn_writer_clone.lock().unwrap();
                        let red_player = if is_v1_red { "Model V1" } else { "Model V0" };
                        let black_player = if is_v1_red { "Model V0" } else { "Model V1" };

                        writeln!(f, "[Event \"Arena v0 vs v1\"]\n[Round \"{}\"]\n[White \"{}\"]\n[Black \"{}\"]\n[Result \"{}\"]\n{}\n", 
                            game_id + 1, red_player, black_player, result_to_string(&result), pgn_content.trim()).unwrap();
                        f.flush().unwrap();
                    }

                    let mut current_v0 = v0_w.lock().unwrap();
                    let mut current_v1 = v1_w.lock().unwrap();
                    let mut current_draws = dr.lock().unwrap();

                    match result {
                        GameResult::RedWin => if is_v1_red { *current_v1 += 1 } else { *current_v0 += 1 },
                        GameResult::BlackWin => if is_v1_red { *current_v0 += 1 } else { *current_v1 += 1 },
                        GameResult::Draw => *current_draws += 1,
                    }
                    
                    println!("🏁 Hoàn thành Trận {}. 📊 Tỉ số: V1: {} | V0: {} | Hòa: {}", 
                        game_id + 1, *current_v1, *current_v0, *current_draws);
                });
            }
        });
    }

    let final_v1 = *v1_wins.lock().unwrap();
    let final_v0 = *v0_wins.lock().unwrap();
    let final_draws = *draws.lock().unwrap();

    println!("=====================================================");
    println!("🏆 KẾT QUẢ CHUNG CUỘC TỪ {} VÁN", num_games);
    println!("🟢 Model V1 thắng: {}", final_v1);
    println!("🔴 Model V0 thắng: {}", final_v0);
    println!("⚪ Hòa: {}", final_draws);
    println!("=====================================================");
    println!("💾 Lịch sử các ván đấu đã được lưu tại {}", cfg.pgn_out);
}
