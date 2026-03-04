use crate::board::{Board, Color, Piece, PieceType};
use crate::eval_queue::EvalQueue;
use crate::mcts::MCTS;
use crate::r#move::Move;

use std::io::{self, BufRead};

pub fn move_to_ucci_string(m: Move) -> String {
    let (from_row, from_col) = Board::square_to_coord(m.from_sq());
    let (to_row, to_col) = Board::square_to_coord(m.to_sq());

    // UCCI files (columns) from left to right: a to i
    let from_file = (b'a' + from_col as u8) as char;
    let to_file = (b'a' + to_col as u8) as char;

    // UCCI ranks from bottom to top: 0 to 9.
    // In our board, Red is bottom (rows 6-9 in array), Black is top (rows 0-3).
    // So row 9 -> rank 0, row 8 -> rank 1... row 0 -> rank 9.
    let from_rank = (b'9' - from_row as u8) as char;
    let to_rank = (b'9' - to_row as u8) as char;

    format!("{}{}{}{}", from_file, from_rank, to_file, to_rank)
}

pub fn parse_ucci_move(board: &Board, ucci: &str) -> Option<Move> {
    if ucci.len() != 4 {
        return None;
    }

    let chars: Vec<char> = ucci.chars().collect();
    let from_file = chars[0] as u8;
    let from_rank = chars[1] as u8;
    let to_file = chars[2] as u8;
    let to_rank = chars[3] as u8;

    if from_file < b'a' || from_file > b'i' || to_file < b'a' || to_file > b'i' {
        return None;
    }
    if from_rank < b'0' || from_rank > b'9' || to_rank < b'0' || to_rank > b'9' {
        return None;
    }

    let from_col = (from_file - b'a') as usize;
    let from_row = (b'9' - from_rank) as usize;

    let to_col = (to_file - b'a') as usize;
    let to_row = (b'9' - to_rank) as usize;

    let from_sq = Board::coord_to_square(from_row, from_col);
    let to_sq = Board::coord_to_square(to_row, to_col);

    let is_capture = !board.is_empty(to_sq);

    Some(Move::new(from_sq, to_sq, is_capture))
}

pub fn parse_fen(board: &mut Board, fen: &str) {
    *board = Board::new(); // reset

    let parts: Vec<&str> = fen.split_whitespace().collect();
    if parts.is_empty() {
        return;
    }

    let position = parts[0];
    let mut row = 0;
    let mut col = 0;

    for c in position.chars() {
        if c == '/' {
            row += 1;
            col = 0;
        } else if c.is_digit(10) {
            col += c.to_digit(10).unwrap() as usize;
        } else {
            let color = if c.is_uppercase() {
                Color::Red
            } else {
                Color::Black
            };
            let piece_type = match c.to_ascii_lowercase() {
                'k' => PieceType::King,
                'a' => PieceType::Advisor,
                'b' | 'e' => PieceType::Elephant,
                'n' | 'h' => PieceType::Horse,
                'r' => PieceType::Rook,
                'c' => PieceType::Cannon,
                'p' => PieceType::Pawn,
                _ => continue,
            };

            let sq = Board::coord_to_square(row, col);
            board.set_piece(sq, Some(Piece::new(piece_type, color)));

            if piece_type == PieceType::King {
                if color == Color::Red {
                    board.red_king_sq = sq;
                } else {
                    board.black_king_sq = sq;
                }
            }
            col += 1;
        }
    }

    if parts.len() > 1 {
        board.side_to_move = if parts[1] == "b" || parts[1] == "B" {
            Color::Black
        } else {
            Color::Red
        };
    }
    board.compute_zobrist_key();
}

pub fn ucci_loop_generic<B: burn::prelude::Backend>(
    model: crate::nn::XiangqiNet<B>,
    device: B::Device,
) {
    let mut board = Board::new();
    board.set_initial_position();

    let tt = crate::tt::TranspositionTable::new(64);

    // Start NN Eval Queue (automatically spawns background thread)
    let eval_queue = EvalQueue::new(model, device, 32, 5);
    // Store transmitter for MCTS
    let eval_tx = eval_queue.tx.clone();

    let mcts = MCTS::new(2_000_000); // Pre-allocate 100k nodes

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap_or_default();
        let tokens: Vec<&str> = line.trim().split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        let cmd = tokens[0];

        match cmd {
            "ucci" => {
                println!("id name Wisdom Engine");
                println!("id author Quang Huy");
                println!("ucciok");
            }
            "isready" => {
                println!("readyok");
            }
            "position" => {
                if tokens.len() > 1 {
                    let mut moves_idx = 0;
                    if tokens[1] == "startpos" {
                        board.set_initial_position();
                        moves_idx = 2;
                    } else if tokens[1] == "fen" && tokens.len() >= 8 {
                        let fen = format!(
                            "{} {} {} {} {} {}",
                            tokens[2], tokens[3], tokens[4], tokens[5], tokens[6], tokens[7]
                        );
                        parse_fen(&mut board, &fen);
                        moves_idx = 8;
                    }

                    // Apply moves
                    if tokens.len() > moves_idx && tokens[moves_idx] == "moves" {
                        for move_str in &tokens[moves_idx + 1..] {
                            if let Some(m) = parse_ucci_move(&board, move_str) {
                                board.make_move(m);
                            }
                        }
                    }
                }
            }
            "go" => {
                let mut simulations = 800; // default MCTS simulations
                if tokens.len() > 2 {
                    if tokens[1] == "simulations" {
                        simulations = tokens[2].parse().unwrap_or(800);
                    } else if tokens[1] == "depth" {
                        // For backwards compatibility with standard UCCI GUIs
                        // Map depth ~4 to 800 roughly
                        let d: usize = tokens[2].parse().unwrap_or(4);
                        simulations = d * 200;
                    }
                }

                // Call MCTS Search instead of Alpha-Beta
                let (best_move, metrics) =
                    mcts.search_best_move(&board, &[], simulations, &eval_tx, &tt, 4, false);

                // UCCI Info format: info depth 0 nodes {root_visits} score winpct {win_pct} pv {pv}
                let mut pv_str = String::new();
                for (m, _nv, _pct) in &metrics.top_moves {
                    pv_str.push_str(&move_to_ucci_string(*m));
                    pv_str.push(' ');
                }

                println!(
                    "info depth 0 nodes {} score winpct {:.1} pv {}",
                    metrics.root_visits,
                    metrics.win_pct,
                    pv_str.trim()
                );

                let move_str = move_to_ucci_string(best_move);
                println!("bestmove {}", move_str);
            }
            "eval" => {
                println!("info eval {}", board.evaluate());
            }
            "d" => {
                println!("Side to move: {:?}", board.side_to_move);
                // Simple display
                for r in 0..10 {
                    for c in 0..9 {
                        let sq = Board::coord_to_square(r, c);
                        if let Some(p) = board.piece_at(sq) {
                            let mut ch = match p.piece_type {
                                PieceType::King => 'k',
                                PieceType::Advisor => 'a',
                                PieceType::Elephant => 'b',
                                PieceType::Horse => 'n',
                                PieceType::Rook => 'r',
                                PieceType::Cannon => 'c',
                                PieceType::Pawn => 'p',
                            };
                            if p.color == Color::Red {
                                ch = ch.to_ascii_uppercase();
                            }
                            print!("{} ", ch);
                        } else {
                            print!(". ");
                        }
                    }
                    println!();
                }
            }
            "quit" => {
                break;
            }
            _ => {}
        }
    }
}
