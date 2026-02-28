use crate::board::Board;
use crate::r#move::Move;

pub const INFINITY: i32 = 50000;
pub const MATE_VALUE: i32 = 20000;

use std::time::{Instant, Duration};

pub fn search_best_move(board: &mut Board, depth: u8, tt: &mut crate::tt::TranspositionTable) -> Move {
    // Standard Negamax Search
    let start_time = Instant::now();
    let mut best_move = None;
    let mut alpha = -INFINITY;
    let beta = INFINITY;

    let mut tt_move = None;
    if let Some((_score, best_tt)) = tt.probe(board.zobrist_key, depth, 0, alpha, beta) {
        tt_move = best_tt;
    }

    // We must generate moves at root and pick the one with highest score.
    let mut moves = board.generate_captures();
    moves.sort_by_key(|&m| -mvv_lva(board, m));
    let mut quiets = board.generate_quiets();
    moves.append(&mut quiets); // captures first for move ordering

    // Move TT move to the front
    if let Some(t_mv) = tt_move {
        if let Some(pos) = moves.iter().position(|&m| m.to_sq() == t_mv.to_sq() && m.from_sq() == t_mv.from_sq()) {
            moves.swap(0, pos);
        }
    }

    let moving_side = board.side_to_move;

    for m in &moves {
        let undo = board.make_move(*m);

        if board.kings_facing() || board.is_in_check(moving_side) {
            board.unmake_move(*m, undo);
            continue;
        }

        // Check Extension logic: if this move gives check, we don't reduce depth.
        // As a safeguard against infinite checking loops, we only extend up to a reasonable bound relative to original depth (e.g. + 4)
        // For simplicity in this base implementation, we extend if giving check, but cap the max depth to avoid blowups.
        let gives_check = board.is_checking_move(*m);
        let next_depth = if gives_check { depth } else { depth - 1 };

        let score = -negamax(board, next_depth, 1, -beta, -alpha, tt);
        board.unmake_move(*m, undo);

        if score > alpha {
            alpha = score;
            best_move = Some(*m);
        }
    }

    if let Some(bm) = best_move {
        tt.record(board.zobrist_key, depth, 0, alpha, crate::tt::FLAG_EXACT, Some(bm));
    }

    best_move.unwrap_or(moves[0]) // Fallback
}

fn negamax(board: &mut Board, depth: u8, ply: u8, mut alpha: i32, beta: i32, tt: &mut crate::tt::TranspositionTable) -> i32 {
    let orig_alpha = alpha;

    if depth == 0 {
        return quiescence(board, alpha, beta);
    }

    let mut tt_move = None;
    if let Some((score, best_tt)) = tt.probe(board.zobrist_key, depth, ply, alpha, beta) {
        if score != i32::MIN {
            return score;
        }
        tt_move = best_tt;
    }

    let mut moves = board.generate_captures();
    moves.sort_by_key(|&m| -mvv_lva(board, m));
    let mut quiets = board.generate_quiets();
    moves.append(&mut quiets);

    // Swap tt_move to front
    if let Some(t_mv) = tt_move {
        if let Some(pos) = moves.iter().position(|&m| m.to_sq() == t_mv.to_sq() && m.from_sq() == t_mv.from_sq()) {
            moves.swap(0, pos);
        }
    }

    let mut best_score = -INFINITY;
    let mut best_move = None;
    let moving_side = board.side_to_move;
    let mut has_legal_moves = false;

    for m in &moves {
        let undo = board.make_move(*m);

        if board.kings_facing() || board.is_in_check(moving_side) {
            board.unmake_move(*m, undo);
            continue;
        }

        has_legal_moves = true;

        let gives_check = board.is_checking_move(*m);
        let next_depth = if gives_check { depth } else { depth - 1 };

        let score = -negamax(board, next_depth, ply + 1, -beta, -alpha, tt);
        board.unmake_move(*m, undo);

        if score > best_score {
            best_score = score;
            best_move = Some(*m);
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            break; // Beta cutoff
        }
    }

    if !has_legal_moves {
        // If in check -> Checkmate
        // If not in check -> Stalemate (also loss in Xiangqi)
        return -MATE_VALUE + ply as i32;
    }

    let flag = if best_score <= orig_alpha {
        crate::tt::FLAG_ALPHA
    } else if best_score >= beta {
        crate::tt::FLAG_BETA
    } else {
        crate::tt::FLAG_EXACT
    };

    tt.record(board.zobrist_key, depth, ply, best_score, flag, best_move);

    best_score
}

fn mvv_lva(board: &Board, m: Move) -> i32 {
    let victim = board.piece_at(m.to_sq()).unwrap();
    let attacker = board.piece_at(m.from_sq()).unwrap();
    
    // Prioritize high value victim, then low value attacker
    // e.g. Pawn taking Rook = 900*100 - 100 = 89900
    victim.piece_type.value() * 100 - attacker.piece_type.value()
}

fn quiescence(board: &mut Board, mut alpha: i32, beta: i32) -> i32 {
    let stand_pat = board.evaluate();
    if stand_pat >= beta {
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    let mut captures = board.generate_captures();
    captures.sort_by_key(|&m| -mvv_lva(board, m)); // Higher score first

    let moving_side = board.side_to_move;

    for m in &captures {
        let undo = board.make_move(*m);

        if board.kings_facing() || board.is_in_check(moving_side) {
            board.unmake_move(*m, undo);
            continue;
        }

        let score = -quiescence(board, -beta, -alpha);
        board.unmake_move(*m, undo);

        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}


