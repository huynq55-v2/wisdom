use crate::board::Board;
use crate::r#move::Move;

const INFINITY: i32 = 50000;
const MATE_VALUE: i32 = 20000;

use std::time::{Instant, Duration};

pub fn search_best_move(board: &mut Board, depth: u8, tt: &mut crate::tt::TranspositionTable) -> Move {
    // 1. Run Check-only Mate Search First with Timeout
    let start_time = Instant::now();
    let mate_depth = 11; // Configurable deep check depth
    if let Some(mate_move) = find_mate(board, mate_depth, tt, start_time, Duration::from_millis(50)) {
        return mate_move; // Found a forced mate
    }

    // 2. Fallback to normal search
    let mut best_move = None;
    let mut alpha = -INFINITY;
    let beta = INFINITY;

    let mut tt_move = None;
    if let Some((_score, best_tt)) = tt.probe(board.zobrist_key, depth, alpha, beta) {
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

        let score = -negamax(board, depth - 1, -beta, -alpha, tt);
        board.unmake_move(*m, undo);

        if score > alpha {
            alpha = score;
            best_move = Some(*m);
        }
    }

    if let Some(bm) = best_move {
        tt.record(board.zobrist_key, depth, alpha, crate::tt::FLAG_EXACT, Some(bm));
    }

    best_move.unwrap_or(moves[0]) // Fallback
}

fn negamax(board: &mut Board, depth: u8, mut alpha: i32, beta: i32, tt: &mut crate::tt::TranspositionTable) -> i32 {
    let orig_alpha = alpha;

    if depth == 0 {
        return quiescence(board, alpha, beta);
    }

    let mut tt_move = None;
    if let Some((score, best_tt)) = tt.probe(board.zobrist_key, depth, alpha, beta) {
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
        let score = -negamax(board, depth - 1, -beta, -alpha, tt);
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
        return -MATE_VALUE + (100 - depth as i32);
    }

    let flag = if best_score <= orig_alpha {
        crate::tt::FLAG_ALPHA
    } else if best_score >= beta {
        crate::tt::FLAG_BETA
    } else {
        crate::tt::FLAG_EXACT
    };

    tt.record(board.zobrist_key, depth, best_score, flag, best_move);

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

pub fn find_mate(
    board: &mut Board, 
    depth: u8, 
    tt: &mut crate::tt::TranspositionTable, 
    start_time: Instant, 
    timeout: Duration
) -> Option<Move> {
    
    let mut best_move = None;
    let mut alpha = -INFINITY;
    let beta = INFINITY;

    let mut checks = Vec::new();

    let mut caps = board.generate_captures();
    let mut quiets = board.generate_quiets();
    caps.append(&mut quiets);

    // Only collect checking moves
    for m in caps {
        if board.is_checking_move(m) {
            checks.push(m);
        }
    }

    let moving_side = board.side_to_move;

    for m in &checks {
        let undo = board.make_move(*m);

        if board.kings_facing() || board.is_in_check(moving_side) {
            board.unmake_move(*m, undo);
            continue;
        }

        let score = -mate_search(board, depth - 1, -beta, -alpha, tt, start_time, timeout);
        board.unmake_move(*m, undo);

        // If timed out, return None
        if start_time.elapsed() > timeout {
            return None;
        }

        if score > alpha {
            alpha = score;
            best_move = Some(*m);
        }
        
        // If we found a forced mate, return immediately
        if alpha >= MATE_VALUE - 100 {
            return best_move;
        }
    }

    if alpha >= MATE_VALUE - 100 {
        return best_move;
    }
    
    None
}

fn mate_search(
    board: &mut Board, 
    depth: u8, 
    mut alpha: i32, 
    beta: i32, 
    tt: &mut crate::tt::TranspositionTable, 
    start_time: Instant, 
    timeout: Duration
) -> i32 {
    if start_time.elapsed() > timeout {
        return 0; // abort signal
    }

    let orig_alpha = alpha;

    if depth == 0 {
        return board.evaluate(); // Only eval when depths runs out without mate
    }

    if let Some((score, _best_tt)) = tt.probe(board.zobrist_key, depth, alpha, beta) {
        if score != i32::MIN {
            return score;
        }
    }

    let mut caps = board.generate_captures();
    let mut quiets = board.generate_quiets();
    caps.append(&mut quiets); // all moves

    let moving_side = board.side_to_move;
    let in_check = board.is_in_check(moving_side);

    let mut moves_to_search = Vec::new();

    if in_check {
        // Defender must consider ALL legal evasions
        moves_to_search = caps;
    } else {
        // Attacker only considers Checking moves
        for m in caps {
            if board.is_checking_move(m) {
                moves_to_search.push(m);
            }
        }
        
        // If attacker has no checking moves left, the mate attack failed
        if moves_to_search.is_empty() {
            return board.evaluate();
        }
    }

    let mut best_score = -INFINITY;
    let mut has_legal_moves = false;
    let mut best_move = None;

    for m in &moves_to_search {
        let undo = board.make_move(*m);

        if board.kings_facing() || board.is_in_check(moving_side) {
            board.unmake_move(*m, undo);
            continue;
        }

        has_legal_moves = true;
        let score = -mate_search(board, depth - 1, -beta, -alpha, tt, start_time, timeout);
        board.unmake_move(*m, undo);

        if score > best_score {
            best_score = score;
            best_move = Some(*m);
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            break;
        }
    }

    if !has_legal_moves {
        if in_check {
            return -MATE_VALUE + (100 - depth as i32);
        } else {
            // Stalemate
            return -MATE_VALUE + (100 - depth as i32);
        }
    }

    let flag = if best_score <= orig_alpha {
        crate::tt::FLAG_ALPHA
    } else if best_score >= beta {
        crate::tt::FLAG_BETA
    } else {
        crate::tt::FLAG_EXACT
    };

    tt.record(board.zobrist_key, depth, best_score, flag, best_move);

    best_score
}
