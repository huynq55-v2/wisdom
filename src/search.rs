use crate::board::{Board, HistoryEntry, RepetitionResult};
use crate::r#move::Move;

pub const INFINITY: i32 = 50000;
pub const MATE_VALUE: i32 = 20000;

pub fn search_best_move(board: &mut Board, depth: u8, tt: &mut crate::tt::TranspositionTable, game_history: &[HistoryEntry]) -> Move {
    let mut history = game_history.to_vec();
    let mut best_move = None;
    let mut alpha = -INFINITY;
    let beta = INFINITY;

    let mut tt_move = None;
    if let Some((_score, best_tt)) = tt.probe(board.zobrist_key, depth, 0, alpha, beta) {
        tt_move = best_tt;
    }

    let moving_side = board.side_to_move;

    // === Algorithm 9: Staged Evaluation ===
    // Phase 1: Process captures first (no chase computation needed)
    let mut captures = board.generate_captures();
    captures.sort_by_key(|&m| -mvv_lva(board, m));

    // TT move to front of captures
    if let Some(t_mv) = tt_move {
        if let Some(pos) = captures.iter().position(|&m| m.to_sq() == t_mv.to_sq() && m.from_sq() == t_mv.from_sq()) {
            captures.swap(0, pos);
        }
    }

    for m in &captures {
        let undo = board.make_move(*m);

        if board.kings_facing() || board.is_in_check(moving_side) {
            board.unmake_move(*m, undo);
            continue;
        }

        let gives_check = board.is_in_check(moving_side.opposite());
        let next_depth = if gives_check { depth } else { depth - 1 };

        history.push(HistoryEntry {
            hash: board.zobrist_key,
            is_check: gives_check,
            chased_set: 0, // Captures are never reversible
            is_reversible: false,
        });

        let score = -negamax(board, next_depth, 1, -beta, -alpha, tt, &mut history);

        history.pop();
        board.unmake_move(*m, undo);

        if score > alpha {
            alpha = score;
            best_move = Some(*m);
        }
    }

    // Phase 2: Process quiet moves (only if no beta cutoff during captures)
    if alpha < beta {
        let mut quiets = board.generate_quiets();

        // TT move to front of quiets
        if let Some(t_mv) = tt_move {
            if let Some(pos) = quiets.iter().position(|&m| m.to_sq() == t_mv.to_sq() && m.from_sq() == t_mv.from_sq()) {
                quiets.swap(0, pos);
            }
        }

        for m in &quiets {
            let piece = board.piece_at(m.from_sq()).unwrap();
            let is_reversible = piece.piece_type != crate::board::PieceType::Pawn;

            let pre_threats = if is_reversible {
                board.get_unprotected_threats(moving_side)
            } else {
                0
            };

            let undo = board.make_move(*m);

            if board.kings_facing() || board.is_in_check(moving_side) {
                board.unmake_move(*m, undo);
                continue;
            }

            let gives_check = board.is_in_check(moving_side.opposite());
            let next_depth = if gives_check { depth } else { depth - 1 };

            let chased_set = if is_reversible && !gives_check {
                let post_threats = board.get_unprotected_threats(moving_side);
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

            let score = -negamax(board, next_depth, 1, -beta, -alpha, tt, &mut history);

            history.pop();
            board.unmake_move(*m, undo);

            if score > alpha {
                alpha = score;
                best_move = Some(*m);
            }
        }
    }

    if let Some(bm) = best_move {
        tt.record(board.zobrist_key, depth, 0, alpha, crate::tt::FLAG_EXACT, Some(bm));
    }

    // Fallback: if no best_move found, pick first legal move
    if best_move.is_none() {
        let mut all_moves = board.generate_captures();
        all_moves.append(&mut board.generate_quiets());
        for m in &all_moves {
            let undo = board.make_move(*m);
            let legal = !board.kings_facing() && !board.is_in_check(moving_side);
            board.unmake_move(*m, undo);
            if legal { return *m; }
        }
    }

    best_move.unwrap()
}

fn negamax(board: &mut Board, depth: u8, ply: u8, mut alpha: i32, beta: i32, tt: &mut crate::tt::TranspositionTable, history: &mut Vec<HistoryEntry>) -> i32 {
    
    // Algorithm 10: Quick prune if draw beats beta and we are idle
    if board.judge_prune(history, history.len(), beta) {
        return 0;
    }

    match board.judge_repetition(history, history.len()) {
        RepetitionResult::Win => return MATE_VALUE - ply as i32,
        RepetitionResult::Loss => return -MATE_VALUE + ply as i32,
        RepetitionResult::Draw => return 0,
        RepetitionResult::Undecided => {}
    }

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

    let mut best_score = -INFINITY;
    let mut best_move = None;
    let moving_side = board.side_to_move;
    let mut has_legal_moves = false;

    // === Algorithm 9: Staged Evaluation ===
    // Phase 1: Process captures (no chase computation, likely to cause beta cutoff)
    let mut captures = board.generate_captures();
    captures.sort_by_key(|&m| -mvv_lva(board, m));

    // TT move to front of captures
    if let Some(t_mv) = tt_move {
        if let Some(pos) = captures.iter().position(|&m| m.to_sq() == t_mv.to_sq() && m.from_sq() == t_mv.from_sq()) {
            captures.swap(0, pos);
        }
    }

    for m in &captures {
        let undo = board.make_move(*m);

        if board.kings_facing() || board.is_in_check(moving_side) {
            board.unmake_move(*m, undo);
            continue;
        }

        has_legal_moves = true;

        let gives_check = board.is_in_check(moving_side.opposite());
        let next_depth = if gives_check { depth } else { depth - 1 };

        history.push(HistoryEntry {
            hash: board.zobrist_key,
            is_check: gives_check,
            chased_set: 0,
            is_reversible: false,
        });

        let score = -negamax(board, next_depth, ply + 1, -beta, -alpha, tt, history);

        history.pop();
        board.unmake_move(*m, undo);

        if score > best_score {
            best_score = score;
            best_move = Some(*m);
        }
        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            // Beta cutoff during captures — skip quiet move generation entirely!
            let flag = crate::tt::FLAG_BETA;
            tt.record(board.zobrist_key, depth, ply, best_score, flag, best_move);
            return best_score;
        }
    }

    // Phase 2: Process quiet moves (expensive chase computation only here)
    let mut quiets = board.generate_quiets();

    // TT move to front of quiets
    if let Some(t_mv) = tt_move {
        if let Some(pos) = quiets.iter().position(|&m| m.to_sq() == t_mv.to_sq() && m.from_sq() == t_mv.from_sq()) {
            quiets.swap(0, pos);
        }
    }

    for m in &quiets {
        let piece = board.piece_at(m.from_sq()).unwrap();
        let is_reversible = piece.piece_type != crate::board::PieceType::Pawn;

        let pre_threats = if is_reversible {
            board.get_unprotected_threats(moving_side)
        } else {
            0
        };

        let undo = board.make_move(*m);

        if board.kings_facing() || board.is_in_check(moving_side) {
            board.unmake_move(*m, undo);
            continue;
        }

        has_legal_moves = true;

        let gives_check = board.is_in_check(moving_side.opposite());
        let next_depth = if gives_check { depth } else { depth - 1 };

        let chased_set = if is_reversible && !gives_check {
            let post_threats = board.get_unprotected_threats(moving_side);
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

        let score = -negamax(board, next_depth, ply + 1, -beta, -alpha, tt, history);

        history.pop();
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


