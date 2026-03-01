use crate::board::{Board, HistoryEntry, RepetitionResult};
use crate::eval_queue::EvalRequest;
use crate::r#move::Move;
use crossbeam_channel::Sender;

pub const INFINITY: i32 = 50000;
pub const MATE_VALUE: i32 = 20000;

pub fn evaluate_node(board: &Board, eval_tx: Option<&Sender<EvalRequest>>) -> i32 {
    let (val, _) = evaluate_node_with_policy(board, eval_tx, false);
    val
}

pub fn evaluate_node_with_policy(
    board: &Board,
    eval_tx: Option<&Sender<EvalRequest>>,
    need_policy: bool,
) -> (i32, Option<Vec<f32>>) {
    if let Some(tx_queue) = eval_tx {
        let tensor = crate::nn::board_to_tensor(board);
        let (tx, rx) = crossbeam_channel::bounded(1);
        tx_queue
            .send(crate::eval_queue::EvalRequest {
                tensor_data: tensor,
                response_tx: tx,
                need_policy,
            })
            .unwrap();

        let (nn_value, policy) = rx.recv().unwrap();
        ((nn_value * 10000.0) as i32, policy)
    } else {
        (board.evaluate(), None)
    }
}

pub fn search_best_move(
    board: &mut Board,
    depth: u8,
    tt: &crate::tt::TranspositionTable,
    game_history: &[HistoryEntry],
    eval_tx: Option<&Sender<EvalRequest>>,
) -> (Move, i32) {
    let mut history = game_history.to_vec();
    let mut best_move = None;
    let mut alpha = -INFINITY;
    let beta = INFINITY;

    let mut tt_move = None;
    if let Some((_score, best_tt)) = tt.probe(board.zobrist_key, depth, 0, alpha, beta) {
        tt_move = best_tt;
    }

    let moving_side = board.side_to_move;

    // Evaluate root policy for move ordering if NN is available
    let mut root_policy: Option<Vec<f32>> = None;
    if depth > 0 {
        let (_, p) = evaluate_node_with_policy(board, eval_tx, true);
        root_policy = p;
    }

    // === Algorithm 9: Staged Evaluation (Merged for Policy-Driven Search) ===
    let mut all_moves = board.generate_captures();
    all_moves.append(&mut board.generate_quiets());

    // Sort all moves by policy logits to prioritize promising branches
    if let Some(ref policy) = root_policy {
        all_moves.sort_by(|a, b| {
            let idx_a = crate::nn::move_to_index(*a);
            let idx_b = crate::nn::move_to_index(*b);
            let logit_a = policy[idx_a];
            let logit_b = policy[idx_b];
            logit_b
                .partial_cmp(&logit_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        // Fallback if no policy: Captures first (sorted by MVV-LVA)
        all_moves.sort_unstable_by_key(|&m| {
            if board.piece_at(m.to_sq()).is_some() {
                -mvv_lva(board, m) - 1000000
            } else {
                0
            }
        });
    }

    // TT move to front
    if let Some(t_mv) = tt_move {
        if let Some(pos) = all_moves
            .iter()
            .position(|&m| m.to_sq() == t_mv.to_sq() && m.from_sq() == t_mv.from_sq())
        {
            all_moves.swap(0, pos);
        }
    }

    for m in &all_moves {
        let piece = board.piece_at(m.from_sq()).unwrap();
        let is_reversible = piece.piece_type != crate::board::PieceType::Pawn || {
            let (from_row, _) = Board::square_to_coord(m.from_sq());
            let (to_row, _) = Board::square_to_coord(m.to_sq());
            from_row == to_row
        };
        let is_capture = board.piece_at(m.to_sq()).is_some();

        let pre_threats = if is_reversible && !is_capture {
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
        let next_depth = depth - 1;

        let chased_set = if is_reversible && !gives_check && !is_capture {
            let post_threats = board.get_unprotected_threats(moving_side);
            post_threats & !pre_threats
        } else {
            0
        };

        history.push(HistoryEntry {
            hash: board.zobrist_key,
            is_check: gives_check,
            chased_set,
            is_reversible: is_reversible && !is_capture,
        });

        let score = -negamax(
            board,
            next_depth,
            1,
            -beta,
            -alpha,
            tt,
            &mut history,
            eval_tx,
        );

        history.pop();
        board.unmake_move(*m, undo);

        if score > alpha {
            alpha = score;
            best_move = Some(*m);
        }
    }

    if let Some(bm) = best_move {
        tt.record(
            board.zobrist_key,
            depth,
            0,
            alpha,
            crate::tt::FLAG_EXACT,
            Some(bm),
        );
    }

    // Fallback: if no best_move found, pick first legal move
    if best_move.is_none() {
        for m in &all_moves {
            let undo = board.make_move(*m);
            let legal = !board.kings_facing() && !board.is_in_check(moving_side);
            board.unmake_move(*m, undo);
            if legal {
                return (*m, alpha);
            }
        }
    }

    (best_move.unwrap(), alpha)
}

fn negamax(
    board: &mut Board,
    depth: u8,
    ply: u8,
    mut alpha: i32,
    beta: i32,
    tt: &crate::tt::TranspositionTable,
    history: &mut Vec<HistoryEntry>,
    eval_tx: Option<&Sender<EvalRequest>>,
) -> i32 {
    // Algorithm 10: Quick prune if draw beats beta and we are idle
    if board.judge_prune(history, history.len(), beta) {
        return 0;
    }

    match board.judge_repetition(history, history.len(), 1) {
        RepetitionResult::Win => return MATE_VALUE - ply as i32,
        RepetitionResult::Loss => return -MATE_VALUE + ply as i32,
        RepetitionResult::Draw => return 0,
        RepetitionResult::Undecided => {}
    }

    let orig_alpha = alpha;

    if depth == 0 {
        if eval_tx.is_some() {
            return evaluate_node(board, eval_tx);
        }
        return quiescence(board, alpha, beta, eval_tx);
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

    // =========================================================
    // PHASE 1: Captures only (sorted by MVV-LVA, early cutoff)
    // =========================================================
    let mut captures = board.generate_captures();
    captures.sort_unstable_by_key(|&m| -mvv_lva(board, m));

    // TT move to front of captures
    if let Some(t_mv) = tt_move {
        if let Some(pos) = captures
            .iter()
            .position(|&m| m.to_sq() == t_mv.to_sq() && m.from_sq() == t_mv.from_sq())
        {
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

        history.push(HistoryEntry {
            hash: board.zobrist_key,
            is_check: gives_check,
            chased_set: 0,
            is_reversible: false,
        });

        let score = -negamax(
            board,
            depth - 1,
            ply + 1,
            -beta,
            -alpha,
            tt,
            history,
            eval_tx,
        );

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
            tt.record(
                board.zobrist_key,
                depth,
                ply,
                best_score,
                crate::tt::FLAG_BETA,
                best_move,
            );
            return best_score;
        }
    }

    // =========================================================
    // PHASE 2: Quiets (only if no beta cutoff during captures)
    // =========================================================
    let quiets = board.generate_quiets();

    // TT move to front of quiets
    let mut quiets = quiets;
    if let Some(t_mv) = tt_move {
        if let Some(pos) = quiets
            .iter()
            .position(|&m| m.to_sq() == t_mv.to_sq() && m.from_sq() == t_mv.from_sq())
        {
            quiets.swap(0, pos);
        }
    }

    for m in &quiets {
        let piece = board.piece_at(m.from_sq()).unwrap();
        let is_reversible = piece.piece_type != crate::board::PieceType::Pawn || {
            let (from_row, _) = Board::square_to_coord(m.from_sq());
            let (to_row, _) = Board::square_to_coord(m.to_sq());
            from_row == to_row
        };

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

        let score = -negamax(
            board,
            depth - 1,
            ply + 1,
            -beta,
            -alpha,
            tt,
            history,
            eval_tx,
        );

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

fn quiescence(
    board: &mut Board,
    mut alpha: i32,
    beta: i32,
    eval_tx: Option<&Sender<EvalRequest>>,
) -> i32 {
    let stand_pat = evaluate_node(board, eval_tx);
    if stand_pat >= beta {
        return beta;
    }
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    let mut captures = board.generate_captures();
    captures.sort_unstable_by_key(|&m| -mvv_lva(board, m)); // Higher score first

    let moving_side = board.side_to_move;

    for m in &captures {
        let undo = board.make_move(*m);

        if board.kings_facing() || board.is_in_check(moving_side) {
            board.unmake_move(*m, undo);
            continue;
        }

        let score = -quiescence(board, -beta, -alpha, eval_tx);
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

pub fn search_best_move_parallel(
    board: &Board,
    depth: u8,
    tt: &crate::tt::TranspositionTable,
    game_history: &[HistoryEntry],
    eval_tx: &crossbeam_channel::Sender<EvalRequest>,
    num_threads: usize,
) -> (Move, i32) {
    let best_move_global = std::sync::Mutex::new(None);
    let best_score_global = std::sync::Mutex::new(-INFINITY);

    std::thread::scope(|s| {
        for thread_id in 0..num_threads {
            // Lazy SMP: thread 0 runs the target depth, helper threads run deeper/shallower depths
            // to probe and fill the TT from different paths.
            let mut local_board = board.clone();
            let local_history = game_history.to_vec();

            let bg = &best_move_global;
            let bs = &best_score_global;

            s.spawn(move || {
                // BUG FIX 3: Lazy SMP - helper threads must search at EQUAL or DEEPER depth
                // to stay alive and populate TT. Shallower threads die instantly and waste cores.
                // Thread 0: depth, Thread 1: depth+1, Thread 2: depth, Thread 3: depth+1, ...
                let thread_depth = depth + (thread_id as u8 % 2);

                let (m, score) = search_best_move(
                    &mut local_board,
                    thread_depth,
                    tt,
                    &local_history,
                    Some(eval_tx),
                );

                if thread_id == 0 {
                    let mut bs_guard = bs.lock().unwrap();
                    if score > *bs_guard {
                        let mut bg_guard = bg.lock().unwrap();
                        *bs_guard = score;
                        *bg_guard = Some(m);
                    }
                }
            });
        }
    });

    (
        best_move_global.into_inner().unwrap().unwrap(),
        best_score_global.into_inner().unwrap(),
    )
}
