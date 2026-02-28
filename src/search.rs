use crate::board::Board;
use crate::r#move::Move;

const INFINITY: i32 = 50000;
const MATE_VALUE: i32 = 20000;

pub fn search_best_move(board: &mut Board, depth: u8) -> Move {
    let mut best_move = None;
    let mut alpha = -INFINITY;
    let beta = INFINITY;

    // We must generate moves at root and pick the one with highest score.
    let mut moves = board.generate_captures();
    let mut quiets = board.generate_quiets();
    moves.append(&mut quiets); // captures first for move ordering

    let moving_side = board.side_to_move;

    for m in &moves {
        let undo = board.make_move(*m);

        if board.kings_facing() || board.is_in_check(moving_side) {
            board.unmake_move(*m, undo);
            continue;
        }

        let score = -negamax(board, depth - 1, -beta, -alpha);
        board.unmake_move(*m, undo);

        if score > alpha {
            alpha = score;
            best_move = Some(*m);
        }
    }

    best_move.unwrap_or(moves[0]) // Fallback
}

fn negamax(board: &mut Board, depth: u8, mut alpha: i32, beta: i32) -> i32 {
    if depth == 0 {
        return board.evaluate();
    }

    let mut moves = board.generate_captures();
    let mut quiets = board.generate_quiets();
    moves.append(&mut quiets);

    let mut best_score = -INFINITY;
    let moving_side = board.side_to_move;
    let mut has_legal_moves = false;

    for m in &moves {
        let undo = board.make_move(*m);

        if board.kings_facing() || board.is_in_check(moving_side) {
            board.unmake_move(*m, undo);
            continue;
        }

        has_legal_moves = true;
        let score = -negamax(board, depth - 1, -beta, -alpha);
        board.unmake_move(*m, undo);

        if score > best_score {
            best_score = score;
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

    best_score
}
