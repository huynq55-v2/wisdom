use crate::board::Board;

pub fn perft(board: &mut Board, depth: u8) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut nodes = 0;

    // Generate all pseudo-legal moves
    let mut moves = board.generate_captures();
    let mut quiets = board.generate_quiets();
    moves.append(&mut quiets);

    // The side that is moving is `board.side_to_move`
    let moving_side = board.side_to_move;

    for m in moves {
        let undo = board.make_move(m);

        // A move is legal if it doesn't leave the moving side in check
        // and doesn't leave the kings facing each other.
        if !board.kings_facing() && !board.is_in_check(moving_side) {
            nodes += perft(board, depth - 1);
        }

        board.unmake_move(m, undo);
    }

    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perft_depth_1() {
        let mut board = Board::new();
        board.set_initial_position();

        let nodes = perft(&mut board, 1);
        // From the standard starting position in Xiangqi:
        // 44 pseudo-legal moves. None of them face kings or leave in check.
        // So perft(1) should be exactly 44.
        assert_eq!(nodes, 44);
    }

    #[test]
    fn test_perft_depth_2() {
        let mut board = Board::new();
        board.set_initial_position();

        let nodes = perft(&mut board, 2);
        // Perft(2) from standard starting position is exactly 1920.
        assert_eq!(nodes, 1920);
    }

    #[test]
    fn test_perft_depth_3() {
        let mut board = Board::new();
        board.set_initial_position();

        let nodes = perft(&mut board, 3);
        // Perft(3) from standard starting position is exactly 79666.
        assert_eq!(nodes, 79666);
    }

    #[test]
    fn test_perft_depth_4() {
        let mut board = Board::new();
        board.set_initial_position();

        let nodes = perft(&mut board, 4);
        // Perft(4) from standard starting position is exactly 3290240.
        assert_eq!(nodes, 3290240);
    }
}
