use crate::board::{Board, Color, PieceType};
use crate::r#move::Move;

/// Offsets for piece movement in the 16-width array
/// 16 = up (+16), -16 = down, 1 = right, -1 = left

const KING_ADVISOR_OFFSETS: [isize; 4] = [16, -16, 1, -1];
const ADVISOR_DIAG_OFFSETS: [isize; 4] = [15, 17, -15, -17]; // diagonals

const ELEPHANT_OFFSETS: [isize; 4] = [-34, -30, 34, 30]; // 15*2, 17*2, etc. (Wait, 15 is up-left. Up-left twice is 30. Up-right twice is 34)
const ELEPHANT_EYES: [isize; 4] = [-17, -15, 17, 15]; // The blocking points

const HORSE_JUMPS: [isize; 8] = [-33, -31, -14, 18, 33, 31, 14, -18]; // 16*2-1, 16*2+1, 16-2, 16+2 etc.
const HORSE_LEGS: [isize; 8] = [-16, -16, 1, 1, 16, 16, -1, -1]; // The blocking points (orthogonal steps before the jump)
// Example for HORSE_JUMPS[0] = 31 (Up 2, Left 1). The leg is Up 1 (16).
// This mapping matches exactly.

const ROOK_CANNON_DIRS: [isize; 4] = [16, -16, 1, -1];

impl Board {
    pub fn generate_captures(&self) -> Vec<Move> {
        let mut moves = Vec::with_capacity(32);
        self.generate_moves(true, &mut moves);
        moves
    }

    pub fn generate_quiets(&self) -> Vec<Move> {
        let mut moves = Vec::with_capacity(64);
        self.generate_moves(false, &mut moves);
        moves
    }

    fn generate_moves(&self, only_captures: bool, moves: &mut Vec<Move>) {
        for sq in 0..256 {
            if let Some(piece) = self.piece_at(sq) {
                if piece.color == self.side_to_move {
                    match piece.piece_type {
                        PieceType::King => self.gen_king_moves(sq, only_captures, moves),
                        PieceType::Advisor => self.gen_advisor_moves(sq, only_captures, moves),
                        PieceType::Elephant => self.gen_elephant_moves(sq, only_captures, moves),
                        PieceType::Horse => self.gen_horse_moves(sq, only_captures, moves),
                        PieceType::Rook => self.gen_rook_moves(sq, only_captures, moves),
                        PieceType::Cannon => self.gen_cannon_moves(sq, only_captures, moves),
                        PieceType::Pawn => {
                            self.gen_pawn_moves(sq, piece.color, only_captures, moves)
                        }
                    }
                }
            }
        }
    }

    // Helper to evaluate if a target square is a valid capture or quiet move based on the flag
    fn add_move_if_valid(
        &self,
        from: usize,
        to: usize,
        only_captures: bool,
        moves: &mut Vec<Move>,
    ) {
        if !Self::is_valid_square(to) {
            return;
        }

        match self.piece_at(to) {
            Some(target_piece) => {
                if target_piece.color != self.side_to_move {
                    // Valid capture
                    if only_captures {
                        moves.push(Move::new(from, to, true));
                    }
                }
            }
            None => {
                // Valid quiet move
                if !only_captures {
                    moves.push(Move::new(from, to, false));
                }
            }
        }
    }

    pub fn is_in_palace(sq: usize, color: Color) -> bool {
        let (row, col) = Self::square_to_coord(sq);
        if col < 3 || col > 5 {
            return false;
        }
        match color {
            Color::Black => row <= 2,
            Color::Red => row >= 7,
        }
    }

    pub fn is_on_own_side(sq: usize, color: Color) -> bool {
        let (row, _col) = Self::square_to_coord(sq);
        match color {
            Color::Black => row <= 4,
            Color::Red => row >= 5,
        }
    }

    fn gen_king_moves(&self, sq: usize, only_captures: bool, moves: &mut Vec<Move>) {
        for &offset in KING_ADVISOR_OFFSETS.iter() {
            if let Some(to) = sq.checked_add_signed(offset) {
                if Self::is_valid_square(to) && Self::is_in_palace(to, self.side_to_move) {
                    self.add_move_if_valid(sq, to, only_captures, moves);
                }
            }
        }
        // "Flying King" logic is handled separately during make_move or facing-check
    }

    fn gen_advisor_moves(&self, sq: usize, only_captures: bool, moves: &mut Vec<Move>) {
        for &offset in ADVISOR_DIAG_OFFSETS.iter() {
            if let Some(to) = sq.checked_add_signed(offset) {
                if Self::is_valid_square(to) && Self::is_in_palace(to, self.side_to_move) {
                    self.add_move_if_valid(sq, to, only_captures, moves);
                }
            }
        }
    }

    fn gen_elephant_moves(&self, sq: usize, only_captures: bool, moves: &mut Vec<Move>) {
        for i in 0..4 {
            if let Some(to) = sq.checked_add_signed(ELEPHANT_OFFSETS[i]) {
                if let Some(eye) = sq.checked_add_signed(ELEPHANT_EYES[i]) {
                    if Self::is_valid_square(to) && Self::is_on_own_side(to, self.side_to_move) {
                        // Check if eye is blocked
                        if self.is_empty(eye) {
                            self.add_move_if_valid(sq, to, only_captures, moves);
                        }
                    }
                }
            }
        }
    }

    fn gen_horse_moves(&self, sq: usize, only_captures: bool, moves: &mut Vec<Move>) {
        for i in 0..8 {
            if let Some(to) = sq.checked_add_signed(HORSE_JUMPS[i]) {
                if let Some(leg) = sq.checked_add_signed(HORSE_LEGS[i]) {
                    if Self::is_valid_square(to) {
                        // Check if the leg is blocked
                        if self.is_empty(leg) {
                            self.add_move_if_valid(sq, to, only_captures, moves);
                        }
                    }
                }
            }
        }
    }

    fn gen_rook_moves(&self, sq: usize, only_captures: bool, moves: &mut Vec<Move>) {
        for &dir in ROOK_CANNON_DIRS.iter() {
            let mut current_opt = sq.checked_add_signed(dir);
            while let Some(current) = current_opt {
                if !Self::is_valid_square(current) {
                    break;
                }
                match self.piece_at(current) {
                    None => {
                        if !only_captures {
                            moves.push(Move::new(sq, current, false));
                        }
                    }
                    Some(piece) => {
                        if piece.color != self.side_to_move && only_captures {
                            moves.push(Move::new(sq, current, true));
                        }
                        break; // Blocked by piece (own or enemy)
                    }
                }
                current_opt = current.checked_add_signed(dir);
            }
        }
    }

    fn gen_cannon_moves(&self, sq: usize, only_captures: bool, moves: &mut Vec<Move>) {
        for &dir in ROOK_CANNON_DIRS.iter() {
            let mut current_opt = sq.checked_add_signed(dir);
            let mut jumped = false;

            while let Some(current) = current_opt {
                if !Self::is_valid_square(current) {
                    break;
                }
                match self.piece_at(current) {
                    None => {
                        if !jumped {
                            if !only_captures {
                                moves.push(Move::new(sq, current, false));
                            }
                        }
                    }
                    Some(piece) => {
                        if !jumped {
                            jumped = true; // First piece encountered acts as the mount
                        } else {
                            // Second piece encountered
                            if piece.color != self.side_to_move && only_captures {
                                moves.push(Move::new(sq, current, true));
                            }
                            break; // Cannot jump over two pieces
                        }
                    }
                }
                current_opt = current.checked_add_signed(dir);
            }
        }
    }

    fn gen_pawn_moves(&self, sq: usize, color: Color, only_captures: bool, moves: &mut Vec<Move>) {
        let forward_dir = if color == Color::Red { -16 } else { 16 };

        // Forward move
        if let Some(forward_to) = sq.checked_add_signed(forward_dir) {
            if Self::is_valid_square(forward_to) {
                self.add_move_if_valid(sq, forward_to, only_captures, moves);
            }
        }

        // Sideways moves (only if crossed river)
        if !Self::is_on_own_side(sq, color) {
            for &side_dir in &[1isize, -1] {
                if let Some(side_to) = sq.checked_add_signed(side_dir) {
                    if Self::is_valid_square(side_to) {
                        self.add_move_if_valid(sq, side_to, only_captures, moves);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::board::Piece;

    use super::*;

    #[test]
    fn test_horse_blocking() {
        let mut board = Board::new();
        // Place Red Horse at (5, 4)
        let horse_sq = Board::coord_to_square(5, 4);
        board.set_piece(horse_sq, Some(Piece::new(PieceType::Horse, Color::Red)));

        // Block one leg going UP (4, 4)
        // Red Pawn at (4, 4) has crossed the river, so it generates 3 moves (1 forward, 2 side).
        board.set_piece(
            Board::coord_to_square(4, 4),
            Some(Piece::new(PieceType::Pawn, Color::Red)),
        );

        board.side_to_move = Color::Red;
        let quiets = board.generate_quiets();

        // Unblocked horse in center has 8 moves. With 1 leg blocked (Up), it loses 2 moves = 6 moves.
        // The Red Pawn at (4,4) generates 3 moves.
        // Total expected moves = 9.
        assert_eq!(quiets.len(), 9);
    }

    #[test]
    fn test_cannon_jumping() {
        let mut board = Board::new();
        let cannon_sq = Board::coord_to_square(5, 4);
        board.set_piece(cannon_sq, Some(Piece::new(PieceType::Cannon, Color::Red)));

        // Mount at (3, 4)
        board.set_piece(
            Board::coord_to_square(3, 4),
            Some(Piece::new(PieceType::Pawn, Color::Red)),
        );

        // Target at (1, 4)
        let target_sq = Board::coord_to_square(1, 4);
        board.set_piece(target_sq, Some(Piece::new(PieceType::Horse, Color::Black)));

        board.side_to_move = Color::Red;
        let captures = board.generate_captures();

        // One valid capture
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0].to_sq(), target_sq);
    }
}
