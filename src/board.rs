#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Color {
    Red,
    Black,
}

impl Color {
    pub fn opposite(&self) -> Self {
        match self {
            Color::Red => Color::Black,
            Color::Black => Color::Red,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PieceType {
    King,    // Tướng
    Advisor, // Sĩ
    Elephant, // Tượng
    Horse,   // Mã
    Rook,    // Xe
    Cannon,  // Pháo
    Pawn,    // Tốt
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    pub piece_type: PieceType,
    pub color: Color,
}

impl Piece {
    pub const fn new(piece_type: PieceType, color: Color) -> Self {
        Self { piece_type, color }
    }
}

// 0x88 method but adjusted for Xiangqi 10x9 board
// The actual playable area uses indices where the column is 0..=8 and row is 0..=9
// In a 16-width array, a square index `sq` is valid if `(sq & 0x8F) <= 8`.
// However, since we need 10 rows, a u8 is perfectly fine (max index 16*9+8 = 152).
// To keep things simple and power-of-2 aligned, we use an array of size 256.
pub struct Board {
    squares: [Option<Piece>; 256],
    pub side_to_move: Color,
}

impl Board {
    pub fn new() -> Self {
        Self {
            squares: [None; 256],
            side_to_move: Color::Red,
        }
    }

    /// Checks if a square index is strictly within the 10x9 board boundaries.
    pub fn is_valid_square(square: usize) -> bool {
        // We use 16 columns per row.
        // Valid column: square % 16 <= 8  (or square & 0x0F <= 8)
        // Valid row: square / 16 <= 9
        (square & 0x0F) <= 8 && square <= 0x98
    }

    /// Returns the coordinate representation (row, col) from 0..9 and 0..8
    pub fn square_to_coord(square: usize) -> (usize, usize) {
        (square / 16, square % 16)
    }

    /// Given a 0-based row (0-9) and col (0-8), returns the 0x88 square index
    pub fn coord_to_square(row: usize, col: usize) -> usize {
        row * 16 + col
    }

    pub fn piece_at(&self, square: usize) -> Option<Piece> {
        if Self::is_valid_square(square) {
            self.squares[square]
        } else {
            None
        }
    }

    pub fn set_piece(&mut self, square: usize, piece: Option<Piece>) {
        if Self::is_valid_square(square) {
            self.squares[square] = piece;
        }
    }

    pub fn is_empty(&self, square: usize) -> bool {
        self.piece_at(square).is_none()
    }

    /// Checks if the two kings are facing each other on the same file with no pieces in between
    pub fn kings_facing(&self) -> bool {
        let mut red_king_sq = None;
        let mut black_king_sq = None;

        // Black palace: coords (0,3) to (2,5)
        for row in 0..=2 {
            for col in 3..=5 {
                let sq = Self::coord_to_square(row, col);
                if let Some(piece) = self.piece_at(sq) {
                    if piece.piece_type == PieceType::King {
                        black_king_sq = Some(sq);
                        break;
                    }
                }
            }
            if black_king_sq.is_some() { break; }
        }

        // Red palace: coords (7,3) to (9,5)
        for row in 7..=9 {
            for col in 3..=5 {
                let sq = Self::coord_to_square(row, col);
                if let Some(piece) = self.piece_at(sq) {
                    if piece.piece_type == PieceType::King {
                        red_king_sq = Some(sq);
                        break;
                    }
                }
            }
            if red_king_sq.is_some() { break; }
        }

        if let (Some(r_sq), Some(b_sq)) = (red_king_sq, black_king_sq) {
            let (r_row, r_col) = Self::square_to_coord(r_sq);
            let (b_row, b_col) = Self::square_to_coord(b_sq);

            if r_col == b_col {
                let mut current = b_row + 1;
                while current < r_row {
                    let sq = Self::coord_to_square(current, r_col);
                    if !self.is_empty(sq) {
                        return false; // Blocked by a piece
                    }
                    current += 1;
                }
                // No pieces found between them
                return true;
            }
        }
        
        false
    }

    pub fn set_initial_position(&mut self) {
        self.squares = [None; 256];
        self.side_to_move = Color::Red;

        use PieceType::*;
        use Color::*;

        // Setup Black (Top, Rows 0-3)
        self.set_piece(Self::coord_to_square(0, 0), Some(Piece::new(Rook, Black)));
        self.set_piece(Self::coord_to_square(0, 1), Some(Piece::new(Horse, Black)));
        self.set_piece(Self::coord_to_square(0, 2), Some(Piece::new(Elephant, Black)));
        self.set_piece(Self::coord_to_square(0, 3), Some(Piece::new(Advisor, Black)));
        self.set_piece(Self::coord_to_square(0, 4), Some(Piece::new(King, Black)));
        self.set_piece(Self::coord_to_square(0, 5), Some(Piece::new(Advisor, Black)));
        self.set_piece(Self::coord_to_square(0, 6), Some(Piece::new(Elephant, Black)));
        self.set_piece(Self::coord_to_square(0, 7), Some(Piece::new(Horse, Black)));
        self.set_piece(Self::coord_to_square(0, 8), Some(Piece::new(Rook, Black)));

        self.set_piece(Self::coord_to_square(2, 1), Some(Piece::new(Cannon, Black)));
        self.set_piece(Self::coord_to_square(2, 7), Some(Piece::new(Cannon, Black)));

        self.set_piece(Self::coord_to_square(3, 0), Some(Piece::new(Pawn, Black)));
        self.set_piece(Self::coord_to_square(3, 2), Some(Piece::new(Pawn, Black)));
        self.set_piece(Self::coord_to_square(3, 4), Some(Piece::new(Pawn, Black)));
        self.set_piece(Self::coord_to_square(3, 6), Some(Piece::new(Pawn, Black)));
        self.set_piece(Self::coord_to_square(3, 8), Some(Piece::new(Pawn, Black)));

        // Setup Red (Bottom, Rows 6-9)
        self.set_piece(Self::coord_to_square(9, 0), Some(Piece::new(Rook, Red)));
        self.set_piece(Self::coord_to_square(9, 1), Some(Piece::new(Horse, Red)));
        self.set_piece(Self::coord_to_square(9, 2), Some(Piece::new(Elephant, Red)));
        self.set_piece(Self::coord_to_square(9, 3), Some(Piece::new(Advisor, Red)));
        self.set_piece(Self::coord_to_square(9, 4), Some(Piece::new(King, Red)));
        self.set_piece(Self::coord_to_square(9, 5), Some(Piece::new(Advisor, Red)));
        self.set_piece(Self::coord_to_square(9, 6), Some(Piece::new(Elephant, Red)));
        self.set_piece(Self::coord_to_square(9, 7), Some(Piece::new(Horse, Red)));
        self.set_piece(Self::coord_to_square(9, 8), Some(Piece::new(Rook, Red)));

        self.set_piece(Self::coord_to_square(7, 1), Some(Piece::new(Cannon, Red)));
        self.set_piece(Self::coord_to_square(7, 7), Some(Piece::new(Cannon, Red)));

        self.set_piece(Self::coord_to_square(6, 0), Some(Piece::new(Pawn, Red)));
        self.set_piece(Self::coord_to_square(6, 2), Some(Piece::new(Pawn, Red)));
        self.set_piece(Self::coord_to_square(6, 4), Some(Piece::new(Pawn, Red)));
        self.set_piece(Self::coord_to_square(6, 6), Some(Piece::new(Pawn, Red)));
        self.set_piece(Self::coord_to_square(6, 8), Some(Piece::new(Pawn, Red)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kings_facing() {
        let mut board = Board::new();
        
        let red_king_sq = Board::coord_to_square(9, 4);
        let black_king_sq = Board::coord_to_square(0, 4);
        
        board.set_piece(red_king_sq, Some(Piece::new(PieceType::King, Color::Red)));
        board.set_piece(black_king_sq, Some(Piece::new(PieceType::King, Color::Black)));
        
        // Kings on the same file with no pieces in between -> facing
        assert!(board.kings_facing());
        
        // Block the file with a piece
        board.set_piece(Board::coord_to_square(5, 4), Some(Piece::new(PieceType::Pawn, Color::Red)));
        assert!(!board.kings_facing());
        
        // Move one king to a different file
        board.set_piece(black_king_sq, None);
        board.set_piece(Board::coord_to_square(0, 3), Some(Piece::new(PieceType::King, Color::Black)));
        
        // Even without blockers, they are not on the same file -> not facing
        board.set_piece(Board::coord_to_square(5, 4), None);
        assert!(!board.kings_facing());
    }
}
