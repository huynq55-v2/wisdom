use crate::r#move::Move;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ViolationLevel {
    Undecided = -1,
    PerpetualIdle = 0,
    PerpetualChase = 1,
    PerpetualCheck = 2,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RepetitionResult {
    Win,
    Loss,
    Draw,
    Undecided,
}

#[derive(Copy, Clone)]
pub struct HistoryEntry {
    pub hash: u64,
    pub is_check: bool,
    pub chased_set: u128,
    pub is_reversible: bool,
}
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
    King,     // Tướng
    Advisor,  // Sĩ
    Elephant, // Tượng
    Horse,    // Mã
    Rook,     // Xe
    Cannon,   // Pháo
    Pawn,     // Tốt
}

impl PieceType {
    pub const fn value(&self) -> i32 {
        match self {
            PieceType::Pawn => 100,
            PieceType::Advisor => 200,
            PieceType::Elephant => 200,
            PieceType::Horse => 400,
            PieceType::Cannon => 450,
            PieceType::Rook => 900,
            PieceType::King => 10000,
        }
    }
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

    pub const fn zobrist_color_index(&self) -> usize {
        match self.color {
            Color::Red => 0,
            Color::Black => 1,
        }
    }

    pub const fn zobrist_type_index(&self) -> usize {
        match self.piece_type {
            PieceType::Pawn => 0,
            PieceType::Advisor => 1,
            PieceType::Elephant => 2,
            PieceType::Horse => 3,
            PieceType::Cannon => 4,
            PieceType::Rook => 5,
            PieceType::King => 6,
        }
    }
}

// 0x88 method but adjusted for Xiangqi 10x9 board
// The actual playable area uses indices where the column is 0..=8 and row is 0..=9
// In a 16-width array, a square index `sq` is valid if `(sq & 0x8F) <= 8`.
// However, since we need 10 rows, a u8 is perfectly fine (max index 16*9+8 = 152).
// To keep things simple and power-of-2 aligned, we use an array of size 256.
#[derive(Clone)]
pub struct Board {
    pub squares: [Option<Piece>; 256],
    pub side_to_move: Color,
    pub red_king_sq: usize,
    pub black_king_sq: usize,
    pub zobrist_key: u64,
}

#[derive(Copy, Clone, Debug)]
pub struct UndoInfo {
    pub captured_piece: Option<Piece>,
    pub previous_red_king_sq: usize,
    pub previous_black_king_sq: usize,
    pub previous_zobrist_key: u64,
}

impl Board {
    pub fn new() -> Self {
        Self {
            squares: [None; 256],
            side_to_move: Color::Red,
            red_king_sq: 0,
            black_king_sq: 0,
            zobrist_key: 0,
        }
    }

    pub fn compute_zobrist_key(&mut self) -> u64 {
        let mut key = 0;
        for sq in 0..256 {
            if let Some(piece) = self.piece_at(sq) {
                let dense_sq = Self::square_to_dense(sq);
                key ^= crate::zobrist::ZOBRIST.pieces[piece.zobrist_color_index()]
                    [piece.zobrist_type_index()][dense_sq];
            }
        }
        if self.side_to_move == Color::Black {
            key ^= crate::zobrist::ZOBRIST.side;
        }
        self.zobrist_key = key;
        key
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();
        for row in 0..10 {
            let mut empty_count = 0;
            for col in 0..9 {
                let sq = Self::coord_to_square(row, col);
                if let Some(piece) = self.piece_at(sq) {
                    if empty_count > 0 {
                        fen.push_str(&empty_count.to_string());
                        empty_count = 0;
                    }
                    let mut c = match piece.piece_type {
                        PieceType::King => 'K',
                        PieceType::Advisor => 'A',
                        PieceType::Elephant => 'B',
                        PieceType::Horse => 'N',
                        PieceType::Rook => 'R',
                        PieceType::Cannon => 'C',
                        PieceType::Pawn => 'P',
                    };
                    if piece.color == Color::Black {
                        c = c.to_ascii_lowercase();
                    }
                    fen.push(c);
                } else {
                    empty_count += 1;
                }
            }
            if empty_count > 0 {
                fen.push_str(&empty_count.to_string());
            }
            if row < 9 {
                fen.push('/');
            }
        }
        let stm = if self.side_to_move == Color::Red {
            "w"
        } else {
            "b"
        };
        format!("{} {}", fen, stm)
    }

    /// Parse chuỗi FEN để tạo ra một Board mới
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let mut board = Board::new();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.is_empty() {
            return Err("Chuỗi FEN trống".to_string());
        }

        let board_part = parts[0];
        let rows: Vec<&str> = board_part.split('/').collect();
        if rows.len() != 10 {
            return Err(format!(
                "FEN không hợp lệ: Cần 10 hàng, nhưng có {}",
                rows.len()
            ));
        }

        // Parse vị trí các quân cờ
        for (r, row_str) in rows.iter().enumerate() {
            let mut c = 0;
            for ch in row_str.chars() {
                if let Some(digit) = ch.to_digit(10) {
                    c += digit as usize;
                } else {
                    let color = if ch.is_lowercase() {
                        Color::Black
                    } else {
                        Color::Red
                    };
                    let piece_type = match ch.to_ascii_lowercase() {
                        'k' => PieceType::King,
                        'a' => PieceType::Advisor,
                        'b' => PieceType::Elephant,
                        'n' => PieceType::Horse,
                        'r' => PieceType::Rook,
                        'c' => PieceType::Cannon,
                        'p' => PieceType::Pawn,
                        _ => return Err(format!("Ký tự không hợp lệ trong FEN: {}", ch)),
                    };

                    let sq = Self::coord_to_square(r, c);
                    board.set_piece(sq, Some(Piece::new(piece_type, color)));

                    // Cập nhật vị trí Tướng
                    if piece_type == PieceType::King {
                        if color == Color::Red {
                            board.red_king_sq = sq;
                        } else {
                            board.black_king_sq = sq;
                        }
                    }
                    c += 1;
                }
            }
        }

        // Parse lượt đi (nếu có)
        if parts.len() > 1 {
            board.side_to_move = match parts[1].to_lowercase().as_str() {
                "b" => Color::Black,
                "w" | "r" => Color::Red,
                _ => return Err("Lượt đi không hợp lệ trong FEN".to_string()),
            };
        } else {
            board.side_to_move = Color::Red;
        }

        board.compute_zobrist_key();
        Ok(board)
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

    pub fn square_to_dense(sq: usize) -> usize {
        (sq / 16) * 9 + (sq % 16)
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
        let (r_row, r_col) = Self::square_to_coord(self.red_king_sq);
        let (b_row, b_col) = Self::square_to_coord(self.black_king_sq);

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

        false
    }

    /// Checks if a specific color's King is under attack by opponent pieces.
    pub fn is_in_check(&self, color: Color) -> bool {
        let king_sq = if color == Color::Red {
            self.red_king_sq
        } else {
            self.black_king_sq
        };
        let opp_color = color.opposite();

        // 1. Check Rook and Cannon attacks
        let dirs = [1, -1, 16, -16];
        for &dir in &dirs {
            let mut current_opt = king_sq.checked_add_signed(dir);
            let mut pieces_between = 0;

            while let Some(current) = current_opt {
                if !Self::is_valid_square(current) {
                    break;
                }

                if let Some(piece) = self.piece_at(current) {
                    if piece.color == opp_color {
                        if pieces_between == 0 && piece.piece_type == PieceType::Rook {
                            return true;
                        }
                        if pieces_between == 1 && piece.piece_type == PieceType::Cannon {
                            return true;
                        }
                    }
                    pieces_between += 1;
                    if pieces_between > 1 {
                        break;
                    }
                }
                current_opt = current.checked_add_signed(dir);
            }
        }

        // 2. Check Horse attacks.
        // Reverse jump: King to Horse offset. Horse needs unblocked leg to reach King.
        let horse_jumps: [isize; 8] = [-33, -31, -14, 18, 33, 31, 14, -18];
        let horse_legs: [isize; 8] = [-16, -16, 1, 1, 16, 16, -1, -1];

        for i in 0..8 {
            let opp_horse_offset = -horse_jumps[i];
            if let Some(opp_horse_sq) = king_sq.checked_add_signed(opp_horse_offset) {
                if Self::is_valid_square(opp_horse_sq) {
                    if let Some(piece) = self.piece_at(opp_horse_sq) {
                        if piece.color == opp_color && piece.piece_type == PieceType::Horse {
                            // Leg is defined from the horse's perspective
                            let leg_sq = opp_horse_sq.checked_add_signed(horse_legs[i]).unwrap();
                            if self.is_empty(leg_sq) {
                                return true;
                            }
                        }
                    }
                }
            }
        }

        // 3. Check Pawn attacks
        // If color == Red, Opponent is Black. Black pawns move down (+16), so check King - 16.
        let opp_forward: isize = if color == Color::Red { -16 } else { 16 };
        for &pawn_offset in &[opp_forward, -1, 1] {
            if let Some(pawn_sq) = king_sq.checked_add_signed(pawn_offset) {
                if Self::is_valid_square(pawn_sq) {
                    if let Some(piece) = self.piece_at(pawn_sq) {
                        if piece.color == opp_color && piece.piece_type == PieceType::Pawn {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }

    /// Determines quickly if a given pseudo-legal move delivers a check.
    /// This is used heavily in Mate Search.
    pub fn is_checking_move(&mut self, m: Move) -> bool {
        let side = self.side_to_move;

        let undo = self.make_move(m);
        let checks = self.is_in_check(side.opposite());
        self.unmake_move(m, undo);

        checks
    }

    pub fn is_defended(&mut self, sq: usize, defending_color: Color) -> bool {
        let original_piece = self.piece_at(sq);
        let mut is_def = false;
        if let Some(mut piece) = original_piece {
            piece.color = defending_color.opposite(); // Pretend it's an enemy
            self.set_piece(sq, Some(piece));

            let original_side = self.side_to_move;
            self.side_to_move = defending_color;
            let captures = self.generate_captures();
            self.side_to_move = original_side;

            self.set_piece(sq, original_piece); // Restore

            for m in captures {
                if m.to_sq() == sq {
                    is_def = true;
                    break;
                }
            }
        }
        is_def
    }

    /// Returns a bitboard of opponent pieces under unprotected attack by `attacker_color`.
    /// Used to compute NEW threats by comparing before/after a move.
    pub fn get_unprotected_threats(&mut self, attacker_color: Color) -> u128 {
        let mut threats: u128 = 0;
        let victim_color = attacker_color.opposite();

        let original_side = self.side_to_move;
        self.side_to_move = attacker_color;
        let attacks = self.generate_captures();
        self.side_to_move = original_side;

        for m in attacks {
            let victim_sq = m.to_sq();
            let attacker_sq = m.from_sq();

            if let Some(victim_piece) = self.piece_at(victim_sq) {
                if let Some(attacker_piece) = self.piece_at(attacker_sq) {
                    if victim_piece.piece_type == PieceType::King {
                        continue;
                    }

                    let is_unprotected =
                        if attacker_piece.piece_type.value() < victim_piece.piece_type.value() {
                            true
                        } else {
                            !self.is_defended(victim_sq, victim_color)
                        };

                    if is_unprotected {
                        let dense_sq = Self::square_to_dense(victim_sq);
                        threats |= 1_u128 << dense_sq;
                    }
                }
            }
        }

        threats
    }

    /// Algorithm 10: Quick prune if draw score (0) >= beta and our side is idle in the cycle.
    /// Avoids computing opponent's violation level entirely.
    pub fn judge_prune(&self, history: &[HistoryEntry], current_ply: usize, beta: i32) -> bool {
        if 0 < beta || history.len() < 4 {
            return false;
        }

        let current_hash = self.zobrist_key;
        let mut i = history.len() as isize - 3;
        while i >= 0 {
            let entry = &history[i as usize];
            if !entry.is_reversible {
                break;
            }
            if entry.hash == current_hash {
                // Found cycle - check if all OUR moves are idle (no check, no chase)
                for idx in (i as usize)..history.len() {
                    if idx % 2 == current_ply % 2 {
                        if history[idx].is_check || history[idx].chased_set != 0 {
                            return false; // We have a violation, can't prune
                        }
                    }
                }
                return true; // All our moves are idle → draw prune
            }
            i -= 2;
        }
        false
    }

    pub fn judge_repetition(
        &self,
        history: &[HistoryEntry],
        current_ply: usize,
        rep_threshold: usize,
    ) -> RepetitionResult {
        if history.len() < 4 {
            return RepetitionResult::Undecided;
        }

        let current_hash = self.zobrist_key;
        let mut rep_count = 0;
        let mut loop_start_index = 0;

        let mut i = history.len() as isize - 3;
        while i >= 0 {
            let entry = &history[i as usize];
            if !entry.is_reversible {
                break;
            }
            if entry.hash == current_hash {
                rep_count += 1;
                loop_start_index = i as usize;
                if rep_count >= rep_threshold {
                    break;
                }
            }
            i -= 2;
        }

        if rep_count < rep_threshold {
            return RepetitionResult::Undecided;
        }

        if rep_count == 0 {
            return RepetitionResult::Undecided;
        }

        let mut our_all_checks = true;
        let mut our_all_chases = true;
        let mut opp_all_checks = true;
        let mut opp_all_chases = true;

        for idx in loop_start_index..history.len() {
            let entry = &history[idx];
            if idx % 2 == current_ply % 2 {
                if !entry.is_check {
                    our_all_checks = false;
                }
                if entry.chased_set == 0 {
                    our_all_chases = false;
                }
            } else {
                if !entry.is_check {
                    opp_all_checks = false;
                }
                if entry.chased_set == 0 {
                    opp_all_chases = false;
                }
            }
        }

        let our_violation = if our_all_checks {
            ViolationLevel::PerpetualCheck
        } else if our_all_chases {
            ViolationLevel::PerpetualChase
        } else {
            ViolationLevel::PerpetualIdle
        };

        let opp_violation = if opp_all_checks {
            ViolationLevel::PerpetualCheck
        } else if opp_all_chases {
            ViolationLevel::PerpetualChase
        } else {
            ViolationLevel::PerpetualIdle
        };

        if our_violation == opp_violation {
            RepetitionResult::Draw
        } else if our_violation > opp_violation {
            RepetitionResult::Loss
        } else {
            RepetitionResult::Win
        }
    }

    /// Evaluates the current board state.
    /// Returns a positive score if `side_to_move` has the advantage.
    /// Returns a negative score if `side_to_move` is at a disadvantage.
    pub fn evaluate(&self) -> i32 {
        let mut score = 0;

        for sq in 0..256 {
            if let Some(piece) = self.piece_at(sq) {
                // Basic Material Values
                let mut piece_val = piece.piece_type.value();

                // Bonus for Pawns crossing the river
                if piece.piece_type == PieceType::Pawn {
                    if !Self::is_on_own_side(sq, piece.color) {
                        piece_val += 100; // Double value for crossed pawn
                    }
                }

                // Add to score if it's our piece, subtract if opponent's
                if piece.color == self.side_to_move {
                    score += piece_val;
                } else {
                    score -= piece_val;
                }
            }
        }

        score
    }

    pub fn make_move(&mut self, m: Move) -> UndoInfo {
        let from = m.from_sq();
        let to = m.to_sq();
        let piece = self.piece_at(from).unwrap();
        let captured = self.piece_at(to);

        let undo = UndoInfo {
            captured_piece: captured,
            previous_red_king_sq: self.red_king_sq,
            previous_black_king_sq: self.black_king_sq,
            previous_zobrist_key: self.zobrist_key,
        };

        let from_dense = Self::square_to_dense(from);
        let to_dense = Self::square_to_dense(to);

        // Remove moving piece from 'from' and add it to 'to' in the hash
        self.zobrist_key ^= crate::zobrist::ZOBRIST.pieces[piece.zobrist_color_index()]
            [piece.zobrist_type_index()][from_dense];
        self.zobrist_key ^= crate::zobrist::ZOBRIST.pieces[piece.zobrist_color_index()]
            [piece.zobrist_type_index()][to_dense];

        // Remove captured piece from the hash
        if let Some(cap) = captured {
            self.zobrist_key ^= crate::zobrist::ZOBRIST.pieces[cap.zobrist_color_index()]
                [cap.zobrist_type_index()][to_dense];
        }

        // Move piece
        self.set_piece(to, Some(piece));
        self.set_piece(from, None);

        // Update king position if moved
        if piece.piece_type == PieceType::King {
            if piece.color == Color::Red {
                self.red_king_sq = to;
            } else {
                self.black_king_sq = to;
            }
        }

        self.side_to_move = self.side_to_move.opposite();
        self.zobrist_key ^= crate::zobrist::ZOBRIST.side;

        undo
    }

    pub fn unmake_move(&mut self, m: Move, undo: UndoInfo) {
        let from = m.from_sq();
        let to = m.to_sq();
        let piece = self.piece_at(to).unwrap();

        // Restore piece to original square
        self.set_piece(from, Some(piece));

        // Restore captured piece (if any)
        self.set_piece(to, undo.captured_piece);

        // Restore king positions
        self.red_king_sq = undo.previous_red_king_sq;
        self.black_king_sq = undo.previous_black_king_sq;

        // Restore zobrist
        self.zobrist_key = undo.previous_zobrist_key;

        // Restore side to move
        self.side_to_move = self.side_to_move.opposite();
    }

    pub fn set_initial_position(&mut self) {
        self.squares = [None; 256];
        self.side_to_move = Color::Red;
        self.red_king_sq = Self::coord_to_square(9, 4);
        self.black_king_sq = Self::coord_to_square(0, 4);

        use Color::*;
        use PieceType::*;

        // Setup Black (Top, Rows 0-3)
        self.set_piece(Self::coord_to_square(0, 0), Some(Piece::new(Rook, Black)));
        self.set_piece(Self::coord_to_square(0, 1), Some(Piece::new(Horse, Black)));
        self.set_piece(
            Self::coord_to_square(0, 2),
            Some(Piece::new(Elephant, Black)),
        );
        self.set_piece(
            Self::coord_to_square(0, 3),
            Some(Piece::new(Advisor, Black)),
        );
        self.set_piece(Self::coord_to_square(0, 4), Some(Piece::new(King, Black)));
        self.set_piece(
            Self::coord_to_square(0, 5),
            Some(Piece::new(Advisor, Black)),
        );
        self.set_piece(
            Self::coord_to_square(0, 6),
            Some(Piece::new(Elephant, Black)),
        );
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

        self.compute_zobrist_key();
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

        board.red_king_sq = red_king_sq;
        board.black_king_sq = black_king_sq;

        board.set_piece(red_king_sq, Some(Piece::new(PieceType::King, Color::Red)));
        board.set_piece(
            black_king_sq,
            Some(Piece::new(PieceType::King, Color::Black)),
        );

        // Kings on the same file with no pieces in between -> facing
        assert!(board.kings_facing());

        // Block the file with a piece
        board.set_piece(
            Board::coord_to_square(5, 4),
            Some(Piece::new(PieceType::Pawn, Color::Red)),
        );
        assert!(!board.kings_facing());

        // Move one king to a different file
        board.set_piece(black_king_sq, None);
        let new_black_king_sq = Board::coord_to_square(0, 3);
        board.black_king_sq = new_black_king_sq;
        board.set_piece(
            new_black_king_sq,
            Some(Piece::new(PieceType::King, Color::Black)),
        );

        // Even without blockers, they are not on the same file -> not facing
        board.set_piece(Board::coord_to_square(5, 4), None);
        assert!(!board.kings_facing());
    }
}
