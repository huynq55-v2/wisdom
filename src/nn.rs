use ndarray::Array4;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

use crate::board::{Board, Color, PieceType};

// ============================================================
// CONSTANTS (Giữ lại để các file khác gọi)
// ============================================================
pub const BOARD_W: usize = 9;
pub const BOARD_H: usize = 10;
pub const NUM_PLANES: usize = 14;
pub const TENSOR_SIZE: usize = NUM_PLANES * BOARD_W * BOARD_H;
pub const ACTION_SPACE: usize = 8100;

// Hàm hỗ trợ chuyển FEN sang mảng 1D (Nếu bác đang dùng trong nn.rs thì giữ lại)
pub fn board_to_tensor(board: &Board) -> [f32; TENSOR_SIZE] {
    let mut data = [0.0f32; TENSOR_SIZE];
    let is_black = board.side_to_move == Color::Black;

    for row in 0..BOARD_H {
        for col in 0..BOARD_W {
            let sq = Board::coord_to_square(row, col);
            if let Some(piece) = board.piece_at(sq) {
                let piece_offset = match piece.piece_type {
                    PieceType::King => 0,
                    PieceType::Advisor => 1,
                    PieceType::Elephant => 2,
                    PieceType::Horse => 3,
                    PieceType::Rook => 4,
                    PieceType::Cannon => 5,
                    PieceType::Pawn => 6,
                };

                let is_mine = piece.color == board.side_to_move;
                let plane = if is_mine {
                    piece_offset
                } else {
                    piece_offset + 7
                };

                let (mapped_row, mapped_col) = if is_black {
                    (9 - row, 8 - col)
                } else {
                    (row, col)
                };

                let idx = plane * (BOARD_H * BOARD_W) + mapped_row * BOARD_W + mapped_col;
                data[idx] = 1.0;
            }
        }
    }

    data
}

// ============================================================
// ONNX MODEL DEFINITION
// ============================================================
pub struct XiangqiOnnx {
    session: Session,
}

impl XiangqiOnnx {
    pub fn new(model_path: &str) -> Self {
        let _ = ort::init().with_name("wisdom_onnx").commit();
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();

        Self { session }
    }

    pub fn forward(&mut self, batch_array: Array4<f32>) -> (Vec<f32>, Vec<f32>) {
        let input_tensor = ort::value::Tensor::from_array(batch_array).unwrap();

        let outputs = self
            .session
            .run(ort::inputs![
                "input" => input_tensor
            ])
            .expect("Lỗi chạy ONNX");

        let policy_output = outputs
            .get("policy")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap();
        let value_output = outputs
            .get("value")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap();

        (value_output.1.to_vec(), policy_output.1.to_vec())
    }
}
