use crate::board::{Board, Color, PieceType};
use burn::nn::{
    BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu,
    conv::{Conv2d, Conv2dConfig},
};
use burn::prelude::*;

/// Total input planes: 7 piece types × 2 colors + 1 side-to-move = 15
pub const NUM_PLANES: usize = 15;
pub const BOARD_H: usize = 10;
pub const BOARD_W: usize = 9;
pub const TENSOR_SIZE: usize = NUM_PLANES * BOARD_H * BOARD_W; // 1350

// ============================================================
// Board → Tensor Conversion
// ============================================================

/// Converts a Board to a flat f32 array of shape [15, 10, 9].
///
/// Plane layout:
///   0: Red King       7: Black King
///   1: Red Advisor    8: Black Advisor
///   2: Red Elephant   9: Black Elephant
///   3: Red Horse     10: Black Horse
///   4: Red Rook      11: Black Rook
///   5: Red Cannon    12: Black Cannon
///   6: Red Pawn      13: Black Pawn
///  14: Side-to-move (all 1.0 if Red to move, all 0.0 if Black)
pub fn board_to_tensor(board: &Board) -> [f32; TENSOR_SIZE] {
    let mut data = [0.0f32; TENSOR_SIZE];

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
                let plane = match piece.color {
                    Color::Red => piece_offset,
                    Color::Black => piece_offset + 7,
                };
                let idx = plane * (BOARD_H * BOARD_W) + row * BOARD_W + col;
                data[idx] = 1.0;
            }
        }
    }

    // Plane 14: side-to-move indicator
    if board.side_to_move == Color::Red {
        let base = 14 * (BOARD_H * BOARD_W);
        for i in 0..(BOARD_H * BOARD_W) {
            data[base + i] = 1.0;
        }
    }

    data
}

// ============================================================
// CNN Model Definition (Value Head Only)
// ============================================================

/// A small CNN that evaluates a Xiangqi board position.
///
/// Architecture:
///   Conv2d(15→64, 3×3, pad=1) → BN → ReLU
///   Conv2d(64→128, 3×3, pad=1) → BN → ReLU
///   Conv2d(128→128, 3×3, pad=1) → BN → ReLU
///   Global Average Pooling → [B, 128]
///   Linear(128→64) → ReLU
///   Linear(64→1) → Tanh
///   Output: value in [-1.0, 1.0]
#[derive(Module, Debug)]
pub struct XiangqiNet<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    conv3: Conv2d<B>,
    bn3: BatchNorm<B, 2>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    relu: Relu,
}

#[derive(Config, Debug)]
pub struct XiangqiNetConfig;

impl XiangqiNetConfig {
    /// Initializes the model with random weights.
    pub fn init<B: Backend>(&self, device: &B::Device) -> XiangqiNet<B> {
        XiangqiNet {
            conv1: Conv2dConfig::new([NUM_PLANES, 64], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            bn1: BatchNormConfig::new(64).init(device),
            conv2: Conv2dConfig::new([64, 128], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            bn2: BatchNormConfig::new(128).init(device),
            conv3: Conv2dConfig::new([128, 128], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            bn3: BatchNormConfig::new(128).init(device),
            fc1: LinearConfig::new(128, 64).init(device),
            fc2: LinearConfig::new(64, 1).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> XiangqiNet<B> {
    /// Forward pass: [B, 15, 10, 9] → [B, 1] (value in [-1, 1])
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        // Conv block 1
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);

        // Conv block 2
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = self.relu.forward(x);

        // Conv block 3
        let x = self.conv3.forward(x);
        let x = self.bn3.forward(x);
        let x = self.relu.forward(x);

        // Global Average Pooling: [B, 128, 10, 9] → [B, 128]
        let [batch, channels, h, w] = x.dims();
        let spatial = h * w;
        let x = x.reshape([batch, channels, spatial]); // [B, 128, 90]
        let x = x.mean_dim(2); // [B, 128, 1]
        let x = x.reshape([batch, channels]); // [B, 128]

        // Value head
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        let x = self.fc2.forward(x); // [B, 1]

        // Tanh to bound output to [-1, 1]
        x.tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_board_to_tensor_initial_position() {
        let mut board = Board::new();
        board.set_initial_position();
        let tensor = board_to_tensor(&board);

        // Red Rook should be at (9, 0) and (9, 8) → plane 4
        let plane_rook_red = 4;
        let idx_a0 = plane_rook_red * 90 + 9 * 9 + 0; // row=9, col=0
        let idx_i0 = plane_rook_red * 90 + 9 * 9 + 8; // row=9, col=8
        assert_eq!(tensor[idx_a0], 1.0, "Red Rook at a0");
        assert_eq!(tensor[idx_i0], 1.0, "Red Rook at i0");

        // Side-to-move plane should be all 1s (Red to move)
        let stm_base = 14 * 90;
        for i in 0..90 {
            assert_eq!(tensor[stm_base + i], 1.0, "STM plane[{}]", i);
        }
    }

    #[test]
    fn test_model_forward_shape() {
        let device = <TestBackend as Backend>::Device::default();
        let config = XiangqiNetConfig;
        let model = config.init::<TestBackend>(&device);

        // Create a dummy batch of 4 boards
        let dummy = Tensor::<TestBackend, 4>::zeros([4, NUM_PLANES, BOARD_H, BOARD_W], &device);
        let output = model.forward(dummy);

        assert_eq!(output.dims(), [4, 1]);

        // Values should be in [-1, 1] due to tanh
        let data = output.to_data();
        for val in data.iter::<f32>() {
            assert!(val >= -1.0 && val <= 1.0, "Output {} out of range", val);
        }
    }
}
