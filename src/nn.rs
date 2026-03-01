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
pub const ACTION_SPACE: usize = 90 * 90; // 8100

// ============================================================
// Action Mapping
// ============================================================

/// Convert a Move to an Index (0..8099) for the Policy Head
pub fn move_to_index(m: crate::r#move::Move) -> usize {
    let from_sq = m.from_sq() as usize;
    let to_sq = m.to_sq() as usize;

    let from_dense = (from_sq / 16) * 9 + (from_sq % 16);
    let to_dense = (to_sq / 16) * 9 + (to_sq % 16);

    from_dense * 90 + to_dense
}

/// Convert an Index from the Policy Head back to a (from_sq, to_sq) tuple
pub fn index_to_move(index: usize) -> (u8, u8) {
    let from_dense = index / 90;
    let to_dense = index % 90;

    let from_sq = (from_dense / 9) * 16 + (from_dense % 9);
    let to_sq = (to_dense / 9) * 16 + (to_dense % 9);

    (from_sq as u8, to_sq as u8)
}

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
///   ├─ Policy: Conv2d(128→2, 1×1) → Flatten [B,180] → Linear(180→8100)
///   └─ Value:  GAP [B,128] → Linear(128→64) → ReLU → Linear(64→1) → Tanh
#[derive(Module, Debug)]
pub struct XiangqiNet<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    conv3: Conv2d<B>,
    bn3: BatchNorm<B, 2>,
    conv_policy: Conv2d<B>,
    policy_head: Linear<B>,
    fc1: Linear<B>,
    value_head: Linear<B>,
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
            conv_policy: Conv2dConfig::new([128, 2], [1, 1]).init(device),
            policy_head: LinearConfig::new(2 * BOARD_H * BOARD_W, ACTION_SPACE).init(device),
            fc1: LinearConfig::new(128, 64).init(device),
            value_head: LinearConfig::new(64, 1).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> XiangqiNet<B> {
    /// Forward pass: [B, 15, 10, 9] → (Value [B, 1], Policy [B, 8100])
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Conv block 1
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);

        // Conv block 2
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = self.relu.forward(x);

        // Conv block 3 → x_spatial: [B, 128, 10, 9]
        let x_spatial = self.conv3.forward(x);
        let x_spatial = self.bn3.forward(x_spatial);
        let x_spatial = self.relu.forward(x_spatial);

        let [batch, channels, h, w] = x_spatial.dims();

        // --- BRANCH 1: POLICY HEAD (preserves spatial coordinates) ---
        let x_pol = self.conv_policy.forward(x_spatial.clone()); // [B, 2, 10, 9]
        let x_pol = x_pol.reshape([batch, 2 * h * w]); // [B, 180]
        let logits_policy = self.policy_head.forward(x_pol);

        // --- BRANCH 2: VALUE HEAD (Global Average Pooling) ---
        let spatial = h * w;
        let x_val = x_spatial.reshape([batch, channels, spatial]); // [B, 128, 90]
        let x_val = x_val.mean_dim(2); // [B, 128, 1]
        let x_val = x_val.reshape([batch, channels]); // [B, 128]
        let x_val = self.fc1.forward(x_val);
        let x_val = self.relu.forward(x_val);
        let value = self.value_head.forward(x_val).tanh();

        (value, logits_policy)
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
        let (value, _policy) = model.forward(dummy);

        assert_eq!(value.dims(), [4, 1]);

        // Values should be in [-1, 1] due to tanh
        let data = value.to_data();
        for val in data.iter::<f32>() {
            assert!(val >= -1.0 && val <= 1.0, "Output {} out of range", val);
        }
    }
}
