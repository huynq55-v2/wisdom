use crate::board::{Board, Color, PieceType};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, Recorder};
use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    record::NamedMpkFileRecorder,
};

/// Total input planes: 7 piece types × 2 colors = 14
pub const NUM_PLANES: usize = 14;
pub const BOARD_H: usize = 10;
pub const BOARD_W: usize = 9;
pub const TENSOR_SIZE: usize = NUM_PLANES * BOARD_H * BOARD_W; // 1260
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

/// Convert a Move to an Index (0..8099) for the Policy Head, mapped to Canonical Perspective
pub fn move_to_index_perspective(m: crate::r#move::Move, stm: Color) -> usize {
    let mut from_sq = m.from_sq() as usize;
    let mut to_sq = m.to_sq() as usize;

    // Nếu phe Đen đang đi, ta xoay tọa độ 180 độ để map đúng với Policy từ NN
    if stm == Color::Black {
        let f_row = from_sq / 16;
        let f_col = from_sq % 16;
        from_sq = (9 - f_row) * 16 + (8 - f_col);

        let t_row = to_sq / 16;
        let t_col = to_sq % 16;
        to_sq = (9 - t_row) * 16 + (8 - t_col);
    }

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

/// Converts a Board to a flat f32 array of shape [14, 10, 9].
/// Canonical Perspective mapping:
/// If Red to move:
///   Planes 0..=6   : Red pieces
///   Planes 7..=13  : Black pieces
///   Board mapped   : As is
///
/// If Black to move:
///   Planes 0..=6   : Black pieces
///   Planes 7..=13  : Red pieces
///   Board mapped   : Rotated 180 degrees (row -> 9 - row, col -> 8 - col)
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

                // Canonical: Plane 0-6: Side to move, 7-13: Opponent
                let is_mine = piece.color == board.side_to_move;
                let plane = if is_mine {
                    piece_offset
                } else {
                    piece_offset + 7
                };

                // Rotate board 180 degrees if Black is to move
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
// RESNET MODEL DEFINITION (Khớp 100% với train.py)
// ============================================================

#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    relu: Relu,
}

impl<B: Backend> ResBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = x.clone();
        let out = self.conv1.forward(x);
        let out = self.bn1.forward(out);
        let out = self.relu.forward(out);

        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);

        let out = out + residual; // Skip connection
        self.relu.forward(out)
    }
}

pub struct ResBlockConfig {
    channels: usize,
}

impl ResBlockConfig {
    pub fn new(channels: usize) -> Self {
        Self { channels }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> ResBlock<B> {
        ResBlock {
            // bias = false y như Python
            conv1: Conv2dConfig::new([self.channels, self.channels], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .with_bias(false)
                .init(device),
            bn1: BatchNormConfig::new(self.channels).init(device),
            conv2: Conv2dConfig::new([self.channels, self.channels], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .with_bias(false)
                .init(device),
            bn2: BatchNormConfig::new(self.channels).init(device),
            relu: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct XiangqiNet<B: Backend> {
    conv_input: Conv2d<B>,
    bn_input: BatchNorm<B, 2>,
    res_blocks: Vec<ResBlock<B>>, // 7 ResBlocks

    conv_policy: Conv2d<B>,
    policy_head: Linear<B>,

    fc1: Linear<B>,
    value_head: Linear<B>,
    relu: Relu,
}

#[derive(Config, Debug)]
pub struct XiangqiNetConfig;

impl XiangqiNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> XiangqiNet<B> {
        let channels = 128;
        let num_res_blocks = 7;

        let mut res_blocks = Vec::with_capacity(num_res_blocks);
        for _ in 0..num_res_blocks {
            res_blocks.push(ResBlockConfig::new(channels).init(device));
        }

        XiangqiNet {
            conv_input: Conv2dConfig::new([NUM_PLANES, channels], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .with_bias(false)
                .init(device),
            bn_input: BatchNormConfig::new(channels).init(device),
            res_blocks,

            conv_policy: Conv2dConfig::new([channels, 2], [1, 1]).init(device),
            policy_head: LinearConfig::new(2 * BOARD_H * BOARD_W, ACTION_SPACE).init(device),

            fc1: LinearConfig::new(channels, 64).init(device),
            value_head: LinearConfig::new(64, 1).init(device),
            relu: Relu::new(),
        }
    }

    pub fn load_model<B: Backend>(&self, path: &str, device: &B::Device) -> XiangqiNet<B> {
        let model = self.init::<B>(device);
        let full_path = format!("{}.mpk", path);
        println!("🧠 Đang nạp bộ não Native Mpk từ: {}", full_path);

        let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
            .load(full_path.into(), device)
            .expect("LỖI: Không nạp được file .mpk.");

        model.load_record(record)
    }
}

impl<B: Backend> XiangqiNet<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = x.dims()[0];

        let mut x = self.conv_input.forward(x);
        x = self.bn_input.forward(x);
        x = self.relu.forward(x);

        // Chạy qua 7 lớp ResBlock
        for block in self.res_blocks.iter() {
            x = block.forward(x);
        }
        let x_spatial = x;
        let [_, channels, h, w] = x_spatial.dims();

        // --- BRANCH 1: POLICY HEAD ---
        let x_pol = self.conv_policy.forward(x_spatial.clone());
        let x_pol = x_pol.reshape([batch_size, 2 * h * w]);
        let logits_policy = self.policy_head.forward(x_pol);

        // --- BRANCH 2: VALUE HEAD ---
        let spatial = h * w;
        let x_val = x_spatial.reshape([batch_size, channels, spatial]);
        let x_val = x_val.mean_dim(2); // GAP
        let x_val = x_val.reshape([batch_size, channels]);
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

        // Red Rook should be at (9, 0) and (9, 8) → plane 4 (it is Red's turn, so plane is 4)
        let plane_rook_red = 4;
        let idx_a0 = plane_rook_red * 90 + 9 * 9 + 0; // row=9, col=0
        let idx_i0 = plane_rook_red * 90 + 9 * 9 + 8; // row=9, col=8
        assert_eq!(tensor[idx_a0], 1.0, "Red Rook at a0");
        assert_eq!(tensor[idx_i0], 1.0, "Red Rook at i0");
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
