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
pub const ACTION_SPACE: usize = 8100; // ĐÃ TĂNG LÊN 8100 (90 x 90)

// ============================================================
// Action Mapping (8100 ACTION SPACE)
// ============================================================

/// Convert a Move to an Absolute Index (0..8099).
pub fn move_to_index(m: crate::r#move::Move) -> usize {
    let from_sq = m.from_sq() as usize;
    let to_sq = m.to_sq() as usize;

    let from_sq90 = (from_sq / 16) * 9 + (from_sq % 16);
    let to_sq90 = (to_sq / 16) * 9 + (to_sq % 16);

    from_sq90 * 90 + to_sq90
}

// Hàm hỗ trợ lật tọa độ trên bàn cờ 16x16 (0x88)
pub fn flip_square(sq: usize) -> usize {
    let r = sq / 16;
    let c = sq % 16;
    let flipped_r = 9 - r;
    let flipped_c = 8 - c;
    flipped_r * 16 + flipped_c
}

// ============================================================
// Board → Tensor Conversion
// ============================================================

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
// RESNET MODEL DEFINITION (Khớp 100% với train.py V3)
// ============================================================

#[derive(Module, Debug)]
pub struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
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
    bn_input: BatchNorm<B>,
    res_blocks: Vec<ResBlock<B>>, // 15 ResBlocks

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
        let num_res_blocks = 8; // Tùy chỉnh số lượng blocks

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
            // ACTION_SPACE BÂY GIỜ LÀ 8100
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

        for block in self.res_blocks.iter() {
            x = block.forward(x);
        }
        let x_spatial = x;
        let [_, channels, h, w] = x_spatial.dims();

        let x_pol = self.conv_policy.forward(x_spatial.clone());
        let x_pol = x_pol.reshape([batch_size, 2 * h * w]);
        let logits_policy = self.policy_head.forward(x_pol);

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

// ============================================================
// Training Support (for burn 0.20)
// ============================================================

use burn::train::{InferenceStep, TrainOutput, TrainStep};

#[derive(Clone, Debug)]
pub struct XiangqiTrainingOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub pred_value: Tensor<B, 2>,
    pub targets_v: Tensor<B, 2>,
    pub pred_policy: Tensor<B, 2>,
    pub targets_p: Tensor<B, 1, burn::tensor::Int>,
}

impl<B: Backend> burn::train::ItemLazy for XiangqiTrainingOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: Backend> burn::train::metric::Adaptor<burn::train::metric::AccuracyInput<B>> for XiangqiTrainingOutput<B> {
    fn adapt(&self) -> burn::train::metric::AccuracyInput<B> {
        burn::train::metric::AccuracyInput::new(self.pred_policy.clone(), self.targets_p.clone())
    }
}

impl<B: Backend> burn::train::metric::Adaptor<burn::train::metric::LossInput<B>> for XiangqiTrainingOutput<B> {
    fn adapt(&self) -> burn::train::metric::LossInput<B> {
        burn::train::metric::LossInput::new(self.loss.clone())
    }
}

#[derive(Clone, Debug)]
pub struct XiangqiTrainingBatch<B: Backend> {
    pub inputs: Tensor<B, 4>,
    pub targets_v: Tensor<B, 2>,
    pub targets_p: Tensor<B, 1, burn::tensor::Int>,
}

impl<B: burn::tensor::backend::AutodiffBackend> TrainStep for XiangqiNet<B> {
    type Input = XiangqiTrainingBatch<B>;
    type Output = XiangqiTrainingOutput<B::InnerBackend>;

    fn step(
        &self,
        batch: XiangqiTrainingBatch<B>,
    ) -> TrainOutput<XiangqiTrainingOutput<B::InnerBackend>> {
        use burn::nn::loss::{CrossEntropyLossConfig, MseLoss};

        let (pred_value, pred_policy) = self.forward(batch.inputs);

        let loss_v = MseLoss::new().forward(
            pred_value.clone(),
            batch.targets_v.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        let loss_p = CrossEntropyLossConfig::new()
            .init(&batch.targets_p.device())
            .forward(pred_policy.clone(), batch.targets_p.clone());

        let loss = loss_v + loss_p;

        TrainOutput::new(
            self,
            loss.backward(),
            XiangqiTrainingOutput {
                loss: loss.inner(),
                pred_value: pred_value.inner(),
                targets_v: batch.targets_v.inner(),
                pred_policy: pred_policy.inner(),
                targets_p: batch.targets_p.clone().inner(),
            },
        )
    }
}

impl<B: Backend> InferenceStep for XiangqiNet<B> {
    type Input = XiangqiTrainingBatch<B>;
    type Output = XiangqiTrainingOutput<B>;

    fn step(&self, batch: XiangqiTrainingBatch<B>) -> XiangqiTrainingOutput<B> {
        use burn::nn::loss::{CrossEntropyLossConfig, MseLoss};

        let (pred_value, pred_policy) = self.forward(batch.inputs);

        let loss_v = MseLoss::new().forward(
            pred_value.clone(),
            batch.targets_v.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        let loss_p = CrossEntropyLossConfig::new()
            .init(&batch.targets_p.device())
            .forward(pred_policy.clone(), batch.targets_p.clone());

        let loss = loss_v + loss_p;

        XiangqiTrainingOutput {
            loss,
            pred_value,
            targets_v: batch.targets_v,
            pred_policy,
            targets_p: batch.targets_p,
        }
    }
}
