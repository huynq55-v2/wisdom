use burn::{
    backend::{Autodiff, Wgpu},
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    nn::loss::MseLoss,
    optim::AdamConfig,
    prelude::*,
    train::{LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use wisdom::nn::{BOARD_H, BOARD_W, NUM_PLANES, TENSOR_SIZE, XiangqiNet, XiangqiNetConfig};

// ==========================================================
// 1. Data Structures and Batcher
// ==========================================================

#[derive(Clone, Debug)]
pub struct SelfPlayItem {
    pub board: [f32; TENSOR_SIZE],
    pub value: f32, // -1.0 to 1.0 (Loss, Draw, Win)
}

pub struct DummyDataset {
    items: Vec<SelfPlayItem>,
}

impl DummyDataset {
    pub fn new(size: usize) -> Self {
        let mut items = Vec::with_capacity(size);
        for _ in 0..size {
            // Random dummy data
            let mut board = [0.0; TENSOR_SIZE];
            board[0] = 1.0;
            items.push(SelfPlayItem { board, value: 0.5 });
        }
        Self { items }
    }
}

impl Dataset<SelfPlayItem> for DummyDataset {
    fn get(&self, index: usize) -> Option<SelfPlayItem> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

pub struct XiangqiBatch<B: Backend> {
    pub inputs: Tensor<B, 4>,
    pub targets: Tensor<B, 2>,
}

impl<B: Backend> Clone for XiangqiBatch<B> {
    fn clone(&self) -> Self {
        Self {
            inputs: self.inputs.clone(),
            targets: self.targets.clone(),
        }
    }
}

impl<B: Backend> std::fmt::Debug for XiangqiBatch<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "XiangqiBatch")
    }
}

#[derive(Clone)]
pub struct XiangqiBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> XiangqiBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<SelfPlayItem, XiangqiBatch<B>> for XiangqiBatcher<B> {
    fn batch(&self, items: Vec<SelfPlayItem>) -> XiangqiBatch<B> {
        let batch_size = items.len();

        // Flatten inputs
        let mut inputs_flat = Vec::with_capacity(batch_size * TENSOR_SIZE);
        let mut targets_flat = Vec::with_capacity(batch_size);

        for item in items {
            inputs_flat.extend_from_slice(&item.board);
            targets_flat.push(item.value);
        }

        let inputs = Tensor::<B, 1>::from_data(inputs_flat.as_slice(), &self.device)
            .reshape([batch_size, NUM_PLANES, BOARD_H, BOARD_W]);

        let targets = Tensor::<B, 1>::from_data(targets_flat.as_slice(), &self.device)
            .reshape([batch_size, 1]);

        XiangqiBatch { inputs, targets }
    }
}

// ==========================================================
// 2. Training Steps Implementation
// ==========================================================

impl<B: burn::tensor::backend::AutodiffBackend> TrainStep<XiangqiBatch<B>, RegressionOutput<B>>
    for XiangqiNet<B>
{
    fn step(&self, batch: XiangqiBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let predictions = self.forward(batch.inputs);

        let loss = MseLoss::new().forward(
            predictions.clone(),
            batch.targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        TrainOutput::new(
            self,
            loss.backward(),
            RegressionOutput {
                loss,
                output: predictions,
                targets: batch.targets,
            },
        )
    }
}

impl<B: Backend> ValidStep<XiangqiBatch<B>, RegressionOutput<B>> for XiangqiNet<B> {
    fn step(&self, batch: XiangqiBatch<B>) -> RegressionOutput<B> {
        let predictions = self.forward(batch.inputs);

        let loss = MseLoss::new().forward(
            predictions.clone(),
            batch.targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        RegressionOutput {
            loss,
            output: predictions,
            targets: batch.targets,
        }
    }
}

// ==========================================================
// 3. Training Loop
// ==========================================================

fn main() {
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    println!("Starting training with dummy dataset...");

    let batch_size = 32;
    let batcher_train = XiangqiBatcher::<MyAutodiffBackend>::new(device.clone());
    let batcher_valid = XiangqiBatcher::<MyBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(2)
        .build(DummyDataset::new(1000));

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(2)
        .build(DummyDataset::new(200));

    let config = XiangqiNetConfig;
    let optim = AdamConfig::new();

    let learner = LearnerBuilder::new("/tmp/wisdom_models")
        .with_file_checkpointer(burn::record::CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(2)
        .summary()
        .build(config.init(&device), optim.init(), 0.001);

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    model_trained
        .save_file(
            "/tmp/wisdom_models/xiangqi_net",
            &burn::record::CompactRecorder::new(),
        )
        .expect("Failed to save model");

    println!("Training completed! Model saved to /tmp/wisdom_models/xiangqi_net");
}
