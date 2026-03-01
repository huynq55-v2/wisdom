use crate::nn::{BOARD_H, BOARD_W, NUM_PLANES, TENSOR_SIZE, XiangqiNet};
use burn::prelude::*;
use crossbeam_channel::{Sender, bounded};
use std::thread;

pub struct EvalRequest {
    pub tensor_data: [f32; TENSOR_SIZE],
    pub response_tx: Sender<(f32, Vec<f32>)>,
}

pub struct EvalQueue {
    pub tx: Sender<EvalRequest>,
}

impl EvalQueue {
    /// Spawns a background thread that listens for evaluation requests,
    /// batches them, runs the CNN forward pass, and sends results back.
    pub fn new<B: Backend>(
        model: XiangqiNet<B>,
        device: B::Device,
        batch_size: usize,
        timeout_ms: u64,
    ) -> Self {
        let (tx, rx) = bounded::<EvalRequest>(1024);

        thread::spawn(move || {
            let mut batch_inputs = Vec::with_capacity(batch_size * TENSOR_SIZE);
            let mut response_channels = Vec::with_capacity(batch_size);

            loop {
                batch_inputs.clear();
                response_channels.clear();

                // Block until we get the first request
                // println!("EvalQueue: waiting for req...");
                match rx.recv() {
                    Ok(req) => {
                        batch_inputs.extend_from_slice(&req.tensor_data);
                        response_channels.push(req.response_tx);
                    }
                    Err(_) => break, // Channel closed, exit thread
                }

                // Try to collect more requests up to batch_size, instantly via try_recv
                while response_channels.len() < batch_size {
                    match rx.try_recv() {
                        Ok(req) => {
                            batch_inputs.extend_from_slice(&req.tensor_data);
                            response_channels.push(req.response_tx);
                        }
                        Err(_) => break, // Queue trống, lập tức mang đi chạy GPU luôn!
                    }
                }

                let current_batch_size = response_channels.len();
                if current_batch_size == 0 {
                    continue;
                }

                // println!("EvalQueue: executing batch size {}", current_batch_size);
                use std::io::Write;
                std::io::stdout().flush().unwrap();

                // Load to tensor
                let inputs = Tensor::<B, 1>::from_data(batch_inputs.as_slice(), &device).reshape([
                    current_batch_size,
                    NUM_PLANES,
                    BOARD_H,
                    BOARD_W,
                ]);

                // Forward pass on GPU (value, policy_logits)
                let (pred_value, pred_policy) = model.forward(inputs);

                // Read values back
                let values = pred_value.into_data().to_vec::<f32>().unwrap();
                let policies = pred_policy.into_data().to_vec::<f32>().unwrap();

                // Dispatch results to waiting threads
                for (i, resp_tx) in response_channels.drain(..).enumerate() {
                    let v = values[i];
                    // Slice the 8100 elements for this specific item in the batch
                    let policy_start = i * crate::nn::ACTION_SPACE;
                    let policy_end = policy_start + crate::nn::ACTION_SPACE;
                    let p = policies[policy_start..policy_end].to_vec();
                    let _ = resp_tx.send((v, p));
                }
            }
        });

        Self { tx }
    }
}
