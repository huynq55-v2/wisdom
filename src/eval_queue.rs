use crate::nn::{BOARD_H, BOARD_W, NUM_PLANES, TENSOR_SIZE, XiangqiNet};
use burn::prelude::*;
use crossbeam_channel::{Receiver, Sender, bounded};
use std::thread;
use std::time::{Duration, Instant};

pub struct EvalRequest {
    pub tensor_data: [f32; TENSOR_SIZE],
    pub response_tx: Sender<f32>,
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
            let timeout = Duration::from_millis(timeout_ms);

            loop {
                batch_inputs.clear();
                response_channels.clear();

                // Block until we get the first request
                match rx.recv() {
                    Ok(req) => {
                        batch_inputs.extend_from_slice(&req.tensor_data);
                        response_channels.push(req.response_tx);
                    }
                    Err(_) => break, // Channel closed, exit thread
                }

                // Try to collect more requests up to batch_size or timeout
                let deadline = Instant::now() + timeout;
                while response_channels.len() < batch_size {
                    let now = Instant::now();
                    if now >= deadline {
                        break;
                    }
                    match rx.recv_timeout(deadline - now) {
                        Ok(req) => {
                            batch_inputs.extend_from_slice(&req.tensor_data);
                            response_channels.push(req.response_tx);
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => break,
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                    }
                }

                let current_batch_size = response_channels.len();
                if current_batch_size == 0 {
                    continue;
                }

                // Load to tensor
                let inputs = Tensor::<B, 1>::from_data(batch_inputs.as_slice(), &device).reshape([
                    current_batch_size,
                    NUM_PLANES,
                    BOARD_H,
                    BOARD_W,
                ]);

                // Forward pass on GPU
                let predictions = model.forward(inputs);

                // Read values back
                // This call may block waiting for GPU synchronization
                let values = predictions.into_data().to_vec::<f32>().unwrap();

                // Dispatch results to waiting threads
                for (i, resp_tx) in response_channels.drain(..).enumerate() {
                    let _ = resp_tx.send(values[i]);
                }
            }
        });

        Self { tx }
    }
}
