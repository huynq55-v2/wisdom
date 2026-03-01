use crate::nn::{BOARD_H, BOARD_W, NUM_PLANES, TENSOR_SIZE, XiangqiNet};
use burn::prelude::*;
use crossbeam_channel::{Sender, bounded};
use std::thread;

pub struct EvalRequest {
    pub tensor_data: [f32; TENSOR_SIZE],
    pub response_tx: Sender<(f32, Option<Vec<f32>>)>,
    pub need_policy: bool,
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
            let mut requests: Vec<EvalRequest> = Vec::with_capacity(batch_size);

            loop {
                batch_inputs.clear();
                requests.clear();

                // Block until we get the first request
                match rx.recv() {
                    Ok(req) => {
                        batch_inputs.extend_from_slice(&req.tensor_data);
                        requests.push(req);
                    }
                    Err(_) => break, // Channel closed, exit thread
                }

                // Collect more requests up to batch_size with a small timeout
                let batch_deadline =
                    std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
                while requests.len() < batch_size {
                    let now = std::time::Instant::now();
                    if now >= batch_deadline {
                        break;
                    }
                    match rx.recv_timeout(batch_deadline - now) {
                        Ok(req) => {
                            batch_inputs.extend_from_slice(&req.tensor_data);
                            requests.push(req);
                        }
                        Err(_) => break, // Timeout or disconnected
                    }
                }

                let current_batch_size = requests.len();
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

                // Forward pass on GPU (value, policy_logits)
                let (pred_value, pred_policy) = model.forward(inputs);

                // Read values back
                let values = pred_value.into_data().to_vec::<f32>().unwrap();

                // Only copy policy from GPU if at least one request needs it
                let any_needs_policy = requests.iter().any(|r| r.need_policy);
                let policies = if any_needs_policy {
                    Some(pred_policy.into_data().to_vec::<f32>().unwrap())
                } else {
                    drop(pred_policy); // Free GPU memory immediately
                    None
                };

                // Dispatch results to waiting threads
                for (i, req) in requests.drain(..).enumerate() {
                    let v = values[i];
                    let p = if req.need_policy {
                        let start = i * crate::nn::ACTION_SPACE;
                        let end = start + crate::nn::ACTION_SPACE;
                        Some(policies.as_ref().unwrap()[start..end].to_vec())
                    } else {
                        None
                    };
                    let _ = req.response_tx.send((v, p));
                }
            }
        });

        Self { tx }
    }
}
