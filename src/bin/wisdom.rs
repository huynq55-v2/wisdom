use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::backend::{NdArray, ndarray::NdArrayDevice};
use wisdom::board::Board;
use wisdom::nn::XiangqiNetConfig;
use wisdom::perft::perft;
use wisdom::ucci::ucci_loop_generic;

pub enum HardwareMode {
    CpuNdArray,
    GpuWgpu,
}

fn start_engine<B: burn::prelude::Backend>(device: B::Device) {
    let config = XiangqiNetConfig::new();
    let model = config.init::<B>(&device);
    ucci_loop_generic::<B>(model, device);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "perft" {
        let mut board = Board::new();
        board.set_initial_position();
        let depth = if args.len() > 2 {
            args[2].parse().unwrap_or(4)
        } else {
            4
        };
        let nodes = perft(&mut board, depth);
        println!("Perft depth {} : {} nodes", depth, nodes);
        return;
    }

    let mode = HardwareMode::CpuNdArray; // TODO: read from config
    match mode {
        HardwareMode::CpuNdArray => {
            println!("🚀 Khởi động Engine với CPU (NdArray)...");
            start_engine::<NdArray<f32>>(NdArrayDevice::Cpu);
        }
        HardwareMode::GpuWgpu => {
            println!("🚀 Khởi động Engine với GPU (Wgpu)...");
            start_engine::<Wgpu<f32, i32>>(WgpuDevice::default());
        }
    }
}
