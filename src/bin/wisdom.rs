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

const MODEL_PATH: &str = "xiangqi_net_weights";

fn start_engine<B: burn::prelude::Backend>(device: B::Device) {
    let config = XiangqiNetConfig::new();

    // Tự động phát hiện file model đã train
    let model = if std::path::Path::new(&format!("{}.mpk", MODEL_PATH)).exists() {
        println!("📦 Phát hiện file model đã huấn luyện!");
        config.load_model::<B>(MODEL_PATH, &device)
    } else {
        println!("⚠️ Cảnh báo: Không tìm thấy file model. Đang dùng model ngẫu nhiên!");
        println!("   (Để có model, hãy chạy train.py hoặc export từ Kaggle)");
        config.init::<B>(&device)
    };

    ucci_loop_generic::<B>(model, device);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Perft mode
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

    // Detect hardware mode from args or default to CPU
    let mode = if args.len() > 1 && args[1] == "gpu" {
        HardwareMode::GpuWgpu
    } else {
        HardwareMode::CpuNdArray
    };

    match mode {
        HardwareMode::CpuNdArray => {
            println!("🚀 Khởi động Wisdom Engine (MCTS + CNN) với CPU (NdArray)...");
            start_engine::<NdArray<f32>>(NdArrayDevice::Cpu);
        }
        HardwareMode::GpuWgpu => {
            println!("🚀 Khởi động Wisdom Engine (MCTS + CNN) với GPU (Wgpu)...");
            start_engine::<Wgpu<f32, i32>>(WgpuDevice::default());
        }
    }
}
