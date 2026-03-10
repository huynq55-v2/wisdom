use wisdom::board::Board;
use wisdom::perft::perft;
use wisdom::ucci::ucci_loop;

const MODEL_PATH: &str = "./wisdom_models/wisdom_net_base.onnx";

fn start_engine() {
    println!("📦 Khởi tạo ONNX model từ: {}", MODEL_PATH);
    let model = wisdom::nn::XiangqiOnnx::new(MODEL_PATH);
    ucci_loop(model);
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

    println!("🚀 Khởi động Wisdom Engine (MCTS + ONNX)...");
    start_engine();
}
