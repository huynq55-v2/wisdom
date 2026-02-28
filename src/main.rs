mod board;
mod r#move;
mod movegen;
mod perft;
mod search;
mod ucci;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "perft" {
        // Run perft test if desired
        let mut board = board::Board::new();
        board.set_initial_position();
        let depth = if args.len() > 2 { args[2].parse().unwrap_or(4) } else { 4 };
        let nodes = perft::perft(&mut board, depth);
        println!("Perft depth {} : {} nodes", depth, nodes);
    } else {
        // Default to UCCI mode
        ucci::ucci_loop();
    }
}
