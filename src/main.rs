mod board;
mod r#move;
mod movegen;

use board::Board;

fn main() {
    let mut board = Board::new();
    board.set_initial_position();

    let quiets = board.generate_quiets();
    let captures = board.generate_captures();

    println!("Initial Board Setup Complete.");
    println!("Number of possible quiet moves: {}", quiets.len());
    println!("Number of possible capture moves: {}", captures.len());

    // Basic perft/visualization can be added here
}
