use macroquad::prelude::*;
use wisdom::board::{Board, Color as PieceColor, Piece, PieceType};
use wisdom::r#move::Move;
use wisdom::search::search_best_move;

const SQUARE_SIZE: f32 = 60.0;
const OFFSET_X: f32 = 50.0;
const OFFSET_Y: f32 = 50.0;
const RADIUS: f32 = 25.0;

fn get_legal_moves(board: &mut Board) -> Vec<Move> {
    let mut moves = board.generate_captures();
    let mut quiets = board.generate_quiets();
    moves.append(&mut quiets);
    let moving_side = board.side_to_move;

    moves.into_iter().filter(|&m| {
        let undo = board.make_move(m);
        let valid = !board.kings_facing() && !board.is_in_check(moving_side);
        board.unmake_move(m, undo);
        valid
    }).collect()
}

fn draw_board() {
    clear_background(Color::new(0.9, 0.8, 0.6, 1.0)); // Wood color

    // Draw lines
    for col in 0..9 {
        let x = OFFSET_X + col as f32 * SQUARE_SIZE;
        draw_line(x, OFFSET_Y, x, OFFSET_Y + 4.0 * SQUARE_SIZE, 2.0, BLACK);
        draw_line(x, OFFSET_Y + 5.0 * SQUARE_SIZE, x, OFFSET_Y + 9.0 * SQUARE_SIZE, 2.0, BLACK);
    }
    // Edges across the river
    draw_line(OFFSET_X, OFFSET_Y + 4.0 * SQUARE_SIZE, OFFSET_X + 8.0 * SQUARE_SIZE, OFFSET_Y + 4.0 * SQUARE_SIZE, 2.0, BLACK);
    draw_line(OFFSET_X, OFFSET_Y + 5.0 * SQUARE_SIZE, OFFSET_X + 8.0 * SQUARE_SIZE, OFFSET_Y + 5.0 * SQUARE_SIZE, 2.0, BLACK);
    draw_line(OFFSET_X, OFFSET_Y + 4.0 * SQUARE_SIZE, OFFSET_X, OFFSET_Y + 5.0 * SQUARE_SIZE, 2.0, BLACK);
    draw_line(OFFSET_X + 8.0 * SQUARE_SIZE, OFFSET_Y + 4.0 * SQUARE_SIZE, OFFSET_X + 8.0 * SQUARE_SIZE, OFFSET_Y + 5.0 * SQUARE_SIZE, 2.0, BLACK);

    for row in 0..10 {
        let y = OFFSET_Y + row as f32 * SQUARE_SIZE;
        draw_line(OFFSET_X, y, OFFSET_X + 8.0 * SQUARE_SIZE, y, 2.0, BLACK);
    }

    // Palaces
    draw_line(OFFSET_X + 3.0 * SQUARE_SIZE, OFFSET_Y, OFFSET_X + 5.0 * SQUARE_SIZE, OFFSET_Y + 2.0 * SQUARE_SIZE, 2.0, BLACK);
    draw_line(OFFSET_X + 5.0 * SQUARE_SIZE, OFFSET_Y, OFFSET_X + 3.0 * SQUARE_SIZE, OFFSET_Y + 2.0 * SQUARE_SIZE, 2.0, BLACK);
    
    draw_line(OFFSET_X + 3.0 * SQUARE_SIZE, OFFSET_Y + 7.0 * SQUARE_SIZE, OFFSET_X + 5.0 * SQUARE_SIZE, OFFSET_Y + 9.0 * SQUARE_SIZE, 2.0, BLACK);
    draw_line(OFFSET_X + 5.0 * SQUARE_SIZE, OFFSET_Y + 7.0 * SQUARE_SIZE, OFFSET_X + 3.0 * SQUARE_SIZE, OFFSET_Y + 9.0 * SQUARE_SIZE, 2.0, BLACK);
}

fn draw_pieces(board: &Board, selected_sq: Option<usize>, legal_moves: &[Move]) {
    for row in 0..10 {
        for col in 0..9 {
            let sq = Board::coord_to_square(row, col);
            let x = OFFSET_X + col as f32 * SQUARE_SIZE;
            let y = OFFSET_Y + row as f32 * SQUARE_SIZE;

            // Highlight selected square
            if Some(sq) == selected_sq {
                draw_circle(x, y, RADIUS + 5.0, YELLOW);
            }

            // Highlight valid move targets
            if selected_sq.is_some() && legal_moves.iter().any(|m| m.to_sq() == sq) {
                draw_circle(x, y, 10.0, GREEN);
            }

            if let Some(piece) = board.piece_at(sq) {
                draw_circle(x, y, RADIUS, if piece.color == PieceColor::Red { RED } else { BLACK });
                draw_circle_lines(x, y, RADIUS, 2.0, WHITE);

                let text = match piece.piece_type {
                    PieceType::King => "K",
                    PieceType::Advisor => "A",
                    PieceType::Elephant => "E",
                    PieceType::Horse => "H",
                    PieceType::Rook => "R",
                    PieceType::Cannon => "C",
                    PieceType::Pawn => "P",
                };

                let text_size = measure_text(text, None, 30, 1.0);
                draw_text(text, x - text_size.width / 2.0, y + text_size.height / 2.0, 30.0, WHITE);
            }
        }
    }
}

// Window conf
fn window_conf() -> Conf {
    Conf {
        window_title: "Wisdom Engine - Xiangqi".to_owned(),
        window_width: 600,
        window_height: 700,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut board = Board::new();
    board.set_initial_position();

    let mut selected_sq: Option<usize> = None;
    let mut legal_moves = Vec::new();
    let mut game_over = false;

    // We play as Red (bottom)
    let human_color = PieceColor::Red;

    loop {
        draw_board();
        draw_pieces(&board, selected_sq, &legal_moves);

        if !game_over {
            if board.side_to_move == human_color {
                // Human turn
                if is_mouse_button_pressed(MouseButton::Left) {
                    let (mx, my) = mouse_position();
                    let col = ((mx - OFFSET_X + SQUARE_SIZE / 2.0) / SQUARE_SIZE).floor() as isize;
                    let row = ((my - OFFSET_Y + SQUARE_SIZE / 2.0) / SQUARE_SIZE).floor() as isize;

                    if col >= 0 && col < 9 && row >= 0 && row < 10 {
                        let sq = Board::coord_to_square(row as usize, col as usize);

                        if let Some(selected) = selected_sq {
                            // Try to execute move
                            if let Some(&m) = legal_moves.iter().find(|m| m.to_sq() == sq) {
                                board.make_move(m);
                                selected_sq = None;
                                legal_moves.clear();
                            } else {
                                // Select another piece if valid
                                if let Some(piece) = board.piece_at(sq) {
                                    if piece.color == human_color {
                                        selected_sq = Some(sq);
                                        legal_moves = get_legal_moves(&mut board)
                                            .into_iter()
                                            .filter(|m| m.from_sq() == sq)
                                            .collect();
                                    } else {
                                        selected_sq = None;
                                        legal_moves.clear();
                                    }
                                } else {
                                    selected_sq = None;
                                    legal_moves.clear();
                                }
                            }
                        } else {
                            // Select piece
                            if let Some(piece) = board.piece_at(sq) {
                                if piece.color == human_color {
                                    selected_sq = Some(sq);
                                    legal_moves = get_legal_moves(&mut board)
                                        .into_iter()
                                        .filter(|m| m.from_sq() == sq)
                                        .collect();
                                }
                            }
                        }
                    }
                }
            } else {
                // Engine turn
                // Wait for a frame to render human's move
                next_frame().await;
                
                let all_moves = get_legal_moves(&mut board);
                if all_moves.is_empty() {
                    game_over = true;
                } else {
                    let mut tt = wisdom::tt::TranspositionTable::new(64);
                    let best_move = search_best_move(&mut board, 4, &mut tt);
                    board.make_move(best_move);
                }
            }
        } else {
            draw_text("GAME OVER", OFFSET_X, OFFSET_Y / 2.0, 50.0, RED);
        }

        next_frame().await
    }
}
