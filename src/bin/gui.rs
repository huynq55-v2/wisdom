use macroquad::prelude::*;
use wisdom::board::{Board, Color as PieceColor, HistoryEntry, PieceType, RepetitionResult};
use wisdom::r#move::Move;
use wisdom::search::alphabeta_search_best_move;

const SQUARE_SIZE: f32 = 60.0;
const OFFSET_X: f32 = 50.0;
const OFFSET_Y: f32 = 50.0;
const RADIUS: f32 = 25.0;

#[derive(PartialEq)]
enum GameMode {
    EngineVsPlayer,
    EngineVsEngine,
}

fn get_legal_moves(board: &mut Board) -> Vec<Move> {
    let mut moves = board.generate_captures();
    let mut quiets = board.generate_quiets();
    moves.append(&mut quiets);
    let moving_side = board.side_to_move;

    moves
        .into_iter()
        .filter(|&m| {
            let undo = board.make_move(m);
            let valid = !board.kings_facing() && !board.is_in_check(moving_side);
            board.unmake_move(m, undo);
            valid
        })
        .collect()
}

fn apply_move_to_game(board: &mut Board, m: Move, history: &mut Vec<HistoryEntry>) {
    let is_capture = !board.is_empty(m.to_sq());
    let piece = board.piece_at(m.from_sq()).unwrap();
    let is_reversible = !is_capture
        && (piece.piece_type != PieceType::Pawn || {
            let (from_row, _) = Board::square_to_coord(m.from_sq());
            let (to_row, _) = Board::square_to_coord(m.to_sq());
            from_row == to_row
        });
    let moving_side = board.side_to_move;

    // Compute pre-move threats BEFORE making the move
    let pre_threats = if is_reversible {
        board.get_unprotected_threats(moving_side)
    } else {
        0
    };

    board.make_move(m);

    let gives_check = board.is_in_check(board.side_to_move);

    // Only NEW threats count as "chase" per WXF rules
    let chased_set = if is_reversible && !gives_check {
        let post_threats = board.get_unprotected_threats(moving_side);
        post_threats & !pre_threats
    } else {
        0
    };

    history.push(HistoryEntry {
        hash: board.zobrist_key,
        is_check: gives_check,
        chased_set,
        is_reversible,
    });
}

fn draw_board() {
    clear_background(Color::new(0.9, 0.8, 0.6, 1.0)); // Wood color

    // Draw lines
    for col in 0..9 {
        let x = OFFSET_X + col as f32 * SQUARE_SIZE;
        draw_line(x, OFFSET_Y, x, OFFSET_Y + 4.0 * SQUARE_SIZE, 2.0, BLACK);
        draw_line(
            x,
            OFFSET_Y + 5.0 * SQUARE_SIZE,
            x,
            OFFSET_Y + 9.0 * SQUARE_SIZE,
            2.0,
            BLACK,
        );
    }
    // Edges across the river
    draw_line(
        OFFSET_X,
        OFFSET_Y + 4.0 * SQUARE_SIZE,
        OFFSET_X + 8.0 * SQUARE_SIZE,
        OFFSET_Y + 4.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );
    draw_line(
        OFFSET_X,
        OFFSET_Y + 5.0 * SQUARE_SIZE,
        OFFSET_X + 8.0 * SQUARE_SIZE,
        OFFSET_Y + 5.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );
    draw_line(
        OFFSET_X,
        OFFSET_Y + 4.0 * SQUARE_SIZE,
        OFFSET_X,
        OFFSET_Y + 5.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );
    draw_line(
        OFFSET_X + 8.0 * SQUARE_SIZE,
        OFFSET_Y + 4.0 * SQUARE_SIZE,
        OFFSET_X + 8.0 * SQUARE_SIZE,
        OFFSET_Y + 5.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );

    for row in 0..10 {
        let y = OFFSET_Y + row as f32 * SQUARE_SIZE;
        draw_line(OFFSET_X, y, OFFSET_X + 8.0 * SQUARE_SIZE, y, 2.0, BLACK);
    }

    // Palaces
    draw_line(
        OFFSET_X + 3.0 * SQUARE_SIZE,
        OFFSET_Y,
        OFFSET_X + 5.0 * SQUARE_SIZE,
        OFFSET_Y + 2.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );
    draw_line(
        OFFSET_X + 5.0 * SQUARE_SIZE,
        OFFSET_Y,
        OFFSET_X + 3.0 * SQUARE_SIZE,
        OFFSET_Y + 2.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );

    draw_line(
        OFFSET_X + 3.0 * SQUARE_SIZE,
        OFFSET_Y + 7.0 * SQUARE_SIZE,
        OFFSET_X + 5.0 * SQUARE_SIZE,
        OFFSET_Y + 9.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );
    draw_line(
        OFFSET_X + 5.0 * SQUARE_SIZE,
        OFFSET_Y + 7.0 * SQUARE_SIZE,
        OFFSET_X + 3.0 * SQUARE_SIZE,
        OFFSET_Y + 9.0 * SQUARE_SIZE,
        2.0,
        BLACK,
    );

    // Draw "River" text (optional, simple decorative text)
    draw_text(
        "楚 河             漢 界",
        OFFSET_X + 1.5 * SQUARE_SIZE,
        OFFSET_Y + 4.6 * SQUARE_SIZE,
        30.0,
        BLACK,
    );
}

fn display_row(r: usize, human_color: PieceColor) -> usize {
    if human_color == PieceColor::Red {
        r
    } else {
        9 - r
    }
}
fn display_col(c: usize, human_color: PieceColor) -> usize {
    if human_color == PieceColor::Red {
        c
    } else {
        8 - c
    }
}

fn draw_pieces(
    board: &Board,
    selected_sq: Option<usize>,
    legal_moves: &[Move],
    human_color: PieceColor,
) {
    for row in 0..10 {
        for col in 0..9 {
            let sq = Board::coord_to_square(row, col);
            let d_row = display_row(row, human_color);
            let d_col = display_col(col, human_color);
            let x = OFFSET_X + d_col as f32 * SQUARE_SIZE;
            let y = OFFSET_Y + d_row as f32 * SQUARE_SIZE;

            // Highlight selected square
            if Some(sq) == selected_sq {
                draw_circle(x, y, RADIUS + 5.0, YELLOW);
            }

            // Highlight valid move targets
            if selected_sq.is_some() && legal_moves.iter().any(|m| m.to_sq() == sq) {
                draw_circle(x, y, 10.0, GREEN);
            }

            if let Some(piece) = board.piece_at(sq) {
                draw_circle(
                    x,
                    y,
                    RADIUS,
                    if piece.color == PieceColor::Red {
                        RED
                    } else {
                        BLACK
                    },
                );
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
                draw_text(
                    text,
                    x - text_size.width / 2.0,
                    y + text_size.height / 2.0,
                    30.0,
                    WHITE,
                );
            }
        }
    }
}

fn draw_button(text: &str, x: f32, y: f32, w: f32, h: f32, active: bool, color: Color) -> bool {
    let bg_color = if active { GREEN } else { color };
    draw_rectangle(x, y, w, h, bg_color);
    draw_rectangle_lines(x, y, w, h, 2.0, BLACK);

    let text_size = measure_text(text, None, 20, 1.0);
    draw_text(
        text,
        x + (w - text_size.width) / 2.0,
        y + (h + text_size.height) / 2.0,
        20.0,
        WHITE,
    );

    if is_mouse_button_pressed(MouseButton::Left) {
        let (mx, my) = mouse_position();
        if mx >= x && mx <= x + w && my >= y && my <= y + h {
            return true;
        }
    }
    false
}

// Window conf
fn window_conf() -> Conf {
    Conf {
        window_title: "Wisdom Engine - Xiangqi".to_owned(),
        window_width: 900, // Widened for control panel
        window_height: 700,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut board = Board::new();
    board.set_initial_position();

    let mut tt = wisdom::tt::TranspositionTable::new(64);
    let mut game_history: Vec<HistoryEntry> = Vec::new();

    let mut selected_sq: Option<usize> = None;
    let mut legal_moves = Vec::new();
    let mut game_over = false;
    let mut game_over_message = String::from("GAME OVER");

    // UI State
    let mut game_mode = GameMode::EngineVsPlayer;
    let mut human_color = PieceColor::Red; // Red is bottom by default
    let mut search_depth: u8 = 4;
    let mut current_eval: Option<i32> = Some(board.evaluate());

    loop {
        draw_board();
        draw_pieces(&board, selected_sq, &legal_moves, human_color);

        // --- DRAW CONTROL PANEL ---
        let panel_x = 620.0;
        let mut py = OFFSET_Y;

        draw_text("Controls", panel_x, py, 30.0, BLACK);
        py += 40.0;

        // Reset Button
        if draw_button("Reset Game", panel_x, py, 200.0, 40.0, false, BLUE) {
            board.set_initial_position();
            tt = wisdom::tt::TranspositionTable::new(64);
            game_history.clear();
            selected_sq = None;
            legal_moves.clear();
            game_over = false;
            game_over_message = "GAME OVER".into();
            current_eval = Some(board.evaluate());
        }
        py += 60.0;

        // Game Mode Toggle
        draw_text("Game Mode:", panel_x, py, 20.0, BLACK);
        py += 25.0;
        if draw_button(
            "Engine vs Player",
            panel_x,
            py,
            180.0,
            30.0,
            game_mode == GameMode::EngineVsPlayer,
            GRAY,
        ) {
            game_mode = GameMode::EngineVsPlayer;
            board.set_initial_position();
            game_history.clear();
            selected_sq = None;
            legal_moves.clear();
            game_over = false;
            game_over_message = "GAME OVER".into();
            current_eval = Some(board.evaluate());
        }
        py += 40.0;
        if draw_button(
            "Engine vs Engine",
            panel_x,
            py,
            180.0,
            30.0,
            game_mode == GameMode::EngineVsEngine,
            GRAY,
        ) {
            game_mode = GameMode::EngineVsEngine;
            board.set_initial_position();
            game_history.clear();
            selected_sq = None;
            legal_moves.clear();
            game_over = false;
            game_over_message = "GAME OVER".into();
            current_eval = Some(board.evaluate());
        }
        py += 40.0;

        // Side Selection (Only Engine vs Player)
        if game_mode == GameMode::EngineVsPlayer {
            draw_text("Player Side:", panel_x, py, 20.0, BLACK);
            py += 25.0;
            if draw_button(
                "Play Red",
                panel_x,
                py,
                80.0,
                30.0,
                human_color == PieceColor::Red,
                GRAY,
            ) {
                human_color = PieceColor::Red;
                board.set_initial_position();
                game_history.clear();
                selected_sq = None;
                legal_moves.clear();
                game_over = false;
                game_over_message = "GAME OVER".into();
                current_eval = Some(board.evaluate());
            }
            if draw_button(
                "Play Black",
                panel_x + 90.0,
                py,
                90.0,
                30.0,
                human_color == PieceColor::Black,
                GRAY,
            ) {
                human_color = PieceColor::Black;
                board.set_initial_position();
                game_history.clear();
                selected_sq = None;
                legal_moves.clear();
                game_over = false;
                game_over_message = "GAME OVER".into();
                current_eval = Some(board.evaluate());
            }
            py += 40.0;
        }

        // Search Depth
        draw_text(
            &format!("Depth: {}", search_depth),
            panel_x,
            py,
            20.0,
            BLACK,
        );
        py += 25.0;
        if draw_button("-", panel_x, py, 40.0, 30.0, false, GRAY) {
            if search_depth > 1 {
                search_depth -= 1;
            }
        }
        if draw_button("+", panel_x + 50.0, py, 40.0, 30.0, false, GRAY) {
            if search_depth < 10 {
                search_depth += 1;
            }
        }
        py += 50.0;

        // Eval Display
        if game_mode == GameMode::EngineVsEngine
            || (game_mode == GameMode::EngineVsPlayer && board.side_to_move == human_color)
        {
            if let Some(eval) = current_eval {
                draw_text(
                    &format!("Eval: {}", eval),
                    panel_x,
                    py,
                    30.0,
                    match eval {
                        e if e > 500 => DARKGREEN,
                        e if e < -500 => MAROON,
                        _ => BLACK,
                    },
                );
            }
        }

        if game_over {
            draw_text(&game_over_message, OFFSET_X, OFFSET_Y / 2.0, 30.0, RED);
        }

        // --- GAME LOGIC ---
        if !game_over {
            let is_human_turn =
                game_mode == GameMode::EngineVsPlayer && board.side_to_move == human_color;

            if is_human_turn {
                // Human Turn handling
                if is_mouse_button_pressed(MouseButton::Left) {
                    let (mx, my) = mouse_position();
                    let d_col =
                        ((mx - OFFSET_X + SQUARE_SIZE / 2.0) / SQUARE_SIZE).floor() as isize;
                    let d_row =
                        ((my - OFFSET_Y + SQUARE_SIZE / 2.0) / SQUARE_SIZE).floor() as isize;

                    if d_col >= 0 && d_col < 9 && d_row >= 0 && d_row < 10 {
                        let c = if human_color == PieceColor::Red {
                            d_col as usize
                        } else {
                            8 - (d_col as usize)
                        };
                        let r = if human_color == PieceColor::Red {
                            d_row as usize
                        } else {
                            9 - (d_row as usize)
                        };
                        let sq = Board::coord_to_square(r, c);

                        if selected_sq.is_some() {
                            // Try to execute move
                            if let Some(&m) = legal_moves.iter().find(|m| m.to_sq() == sq) {
                                apply_move_to_game(&mut board, m, &mut game_history);
                                selected_sq = None;
                                legal_moves.clear();

                                if game_history.len() >= 4 {
                                    match board.judge_repetition(
                                        &game_history,
                                        game_history.len(),
                                        2,
                                    ) {
                                        RepetitionResult::Win => {
                                            game_over = true;
                                            game_over_message = "Rule Violation: You Lose!".into();
                                            current_eval = Some(-20000);
                                        }
                                        RepetitionResult::Loss => {
                                            game_over = true;
                                            game_over_message =
                                                "Opponent Violation: You Win!".into();
                                            current_eval = Some(20000);
                                        }
                                        RepetitionResult::Draw => {
                                            game_over = true;
                                            game_over_message = "Draw by Repetition!".into();
                                            current_eval = Some(0);
                                        }
                                        RepetitionResult::Undecided => {
                                            let all = get_legal_moves(&mut board);
                                            if all.is_empty() {
                                                game_over = true;
                                                game_over_message = "Checkmate!".into();
                                                current_eval = Some(-20000);
                                            } else {
                                                current_eval = Some(board.evaluate());
                                            }
                                        }
                                    }
                                } else {
                                    let all = get_legal_moves(&mut board);
                                    if all.is_empty() {
                                        game_over = true;
                                        game_over_message = "Checkmate!".into();
                                        current_eval = Some(-20000);
                                    } else {
                                        current_eval = Some(board.evaluate());
                                    }
                                }
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
                // Engine Turn handling
                next_frame().await; // render human move or previous frame

                let all_moves = get_legal_moves(&mut board);
                if all_moves.is_empty() {
                    game_over = true;
                    game_over_message = "Checkmate!".into();
                    current_eval = Some(-20000);
                } else {
                    let (best_move, _) = alphabeta_search_best_move(
                        &mut board,
                        search_depth,
                        &tt,
                        &game_history,
                        None,
                    );
                    apply_move_to_game(&mut board, best_move, &mut game_history);

                    if game_history.len() >= 4 {
                        match board.judge_repetition(&game_history, game_history.len(), 2) {
                            RepetitionResult::Win => {
                                game_over = true;
                                game_over_message = "Engine Violation: You Win!".into();
                                current_eval = Some(20000);
                            }
                            RepetitionResult::Loss => {
                                game_over = true;
                                game_over_message = "Rule Violation: Engine Wins!".into();
                                current_eval = Some(-20000);
                            }
                            RepetitionResult::Draw => {
                                game_over = true;
                                game_over_message = "Draw by Repetition!".into();
                                current_eval = Some(0);
                            }
                            RepetitionResult::Undecided => {
                                if get_legal_moves(&mut board).is_empty() {
                                    game_over = true;
                                    game_over_message = "Checkmate!".into();
                                    current_eval = Some(-20000);
                                } else {
                                    current_eval = Some(board.evaluate());
                                }
                            }
                        }
                    } else {
                        if get_legal_moves(&mut board).is_empty() {
                            game_over = true;
                            game_over_message = "Checkmate!".into();
                            current_eval = Some(-20000);
                        } else {
                            current_eval = Some(board.evaluate());
                        }
                    }
                }
            }
        }

        next_frame().await
    }
}
