use burn::backend::{NdArray, ndarray::NdArrayDevice};
use macroquad::prelude::*;
use wisdom::board::{Board, Color as PieceColor, HistoryEntry, PieceType, RepetitionResult};
use wisdom::eval_queue::EvalQueue;
use wisdom::mcts::MCTS;
use wisdom::r#move::Move;
use wisdom::nn::XiangqiNetConfig;
use wisdom::tt::TranspositionTable;

const SQUARE_SIZE: f32 = 60.0;
const OFFSET_X: f32 = 50.0;
const OFFSET_Y: f32 = 50.0;
const RADIUS: f32 = 25.0;

const MODEL_PATH: &str = "wisdom_model";

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

    let pre_threats = if is_reversible {
        board.get_unprotected_threats(moving_side)
    } else {
        0
    };

    board.make_move(m);

    let gives_check = board.is_in_check(board.side_to_move);

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

fn format_move(m: Move) -> String {
    let (from_row, from_col) = Board::square_to_coord(m.from_sq());
    let (to_row, to_col) = Board::square_to_coord(m.to_sq());

    // Convert to UCCI format. Files a-i, Ranks 0-9 (from bottom to top).
    // Our row 9 is bottom, so UCCI rank is 9 - row.
    let from_file = (b'a' + from_col as u8) as char;
    let from_rank = (b'0' + (9 - from_row) as u8) as char;
    let to_file = (b'a' + to_col as u8) as char;
    let to_rank = (b'0' + (9 - to_row) as u8) as char;

    format!("{}{}{}{}", from_file, from_rank, to_file, to_rank)
}

fn draw_board() {
    clear_background(Color::new(0.9, 0.8, 0.6, 1.0));

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

            if Some(sq) == selected_sq {
                draw_circle(x, y, RADIUS + 5.0, YELLOW);
            }

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

fn window_conf() -> Conf {
    Conf {
        window_title: "Wisdom Engine - Xiangqi (MCTS)".to_owned(),
        window_width: 900,
        window_height: 700,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // Chỉ ghi tên, KHÔNG GHI ĐUÔI
    const MODEL_PATH: &str = "wisdom_model"; 

    // --- Initialize NN + MCTS Engine ---
    type B = NdArray<f32>;
    let device = NdArrayDevice::Cpu;
    let config = XiangqiNetConfig::new();

    // Check file để in ra UI
    let file_to_check = format!("{}.pt", MODEL_PATH);
    let model = if std::path::Path::new(&file_to_check).exists() {
        println!("📦 GUI: Phát hiện file {}, đang nạp...", file_to_check);
        config.load_model::<B>(MODEL_PATH, &device)
    } else {
        println!("⚠️ GUI: Không tìm thấy {}. Dùng model ngẫu nhiên.", file_to_check);
        config.init::<B>(&device)
    };

    let eval_queue = EvalQueue::new(model, device, 32, 5);
    let eval_tx = eval_queue.tx.clone();
    let tt = TranspositionTable::new(64);
    let mcts = MCTS::new(200_000);

    // --- Game State ---
    let mut board = Board::new();
    board.set_initial_position();

    let mut game_history: Vec<HistoryEntry> = Vec::new();

    let mut selected_sq: Option<usize> = None;
    let mut legal_moves = Vec::new();
    let mut game_over = false;
    let mut game_over_message = String::from("GAME OVER");

    // UI State
    let mut game_mode = GameMode::EngineVsPlayer;
    let mut human_color = PieceColor::Red;
    let mut mcts_simulations: usize = 800;
    let mut current_eval: Option<String> = Some("Ready".to_string());
    let mut engine_policy: Vec<String> = Vec::new();

    loop {
        draw_board();
        draw_pieces(&board, selected_sq, &legal_moves, human_color);

        // --- DRAW CONTROL PANEL ---
        let panel_x = 620.0;
        let mut py = OFFSET_Y;

        draw_text("MCTS Engine", panel_x, py, 30.0, BLACK);
        py += 40.0;

        // Reset Button
        if draw_button("Reset Game", panel_x, py, 200.0, 40.0, false, BLUE) {
            board.set_initial_position();
            game_history.clear();
            selected_sq = None;
            legal_moves.clear();
            game_over = false;
            game_over_message = "GAME OVER".into();
            current_eval = Some("Ready".to_string());
            engine_policy.clear();
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
            current_eval = Some("Ready".to_string());
            engine_policy.clear();
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
            current_eval = Some("Ready".to_string());
            engine_policy.clear();
        }
        py += 40.0;

        // Side Selection (Only for Engine vs Player)
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
                current_eval = Some("Ready".to_string());
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
                current_eval = Some("Ready".to_string());
            }
            py += 40.0;
        }

        // MCTS Simulations Control
        draw_text(
            &format!("Simulations: {}", mcts_simulations),
            panel_x,
            py,
            20.0,
            BLACK,
        );
        py += 25.0;
        if draw_button("-", panel_x, py, 40.0, 30.0, false, GRAY) {
            if mcts_simulations > 100 {
                mcts_simulations -= 100;
            }
        }
        if draw_button("+", panel_x + 50.0, py, 40.0, 30.0, false, GRAY) {
            if mcts_simulations < 5000 {
                mcts_simulations += 100;
            }
        }
        py += 50.0;

        // Eval Display (MCTS style)
        if let Some(ref eval_str) = current_eval {
            draw_text(eval_str, panel_x, py, 22.0, DARKGREEN);
            py += 30.0;
        }

        // Draw Policy
        if !engine_policy.is_empty() {
            draw_text("Top Moves:", panel_x, py, 20.0, BLACK);
            py += 25.0;
            for line in &engine_policy {
                draw_text(line, panel_x, py, 18.0, DARKGRAY);
                py += 20.0;
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
                                            current_eval = Some("LOST".to_string());
                                        }
                                        RepetitionResult::Loss => {
                                            game_over = true;
                                            game_over_message =
                                                "Opponent Violation: You Win!".into();
                                            current_eval = Some("WON".to_string());
                                        }
                                        RepetitionResult::Draw => {
                                            game_over = true;
                                            game_over_message = "Draw by Repetition!".into();
                                            current_eval = Some("DRAW".to_string());
                                        }
                                        RepetitionResult::Undecided => {
                                            let all = get_legal_moves(&mut board);
                                            if all.is_empty() {
                                                game_over = true;
                                                game_over_message = "Checkmate!".into();
                                                current_eval = Some("CHECKMATE".to_string());
                                            } else {
                                                current_eval = Some("Your turn".to_string());
                                            }
                                        }
                                    }
                                } else {
                                    let all = get_legal_moves(&mut board);
                                    if all.is_empty() {
                                        game_over = true;
                                        game_over_message = "Checkmate!".into();
                                        current_eval = Some("CHECKMATE".to_string());
                                    } else {
                                        current_eval = Some("Your turn".to_string());
                                    }
                                }
                            } else {
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
                // ========================================
                // ENGINE TURN: Use MCTS!
                // ========================================
                next_frame().await; // Render the board before AI thinks

                let all_moves = get_legal_moves(&mut board);
                if all_moves.is_empty() {
                    game_over = true;
                    game_over_message = "Checkmate!".into();
                    current_eval = Some("CHECKMATE".to_string());
                } else {
                    current_eval = Some(format!("Thinking... ({} sims)", mcts_simulations));

                    let start = std::time::Instant::now();
                    let best_move = mcts.search_best_move(
                        &board,
                        mcts_simulations,
                        &eval_tx,
                        &tt,
                        4, // num_threads
                    );
                    let elapsed = start.elapsed();

                    // Gather MCTS stats for display
                    let root_start = mcts.tree[0]
                        .children_index
                        .load(std::sync::atomic::Ordering::Acquire);
                    let root_children = mcts.tree[0]
                        .num_children
                        .load(std::sync::atomic::Ordering::Acquire);
                    let root_visits = mcts.tree[0]
                        .visits
                        .load(std::sync::atomic::Ordering::Acquire);

                    let mut best_child_visits = 0u32;
                    let mut best_child_q = 0.0f32;
                    let mut children_stats: Vec<(Move, u32)> =
                        Vec::with_capacity(root_children as usize);

                    for i in 0..root_children {
                        let idx = root_start as usize + i as usize;
                        let node = &mcts.tree[idx];
                        let nv = node.visits.load(std::sync::atomic::Ordering::Acquire);

                        children_stats.push((Move(node.get_move()), nv));

                        if nv > best_child_visits {
                            best_child_visits = nv;
                            if nv > 0 {
                                best_child_q = -node.get_value() / nv as f32;
                            }
                        }
                    }

                    children_stats.sort_by(|a, b| b.1.cmp(&a.1));
                    engine_policy.clear();
                    let total_visits = std::cmp::max(1, root_visits) as f32;
                    for (i, &(mv, nv)) in children_stats.iter().take(5).enumerate() {
                        if nv > 0 {
                            let pct = (nv as f32 / total_visits) * 100.0;
                            engine_policy.push(format!(
                                "{}. {} ({:.1}%)",
                                i + 1,
                                format_move(mv),
                                pct
                            ));
                        }
                    }

                    // Display: Win% relative to the engine's side
                    let win_pct = ((best_child_q + 1.0) / 2.0 * 100.0).clamp(0.0, 100.0);
                    current_eval = Some(format!(
                        "V:{} N:{} W:{:.0}% {:.1}s",
                        root_visits,
                        best_child_visits,
                        win_pct,
                        elapsed.as_secs_f32()
                    ));

                    apply_move_to_game(&mut board, best_move, &mut game_history);

                    if game_history.len() >= 4 {
                        match board.judge_repetition(&game_history, game_history.len(), 2) {
                            RepetitionResult::Win => {
                                game_over = true;
                                game_over_message = "Engine Violation: You Win!".into();
                                current_eval = Some("WON".to_string());
                            }
                            RepetitionResult::Loss => {
                                game_over = true;
                                game_over_message = "Rule Violation: Engine Wins!".into();
                                current_eval = Some("LOST".to_string());
                            }
                            RepetitionResult::Draw => {
                                game_over = true;
                                game_over_message = "Draw by Repetition!".into();
                                current_eval = Some("DRAW".to_string());
                            }
                            RepetitionResult::Undecided => {
                                if get_legal_moves(&mut board).is_empty() {
                                    game_over = true;
                                    game_over_message = "Checkmate!".into();
                                    current_eval = Some("CHECKMATE".to_string());
                                }
                            }
                        }
                    } else {
                        if get_legal_moves(&mut board).is_empty() {
                            game_over = true;
                            game_over_message = "Checkmate!".into();
                            current_eval = Some("CHECKMATE".to_string());
                        }
                    }
                }
            }
        }

        next_frame().await
    }
}
