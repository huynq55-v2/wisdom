use wisdom::board::{Board, Color, HistoryEntry, PieceType, RepetitionResult};
use wisdom::ucci::parse_fen;

/// Hàm dịch ký hiệu đại số (VD: R4+1, K5=4) sang nước đi (Move) hợp lệ của Engine
fn parse_algebraic(board: &Board, alg: &str) -> wisdom::r#move::Move {
    let chars: Vec<char> = alg.chars().collect();
    let is_front_back = chars[0] == '+' || chars[0] == '-';

    let (modifier, p_char, start_file_digit, op, arg_digit) = if is_front_back {
        // Định dạng Tiền/Hậu: VD "+R+2"
        let p = chars[1];
        let op = chars[2];
        let arg = chars[3].to_digit(10).unwrap() as u8;
        (Some(chars[0]), p, None, op, arg)
    } else {
        // Định dạng thông thường: VD "R2+1"
        let p = chars[0];
        let file = chars[1].to_digit(10).unwrap() as u8;
        let op = chars[2];
        let arg = chars[3].to_digit(10).unwrap() as u8;
        (None, p, Some(file), op, arg)
    };

    let moving_side = board.side_to_move;

    let expected_piece_type = match p_char {
        'K' => PieceType::King,
        'A' => PieceType::Advisor,
        'B' => PieceType::Elephant,
        'N' => PieceType::Horse,
        'R' => PieceType::Rook,
        'C' => PieceType::Cannon,
        'P' => PieceType::Pawn,
        _ => panic!("Unknown piece type: {}", p_char),
    };

    let mut expected_from_col_index: Option<u8> = None;
    let mut expected_from_row_index: Option<usize> = None;

    if let Some(file_digit) = start_file_digit {
        // Phân tích cột xuất phát theo định dạng cũ
        let col = if moving_side == Color::Red {
            9 - file_digit
        } else {
            file_digit - 1
        };
        expected_from_col_index = Some(col);
    } else if let Some(mod_char) = modifier {
        // Quét bàn cờ tìm cột có 2 quân cùng loại
        let mut col_pieces: [Vec<usize>; 9] = Default::default();
        for sq in 0..256 {
            if !Board::is_valid_square(sq) {
                continue;
            }
            if let Some(piece) = board.piece_at(sq) {
                if piece.piece_type == expected_piece_type && piece.color == moving_side {
                    let (r, c) = Board::square_to_coord(sq);
                    col_pieces[c].push(r);
                }
            }
        }

        // Xác định hàng (row) của quân Tiền/Hậu
        for c in 0..9 {
            if col_pieces[c].len() >= 2 {
                col_pieces[c].sort(); // Sắp xếp hàng từ 0 -> 9 (Từ trên xuống dưới)
                expected_from_col_index = Some(c as u8);

                // Đỏ ở dưới (hàng 9) tiến lên trên (hàng 0) -> Quân Tiền có hàng NHỎ hơn
                // Đen ở trên (hàng 0) tiến xuống dưới (hàng 9) -> Quân Tiền có hàng LỚN hơn
                let (front_row, back_row) = if moving_side == Color::Red {
                    (col_pieces[c][0], col_pieces[c][col_pieces[c].len() - 1])
                } else {
                    (col_pieces[c][col_pieces[c].len() - 1], col_pieces[c][0])
                };

                if mod_char == '+' {
                    expected_from_row_index = Some(front_row); // Tiền (+)
                } else if mod_char == '-' {
                    expected_from_row_index = Some(back_row); // Hậu (-)
                }
                break;
            }
        }
    }

    let mut moves = board.generate_captures();
    moves.append(&mut board.generate_quiets());

    for m in moves {
        let from_sq = m.from_sq();
        let to_sq = m.to_sq();

        let (from_row, from_col) = Board::square_to_coord(from_sq);
        let (to_row, to_col) = Board::square_to_coord(to_sq);

        // Kiểm tra cột xuất phát
        if let Some(exp_c) = expected_from_col_index {
            if from_col as u8 != exp_c {
                continue;
            }
        }

        // Kiểm tra hàng xuất phát (Chỉ dùng cho Tiền/Hậu)
        if let Some(exp_r) = expected_from_row_index {
            if from_row != exp_r {
                continue;
            }
        }

        let piece = board.piece_at(from_sq).unwrap();
        if piece.piece_type != expected_piece_type {
            continue;
        }
        if piece.color != moving_side {
            continue;
        }

        let is_straight = matches!(
            piece.piece_type,
            PieceType::King | PieceType::Rook | PieceType::Cannon | PieceType::Pawn
        );

        match op {
            '=' => {
                // Bình ngang
                let expected_to_col = if moving_side == Color::Red {
                    9 - arg_digit
                } else {
                    arg_digit - 1
                };
                if to_col as u8 == expected_to_col && to_row == from_row {
                    return m;
                }
            }
            '+' => {
                // Tiến
                if is_straight {
                    if from_col != to_col {
                        continue;
                    }
                    let expected_to_row = if moving_side == Color::Red {
                        from_row as i32 - arg_digit as i32
                    } else {
                        from_row as i32 + arg_digit as i32
                    };
                    if to_row as i32 == expected_to_row {
                        return m;
                    }
                } else {
                    let expected_to_col = if moving_side == Color::Red {
                        9 - arg_digit
                    } else {
                        arg_digit - 1
                    };
                    let is_forward = if moving_side == Color::Red {
                        to_row < from_row
                    } else {
                        to_row > from_row
                    };
                    if is_forward && to_col as u8 == expected_to_col {
                        return m;
                    }
                }
            }
            '-' => {
                // Thoái
                if is_straight {
                    if from_col != to_col {
                        continue;
                    }
                    let expected_to_row = if moving_side == Color::Red {
                        from_row as i32 + arg_digit as i32
                    } else {
                        from_row as i32 - arg_digit as i32
                    };
                    if to_row as i32 == expected_to_row {
                        return m;
                    }
                } else {
                    let expected_to_col = if moving_side == Color::Red {
                        9 - arg_digit
                    } else {
                        arg_digit - 1
                    };
                    let is_backward = if moving_side == Color::Red {
                        to_row > from_row
                    } else {
                        to_row < from_row
                    };
                    if is_backward && to_col as u8 == expected_to_col {
                        return m;
                    }
                }
            }
            _ => {}
        }
    }
    panic!("Move not found or illegal: {}", alg);
}

fn run_repetition_scenario(fen: &str, moves: &[&str]) -> RepetitionResult {
    let mut board = Board::new();
    parse_fen(&mut board, fen);

    let mut history: Vec<HistoryEntry> = Vec::new();

    for alg_move in moves {
        let m = parse_algebraic(&board, alg_move);

        let piece = board.piece_at(m.from_sq()).unwrap();
        let is_reversible = piece.piece_type != PieceType::Pawn || {
            let (from_row, _) = Board::square_to_coord(m.from_sq());
            let (to_row, _) = Board::square_to_coord(m.to_sq());
            from_row == to_row
        };
        let moving_side = board.side_to_move;

        let pre_threats = if is_reversible {
            board.get_unprotected_threats(moving_side)
        } else {
            0
        };

        board.make_move(m);

        let gives_check = board.is_in_check(moving_side.opposite());

        let chased_set = if is_reversible && !gives_check {
            let post_threats = board.get_unprotected_threats(moving_side);
            post_threats & !pre_threats
        } else {
            0
        };

        println!(
            "Move: {:<5} | Side: {:?} | Check: {:<5} | ChasedSet: {:b}",
            alg_move, moving_side, gives_check, chased_set
        );

        history.push(HistoryEntry {
            hash: board.zobrist_key,
            is_check: gives_check,
            chased_set,
            is_reversible,
        });
    }

    board.judge_repetition(&history, history.len(), 1)
}

#[test]
fn test_diagram_1() {
    let fen = "c2a1k3/2c1a2R1/e6P1/8p/9/9/9/9/9/4K4 w - - 0 1";
    // 3 Chu kỳ lặp lại cho Perpetual Check (để đảm bảo đủ history)
    let moves = [
        "R2+1", "K6+1", "R2-1", "K6-1", "R2+1", "K6+1", "R2-1", "K6-1", "R2+1", "K6+1", "R2-1",
    ];
    assert_eq!(run_repetition_scenario(fen, &moves), RepetitionResult::Win);
}

#[test]
fn test_diagram_3() {
    let fen = "3kR4/5R3/9/2e6/9/n8/3pE4/4p4/9/4K4 w - - 0 1";
    let moves = [
        // --- Chu kỳ 1 ---
        "R4+1", "K4+1", "R5=6", "K4=5", "R6=5", "K5=4", "R4-1", "K4-1",
        // --- Chu kỳ 2 (Lặp lại để tạo Repetition) ---
        "R4+1", "K4+1", "R5=6", "K4=5", "R6=5", "K5=4", "R4-1",
    ];
    // Đỏ bị xử thua (Loss) vì lỗi Chiếu luân phiên (Perpetual Check). Đen Win
    assert_eq!(run_repetition_scenario(fen, &moves), RepetitionResult::Win);
}

#[test]
fn test_diagram_4_mutual_perpetual_check() {
    // Diagram 4: Mutual Perpetual Check (Cả 2 bên cùng Chiếu vĩnh viễn)
    // - Đỏ dùng Xe và Pháo luân phiên chiếu.
    // - Đen dùng Xe (nhờ Pháo mở đường) luân phiên chiếu.
    // Kết quả phải được xử Hòa (Draw) theo luật WXF.
    let fen = "3akr3/5c3/1P2e4/4R4/9/9/9/9/9/4CK3 w - - 0 1";

    let moves = [
        "R5+1",
        "C6=5", // 1. Khởi tạo vòng lặp: Xe Đỏ ăn Tượng chiếu, Pháo Đen cản (mở đường Xe Đen chiếu)
        "R5=4",
        "C5=6", // 2. Xe Đỏ cản (mở đường Pháo Đỏ chiếu), Pháo Đen chạy ra (hết chiếu)
        // --- Lặp lại chu kỳ để đưa vào History ---
        "R4=5", "C6=5", // 3. R4=5 (Xe Đỏ chiếu)  | C6=5 (Xe Đen chiếu)
        "R5=4", "C5=6", // 4. R5=4 (Pháo Đỏ chiếu)| C5=6 (Pháo Đen chặn)
        "R4=5", "C6=5", // 5.
        "R5=4",
    ];

    // Khi gọi judge_repetition, cả our_violation và opp_violation đều sẽ là PerpetualCheck (2).
    // 2 == 2 -> Trả về Draw.
    assert_eq!(run_repetition_scenario(fen, &moves), RepetitionResult::Draw);
}

#[test]
fn test_diagram_5() {
    let fen = "5c3/8R/4k2P1/3P5/5P3/6P2/5pH2/5A3/4CK3/6rh1 w - - 0 1";

    let moves = [
        "P4=5", "P6=5", "P5=4", "P5=6", "P4=5", "P6=5", "P5=4", "P5=6", "P4=5", "P6=5", "P5=4",
    ];

    assert_eq!(run_repetition_scenario(fen, &moves), RepetitionResult::Win);
}

#[test]
fn test_diagram_6() {
    let fen = "3ak1e2/4a4/4e4/r3C2R1/p8/9/P8/4p4/4p4/5K3 w - - 0 1";

    let moves = ["R2=3", "E7+9", "R3=2", "E9-7", "R2=3"];

    assert_eq!(run_repetition_scenario(fen, &moves), RepetitionResult::Draw);
}

#[test]
fn test_diagram_7() {
    let fen = "2e2k3/9/4e2P1/3H5/1R7p/6P2/5p2r/8h/4p4/5K3 w - - 0 1";

    let moves = ["H6+7", "K6+1", "H7-6", "K6-1", "H6+7", "K6+1", "H7-6"];

    assert_eq!(run_repetition_scenario(fen, &moves), RepetitionResult::Draw);
}

#[test]
fn test_diagram_8() {
    let fen = "3aka3/4h4/4e4/4C4/p3p3p/9/2c7/2C7/4K4/9 w - - 0 1";

    let moves = [
        "C7=3", "C3=7", "C3=7", "C7=3", "C7=3", "C3=7", "C3=7", "C7=3", "C7=3",
    ];

    assert_eq!(run_repetition_scenario(fen, &moves), RepetitionResult::Draw);
}

#[test]
fn test_diagram_9() {
    let fen = "2eHka3/1H2aPP2/h3e4/p8/5c3/2C6/9/4E4/4A4/r1EK1A1h1 w - - 0 1";

    let moves = ["C7=5", "C6=5", "C5=2", "C5=8", "C2=5", "C8=5", "C5=2"];

    assert_eq!(run_repetition_scenario(fen, &moves), RepetitionResult::Draw);
}

#[test]
fn test_diagram_18() {
    let fen = "2eakae2/3r5/2h1c2c1/p1R1p1p2/7hp/2P2CP2/P2rP3P/2H1C1H2/4A4/2E1KAE1R w - - 0 1";

    let moves = [
        "C4-1", "+R+2", "C4-2", "+R-2", "C4+2", "+R+2", "C4-2", "+R-2", "C4+2",
    ];

    assert_eq!(run_repetition_scenario(fen, &moves), RepetitionResult::Win);
}
