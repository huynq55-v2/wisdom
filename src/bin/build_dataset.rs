use std::fs::{self, File};
use std::io::{BufWriter, Write};
use walkdir::WalkDir;
use wisdom::board::{Board, Color};
use wisdom::r#move::Move;
use wisdom::nn::move_to_index;

fn main() {
    let input_dir = "./dhtmlxq_data";
    let output_file = "./xiangqi_dataset.csv";

    let out_file = File::create(output_file).expect("Không thể tạo file đầu ra");
    let mut writer = BufWriter::new(out_file);

    writeln!(writer, "fen,policy,value").unwrap();

    let mut success_count = 0;
    let mut fail_count = 0;

    println!("Đang quét thư mục và phân tích kỳ phổ...");

    for entry in WalkDir::new(input_dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            if let Ok(bytes) = fs::read(path) {
                let content = String::from_utf8_lossy(&bytes);
                if content.contains("[DhtmlXQ]") {
                    let file_name = path.file_name().unwrap_or_default().to_string_lossy();

                    if parse_and_save(&content, &mut writer) {
                        success_count += 1;
                        println!("✅ [{} OK] - Đã xử lý: {}", success_count, file_name);
                    } else {
                        fail_count += 1;
                        println!(
                            "❌ [BỎ QUA] - Lỗi định dạng, phi lý hoặc sai kết quả: {}",
                            file_name
                        );
                    }
                }
            }
        }
    }

    println!("========================================");
    println!("Hoàn thành!");
    println!("Trích xuất thành công: {} ván", success_count);
    println!("Bỏ qua/Lỗi: {} ván", fail_count);
    println!("Dữ liệu đã được lưu tại: {}", output_file);
}

/// Hàm phụ trợ sinh các nước đi hợp lệ tuyệt đối theo luật
fn get_legal_moves(board: &mut Board) -> Vec<Move> {
    let mut pseudo_moves = board.generate_captures();
    pseudo_moves.append(&mut board.generate_quiets());
    let mut legal_moves = Vec::new();
    let current_side = board.side_to_move;

    for &m in &pseudo_moves {
        let undo = board.make_move(m);
        // Nước đi hợp lệ: Tướng không được đối mặt và không bị chiếu
        if !board.kings_facing() && !board.is_in_check(current_side) {
            legal_moves.push(m);
        }
        board.unmake_move(m, undo);
    }
    legal_moves
}

fn parse_and_save(content: &str, writer: &mut BufWriter<File>) -> bool {
    let binit = extract_tag(content, "DhtmlXQ_binit");
    let standard_binit = "0919293949596979891777062646668600102030405060708012720323436383";
    if !binit.is_empty() && binit != standard_binit {
        return false;
    }

    let result_str = extract_tag(content, "DhtmlXQ_result");
    let red_score = if result_str.contains("红胜")
        || result_str.contains("黑负")
        || result_str.contains("红先胜")
    {
        1.0
    } else if result_str.contains("黑胜")
        || result_str.contains("红负")
        || result_str.contains("红先负")
    {
        -1.0
    } else if result_str.contains("和局") || result_str.contains("和") {
        0.0
    } else {
        return false;
    };

    let movelist = extract_tag(content, "DhtmlXQ_movelist");
    let chars: Vec<char> = movelist.chars().filter(|c| c.is_ascii_digit()).collect();
    if chars.is_empty() || chars.len() % 4 != 0 {
        return false;
    }

    // FIX 3: Lọc chặt chẽ hơn. Dưới 30 half-moves (15 hiệp) đa phần là rớt mạng.
    if chars.len() / 4 < 30 {
        return false;
    }

    let mut board = Board::new();
    board.set_initial_position();
    let mut temp_lines = Vec::with_capacity(chars.len() / 4);

    for chunk in chars.chunks(4) {
        let x1 = chunk[0].to_digit(10).unwrap() as usize;
        let y1 = chunk[1].to_digit(10).unwrap() as usize;
        let x2 = chunk[2].to_digit(10).unwrap() as usize;
        let y2 = chunk[3].to_digit(10).unwrap() as usize;

        // FIX 1: Bảo vệ an toàn bộ nhớ (Chống Panic)
        if x1 > 8 || y1 > 9 || x2 > 8 || y2 > 9 {
            return false; // Vứt bỏ kỳ phổ chứa tọa độ ngoài bàn cờ
        }

        let from_sq = Board::coord_to_square(y1, x1);
        let to_sq = Board::coord_to_square(y2, x2);

        let piece = match board.piece_at(from_sq) {
            Some(p) => p,
            None => return false,
        };

        if piece.color != board.side_to_move {
            return false;
        }

        let is_capture = !board.is_empty(to_sq);
        let m = Move::new(from_sq, to_sq, is_capture);

        // FIX 2: Tối ưu hiệu năng siêu tốc
        let mut pseudo_moves = board.generate_captures();
        pseudo_moves.append(&mut board.generate_quiets());

        if !pseudo_moves.contains(&m) {
            return false; // Nước đi sai luật di chuyển (Ví dụ Tượng qua sông)
        }

        // Lưu dữ liệu TRƯỚC KHI đánh nước m
        let fen = board.to_fen();
        let policy_index = move_to_index(m);
        let current_value = if board.side_to_move == Color::Red {
            red_score
        } else {
            -red_score
        };
        temp_lines.push(format!("{},{},{}", fen, policy_index, current_value));

        let current_side = board.side_to_move;

        // Đi thử trên bàn cờ thật
        board.make_move(m);

        // Xác minh xem nước đi vừa rồi có khiến Vua tự sát không
        if board.kings_facing() || board.is_in_check(current_side) {
            // Không thèm unmake_move vì đằng nào cũng vứt luôn ván này
            return false;
        }
    }

    // KIỂM TOÁN END GAME (Giữ nguyên logic cũ của bạn vì nó rất chuẩn)
    let final_legal_moves = get_legal_moves(&mut board);
    if final_legal_moves.is_empty() {
        let actual_red_score = if board.side_to_move == Color::Red {
            -1.0
        } else {
            1.0
        };
        if actual_red_score != red_score {
            return false;
        }
    }

    for line in temp_lines {
        if let Err(_) = writeln!(writer, "{}", line) {
            return false;
        }
    }

    true
}

fn extract_tag<'a>(content: &'a str, tag: &str) -> &'a str {
    let start_tag = format!("[{}]", tag);
    let end_tag = format!("[/{}]", tag);

    if let Some(start_idx) = content.find(&start_tag) {
        let value_start = start_idx + start_tag.len();
        if let Some(end_idx) = content[value_start..].find(&end_tag) {
            return &content[value_start..value_start + end_idx];
        }
    }
    ""
}
