use std::fs::{self, File};
use std::io::{BufWriter, Write};
use walkdir::WalkDir;
use wisdom::board::{Board, Color};
use wisdom::nn::move_to_index;
use wisdom::r#move::Move;

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
                    // Lấy tên file để in ra màn hình
                    let file_name = path.file_name().unwrap_or_default().to_string_lossy();

                    if parse_and_save(&content, &mut writer) {
                        success_count += 1;
                        println!("✅ [{} OK] - Đã xử lý: {}", success_count, file_name);
                    } else {
                        fail_count += 1;
                        println!("❌ [BỎ QUA] - Lỗi định dạng hoặc kết quả: {}", file_name);
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

fn parse_and_save(content: &str, writer: &mut BufWriter<File>) -> bool {
    // 1. KIỂM TRA XEM CÓ PHẢI VÁN CỜ TIÊU CHUẨN KHÔNG
    let binit = extract_tag(content, "DhtmlXQ_binit");
    let standard_binit = "0919293949596979891777062646668600102030405060708012720323436383";
    if !binit.is_empty() && binit != standard_binit {
        return false; 
    }

    // 2. TRÍCH XUẤT KẾT QUẢ VÁN CỜ
    let result_str = extract_tag(content, "DhtmlXQ_result");
    let red_score = if result_str.contains("红胜") || result_str.contains("黑负") || result_str.contains("红先胜") {
        1.0
    } else if result_str.contains("黑胜") || result_str.contains("红负") || result_str.contains("红先负") {
        -1.0
    } else if result_str.contains("和局") || result_str.contains("和") {
        0.0
    } else {
        return false;
    };

    // 3. TRÍCH XUẤT MOVELIST (FIX LỖI KÝ TỰ XUỐNG DÒNG)
    let movelist = extract_tag(content, "DhtmlXQ_movelist");
    // Lọc loại bỏ mọi dấu cách, \r, \n. Chỉ giữ lại đúng số [0-9]
    let chars: Vec<char> = movelist.chars().filter(|c| c.is_ascii_digit()).collect();
    if chars.is_empty() || chars.len() % 4 != 0 {
        return false;
    }

    let mut board = Board::new();
    board.set_initial_position();

    // FIX LỖI RÒ RỈ DATA: Tạo mảng tạm để chứa các dòng FEN
    let mut temp_lines = Vec::with_capacity(chars.len() / 4);

    for chunk in chars.chunks(4) {
        let x1 = chunk[0].to_digit(10).unwrap() as usize;
        let y1 = chunk[1].to_digit(10).unwrap() as usize;
        let x2 = chunk[2].to_digit(10).unwrap() as usize;
        let y2 = chunk[3].to_digit(10).unwrap() as usize;

        let from_sq = Board::coord_to_square(y1, x1);
        let to_sq = Board::coord_to_square(y2, x2);

        // Kiểm tra xem ô xuất phát có quân cờ không
        let piece = match board.piece_at(from_sq) {
            Some(p) => p,
            None => return false, // Lỗi: Không có quân cờ ở tọa độ này
        };

        // FIX LỖI SAI LƯỢT: Quân cờ phải thuộc về phe đang đến lượt
        if piece.color != board.side_to_move {
            return false; // Lỗi: File kỳ phổ bị mất/nhảy nước đi
        }

        let is_capture = !board.is_empty(to_sq);
        let m = Move::new(from_sq, to_sq, is_capture);

        // Tính Policy Index và gán FEN
        let fen = board.to_fen();
        let policy_index = move_to_index(m);
        let current_value = if board.side_to_move == Color::Red { red_score } else { -red_score };

        // Ghi tạm vào RAM thay vì ghi trực tiếp ra File
        temp_lines.push(format!("{},{},{}", fen, policy_index, current_value));

        board.make_move(m);
    }

    // 4. LƯU VÀO FILE (Chỉ lưu khi TOÀN BỘ ván cờ đã được parse thành công)
    for line in temp_lines {
        if let Err(_) = writeln!(writer, "{}", line) {
            return false;
        }
    }

    true
}

// Hàm phụ trợ để tìm và cắt nội dung giữa các Tag
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