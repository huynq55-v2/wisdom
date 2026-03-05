use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};

/// Lấy giá trị index của move theo Action Space của Neural Network (0..8099)
fn get_move_index(best_move: &str) -> usize {
    let chars: Vec<char> = best_move.chars().collect();

    // UCCI: a0 là góc dưới bên trái (phe Đỏ).
    // Hệ tọa độ của Board: Row 9 là phe Đỏ (bottom), Row 0 là phe Đen (top).
    let from_col = chars[0] as usize - 'a' as usize;
    let from_row = 9 - (chars[1] as usize - '0' as usize);
    let to_col = chars[2] as usize - 'a' as usize;
    let to_row = 9 - (chars[3] as usize - '0' as usize);

    // Chuyển thẳng sang dense index (0..89)
    let from_dense = from_row * 9 + from_col;
    let to_dense = to_row * 9 + to_col;

    // Trả về index của Action Space (0..8099)
    from_dense * 90 + to_dense
}

/// Loại bỏ các thành phần rườm rà phía sau của FEN
fn clean_fen(fen: &str) -> String {
    let parts: Vec<&str> = fen.split_whitespace().collect();
    if parts.len() >= 2 {
        // Chỉ lấy bảng (parts[0]) và lượt đi (parts[1])
        format!("{} {}", parts[0], parts[1])
    } else {
        fen.to_string()
    }
}

/// Map score về khoảng [-1.0, 1.0]
fn map_score(eval_cp: f32) -> f32 {
    // khong map vi script python da map roi
    return eval_cp;
}

fn main() -> io::Result<()> {
    let input_path = "pikafish_FEN.csv";
    let output_path = "pikafish_FEN_processed.csv";

    let input_file = match File::open(input_path) {
        Ok(file) => file,
        Err(e) => {
            eprintln!(
                "Lỗi: Không thể mở file '{}' ({}). Bạn nhớ chạy ở thư mục chứa tệp này nhé.",
                input_path, e
            );
            return Err(e);
        }
    };
    let reader = BufReader::new(input_file);

    let mut output_file = File::create(output_path).expect("Không thể tạo file output");

    let mut lines = reader.lines();

    // Bỏ qua dòng đầu tiên (Header)
    if let Some(_) = lines.next() {}

    let mut count = 0;
    for line in lines {
        let line = line?;
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 3 {
            continue;
        }

        let raw_fen = parts[0];
        let eval_cp_str = parts[1];
        let best_move_str = parts[2];

        // 1. Chuyển FEN về dạng ngắn
        let cleaned_fen = clean_fen(raw_fen);

        // 2. Map score
        let eval_cp: f32 = eval_cp_str.parse().unwrap_or(0.0);
        let mapped_score = map_score(eval_cp);

        // 3. Get Index of best move
        let move_index = get_move_index(best_move_str);

        // 4. Lưu thành file CSV (ghi ra file output)
        writeln!(
            output_file,
            "{},{},{}",
            cleaned_fen, mapped_score, move_index
        )?;
        count += 1;
    }

    println!(
        "Xử lý thành công {} dòng! Đã xuất dữ liệu ra file '{}'.",
        count, output_path
    );
    Ok(())
}
