use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};

/// Chuyển đổi nước đi dạng UCCI (vd: "h2e2") thành Index của Action Space (0..8099)
fn get_move_index(best_move: &str) -> usize {
    let chars: Vec<char> = best_move.chars().collect();
    if chars.len() < 4 {
        return 0;
    }

    // 1. Trích xuất File (cột: a-i -> 0-8)
    let from_file = (chars[0] as usize) - ('a' as usize);
    let to_file = (chars[2] as usize) - ('a' as usize);

    // 2. Trích xuất Rank (hàng: 0-9).
    // Trong UCCI, '0' là hàng đáy phe Đỏ.
    // Trong Board array của bạn, Row 0 là đỉnh (phe Đen), Row 9 là đáy (phe Đỏ).
    // Công thức: array_row = 9 - ucci_rank
    let from_rank = 9 - ((chars[1] as usize) - ('0' as usize));
    let to_rank = 9 - ((chars[3] as usize) - ('0' as usize));

    // 3. Chuyển sang Square Index (0..89)
    // Công thức: index = row * 9 + col
    let from_sq90 = from_rank * 9 + from_file;
    let to_sq90 = to_rank * 9 + to_file;

    // 4. Trả về Action Index (0..8099)
    from_sq90 * 90 + to_sq90
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
