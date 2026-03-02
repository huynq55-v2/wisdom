use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;
use wisdom::r#move::Move;
use wisdom::nn::{index_to_move, move_to_index};

fn main() {
    let args: Vec<String> = env::args().collect();
    let input_file = if args.len() > 1 {
        &args[1]
    } else {
        "xiangqi_dataset.csv"
    };

    let output_file = "xiangqi_dataset_augmented.csv";

    println!("Bắt đầu Augmentation (Mirroring) cho file: {}", input_file);
    let start_time = Instant::now();

    let infile = File::open(input_file).expect("Không thể mở file input");
    let reader = BufReader::new(infile);

    let outfile = File::create(output_file).expect("Không thể tạo file output");
    let mut writer = BufWriter::new(outfile);

    // Ghi Header
    writeln!(writer, "fen,policy,value").unwrap();

    let mut line_count = 0;

    for line in reader.lines() {
        let l = line.expect("Lỗi đọc dòng");
        if l.starts_with("fen") || l.is_empty() {
            continue;
        }

        let parts: Vec<&str> = l.split(',').collect();
        if parts.len() < 3 {
            continue;
        }

        let fen = parts[0];
        let policy_idx_str = parts[1].trim();
        let value = parts[2].trim();

        let policy_idx: usize = match policy_idx_str.parse() {
            Ok(idx) => idx,
            Err(_) => continue,
        };

        // 1. Ghi lại dòng Data gốc
        if writeln!(writer, "{},{},{}", fen, policy_idx, value).is_err() {
            break;
        }

        // 2. Tạo Data đối xứng (Mirror)
        let mirrored_fen = mirror_fen(fen);
        let mirrored_policy = mirror_policy(policy_idx);

        // 3. Ghi dòng Data đối xứng
        if writeln!(writer, "{},{},{}", mirrored_fen, mirrored_policy, value).is_err() {
            break;
        }

        line_count += 1;
        if line_count % 1_000_000 == 0 {
            println!(
                "Đã xử lý {} ván cờ (Tạo ra {} dòng data)...",
                line_count,
                line_count * 2
            );
        }
    }

    println!("========================================");
    println!("Hoàn thành Data Augmentation!");
    println!("Dữ liệu gốc: {} dòng", line_count);
    println!("Dữ liệu sau khi Augment: {} dòng", line_count * 2);
    println!("Thời gian chạy: {:?}", start_time.elapsed());
    println!("Đã lưu file mới tại: {}", output_file);
}

/// Lật ngược FEN theo chiều dọc (cột 0 thành cột 8, cột 1 thành cột 7...)
fn mirror_fen(fen: &str) -> String {
    let parts: Vec<&str> = fen.split(' ').collect();
    let board_part = parts[0];

    let mirrored_board: Vec<String> = board_part
        .split('/')
        .map(|row| {
            let mut expanded = String::new();
            for c in row.chars() {
                if let Some(digit) = c.to_digit(10) {
                    expanded.push_str(&"1".repeat(digit as usize));
                } else {
                    expanded.push(c);
                }
            }
            // Đảo ngược chuỗi đã expand (ví dụ: "R1...H" -> "H...1R")
            let reversed: String = expanded.chars().rev().collect();

            // Nén lại thành FEN (ví dụ: "H111" -> "H3")
            compress_row(&reversed)
        })
        .collect();

    let mut result = mirrored_board.join("/");
    if parts.len() > 1 {
        result.push(' ');
        result.push_str(parts[1]); // Giữ nguyên lượt đi w/b
    }
    result
}

fn compress_row(row: &str) -> String {
    let mut compressed = String::new();
    let mut empty_count = 0;
    for c in row.chars() {
        if c == '1' {
            empty_count += 1;
        } else {
            if empty_count > 0 {
                compressed.push_str(&empty_count.to_string());
                empty_count = 0;
            }
            compressed.push(c);
        }
    }
    if empty_count > 0 {
        compressed.push_str(&empty_count.to_string());
    }
    compressed
}

/// Lật ngược nước đi: (col_from, row_from) -> (8 - col_from, row_from)
fn mirror_policy(index: usize) -> usize {
    // Bước 1: Chuyển index về Move object
    let m = index_to_move(index);

    // Bước 2: Tính toán tọa độ mới
    let from_sq = m.0 as usize;
    let to_sq = m.1 as usize;

    // Chuyển 0x88 Square sang (row, col) - Chú ý: 0x88 dùng col (0-8) row (0-9)
    let f_row = from_sq / 16;
    let f_col = from_sq % 16;
    let t_row = to_sq / 16;
    let t_col = to_sq % 16;

    // Lật cột: new_col = 8 - old_col
    let nf_col = 8 - f_col;
    let nt_col = 8 - t_col;

    let nf_sq = f_row * 16 + nf_col;
    let nt_sq = t_row * 16 + nt_col;

    let mirrored_move = Move::new(nf_sq as usize, nt_sq as usize, false);

    move_to_index(mirrored_move)
}
