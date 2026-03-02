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

    let output_file = "xiangqi_dataset_cleaned.csv";

    println!("Bắt đầu dọn dẹp (Canonicalize) cho file: {}", input_file);
    let start_time = Instant::now();

    let infile = File::open(input_file).expect("Không thể mở file input");
    let reader = BufReader::new(infile);

    let outfile = File::create(output_file).expect("Không thể tạo file output");
    let mut writer = BufWriter::new(outfile);

    // Ghi Header
    writeln!(writer, "fen,policy,value").unwrap();

    let mut line_count = 0;
    let mut flipped_count = 0;

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

        let fen_parts: Vec<&str> = fen.split(' ').collect();
        let stm = if fen_parts.len() > 1 {
            fen_parts[1]
        } else {
            "w"
        };

        if stm.to_lowercase() == "b" {
            // Transform Black's perspective to Red's perspective
            let flipped_fen = mirror_fen_and_swap_colors(fen);
            let flipped_policy = mirror_policy_180(policy_idx);
            // Value is assumed from STM perspective so we keep it the same.

            // Note: Our value is MCTS winrate/evaluation from current player's perspective.
            // Actually, wait: If the dataset records value from the POV of the person who just played, or the person who is ABOUT to play?
            // Usually value is "Winning chance of side-to-move".
            // If it is from the perspective of Side-to-move, we DO NOT invert value when flipping the board!
            // Let's assume value is from STM perspective (which is standard). We keep value the same.

            if writeln!(writer, "{},{},{}", flipped_fen, flipped_policy, value).is_err() {
                break;
            }
            flipped_count += 1;
        } else {
            // Already White/Red to move, leave as-is
            if writeln!(writer, "{},{},{}", fen, policy_idx, value).is_err() {
                break;
            }
        }

        line_count += 1;
        if line_count % 1_000_000 == 0 {
            println!(
                "Đã xử lý {} ván cờ (Đã lật góc nhìn {} ván)...",
                line_count, flipped_count
            );
        }
    }

    println!("========================================");
    println!("Hoàn thành tạo Canonical Dataset!");
    println!("Tổng số dòng: {}", line_count);
    println!("Số dòng Black đã lật thành Red: {}", flipped_count);
    println!("Thời gian chạy: {:?}", start_time.elapsed());
    println!("Đã lưu file mới tại: {}", output_file);
}

/// Lật 180 độ (dọc + ngang) và Tráo màu cờ (Hoa <-> Thường)
fn mirror_fen_and_swap_colors(fen: &str) -> String {
    let parts: Vec<&str> = fen.split(' ').collect();
    let board_part = parts[0];

    // Lật 180 độ nghĩa là Row 0 <-> Row 9, và Col 0 <-> Col 8
    let mut expanded_rows: Vec<String> = board_part
        .split('/')
        .map(|row| {
            let mut expanded = String::new();
            for c in row.chars() {
                if let Some(digit) = c.to_digit(10) {
                    expanded.push_str(&"1".repeat(digit as usize));
                } else {
                    // Swap colors! Upper case (Red) to Lower case (Black) and vice versa
                    let swapped = if c.is_ascii_uppercase() {
                        c.to_ascii_lowercase()
                    } else {
                        c.to_ascii_uppercase()
                    };
                    expanded.push(swapped);
                }
            }
            expanded
        })
        .collect();

    // Đảo ngược thứ tự các hàng (Row 0 <-> Row 9)
    expanded_rows.reverse();

    // Mỗi hàng, đảo ngược thứ tự các cột (Col 0 <-> Col 8)
    let mirrored_board: Vec<String> = expanded_rows
        .into_iter()
        .map(|row| {
            let reversed: String = row.chars().rev().collect();
            compress_row(&reversed)
        })
        .collect();

    // Sau khi lật, lượt đi gán cứng là 'w'
    format!("{} w", mirrored_board.join("/"))
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

/// Xoay 180 độ toạ độ policy
fn mirror_policy_180(index: usize) -> usize {
    let m = index_to_move(index);

    let from_sq = m.0 as usize;
    let to_sq = m.1 as usize;

    let f_row = from_sq / 16;
    let f_col = from_sq % 16;
    let t_row = to_sq / 16;
    let t_col = to_sq % 16;

    // Rotate 180 (row -> 9 - row, col -> 8 - col)
    let nf_row = 9 - f_row;
    let nf_col = 8 - f_col;
    let nt_row = 9 - t_row;
    let nt_col = 8 - t_col;

    let nf_sq = nf_row * 16 + nf_col;
    let nt_sq = nt_row * 16 + nt_col;

    let mirrored_move = Move::new(nf_sq as usize, nt_sq as usize, false);
    move_to_index(mirrored_move)
}
