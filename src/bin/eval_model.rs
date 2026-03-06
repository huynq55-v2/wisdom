use burn::backend::NdArray;
use burn::tensor::Tensor;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::time::Instant;

use wisdom::board::{Board, Color};
use wisdom::nn::{XiangqiNetConfig, board_to_tensor};

// Kích thước của đầu vào
const CHANNELS: usize = 14;
const HEIGHT: usize = 10;
const WIDTH: usize = 9;

/// Hàm lật tọa độ 1 ô trên bàn cờ (0..89)
fn flip_sq90(sq90: usize) -> usize {
    89 - sq90
}

/// Hàm lật lại Action Index (0..8099)
fn flip_action_index(idx: usize) -> usize {
    let from_sq90 = idx / 90;
    let to_sq90 = idx % 90;

    let flipped_from = flip_sq90(from_sq90);
    let flipped_to = flip_sq90(to_sq90);

    flipped_from * 90 + flipped_to
}

fn main() -> io::Result<()> {
    // 1. Khởi tạo Device và Load Model
    let device = Default::default();
    let config = XiangqiNetConfig::new(); // Tùy chỉnh nếu bạn có tham số khác

    println!("⏳ Đang nạp mô hình từ file .mpk...");
    // Lưu ý: Đảm bảo bạn đã có file `xiangqi_net_weights.mpk`
    let model = config.load_model::<NdArray<f32>>("xiangqi_net_weights", &device);
    println!("✅ Nạp mô hình thành công!");

    // 2. Mở file dataset
    let dataset_path = "./wisdom_models/replay_buffer.csv";
    let file = match File::open(dataset_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("❌ Không tìm thấy file '{}': {}", dataset_path, e);
            return Ok(());
        }
    };
    let reader = BufReader::new(file);

    let mut total_valid_samples = 0;
    let mut correct_predictions = 0;
    let mut line_count = 0;

    println!("🚀 Bắt đầu chạy Evaluation...");
    let start_time = Instant::now();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        line_count += 1;

        // Xử lý an toàn: Biến đổi tất cả dấu phẩy (nếu là CSV) thành dấu cách
        let clean_line = line.replace(",", " ");

        let parts: Vec<&str> = clean_line.split_whitespace().collect();
        if parts.len() < 4 {
            // Nếu muốn debug dòng lỗi, bỏ comment dòng dưới:
            // println!("⚠️ Dòng {} thiếu dữ liệu", line_count);
            continue;
        }

        let board_fen = parts[0];
        let stm_str = parts[1];
        let _value: f32 = parts[2].parse().unwrap_or(0.0);
        let target_policy: usize = match parts[3].parse() {
            Ok(p) => p,
            Err(_) => continue,
        };

        // Lọc ngay các FEN có policy không hợp lệ (>= 8100)
        if target_policy >= 8100 {
            continue;
        }

        // Tạo FEN hoàn chỉnh để nạp vào Board
        let full_fen = format!("{} {}", board_fen, stm_str);
        let board = match Board::from_fen(&full_fen) {
            Ok(b) => b,
            Err(_) => continue, // Bỏ qua nếu FEN bị lỗi logic cờ
        };

        let is_black = board.side_to_move == Color::Black;

        // 3. Chuyển đổi Board thành mảng float (Hàm này đã TỰ ĐỘNG LẬT TENSOR nếu là Black)
        let tensor_data = board_to_tensor(&board);

        // Đưa vào Burn Tensor với Shape: [BatchSize=1, Channels=14, Height=10, Width=9]
        let input_tensor = Tensor::<NdArray<f32>, 1>::from_floats(tensor_data, &device)
            .reshape([1, CHANNELS, HEIGHT, WIDTH]);

        // 4. Chạy model (Inference)
        let (_pred_value, pred_logits) = model.forward(input_tensor);

        // 5. Lấy Argmax (Chỉ số có xác suất cao nhất)
        let logits_data = pred_logits.into_data();
        let logits_slice = logits_data
            .as_slice::<f32>()
            .expect("Không thể lấy slice f32");

        let mut max_logit = f32::NEG_INFINITY;
        let mut best_action_idx = 0;

        for (idx, &val) in logits_slice.iter().enumerate() {
            if val > max_logit {
                max_logit = val;
                best_action_idx = idx;
            }
        }

        // 6. Xử lý Lật lại Policy nếu phe Đen đi
        // Tensor bị lật -> Model xuất ra Index bị lật -> Ta phải lật lại để khớp với dataset gốc
        let mut final_pred_idx = best_action_idx;
        if is_black {
            final_pred_idx = flip_action_index(best_action_idx);
        }

        // 7. So sánh kết quả
        total_valid_samples += 1;
        if final_pred_idx == target_policy {
            correct_predictions += 1;
        }

        // 8. In tiến độ theo line_count (Cứ đọc xong 1000 dòng thì in 1 lần)
        if line_count % 1000 == 0 {
            let current_acc = if total_valid_samples > 0 {
                (correct_predictions as f64 / total_valid_samples as f64) * 100.0
            } else {
                0.0
            };
            println!(
                "Đang đọc dòng {:>7} | Hợp lệ: {:>7} | Đoán đúng: {:>7} | Accuracy: {:>5.2}%",
                line_count, total_valid_samples, correct_predictions, current_acc
            );
        }
    }

    let elapsed = start_time.elapsed();
    let final_acc = if total_valid_samples > 0 {
        (correct_predictions as f64 / total_valid_samples as f64) * 100.0
    } else {
        0.0
    };

    println!("\n========================================");
    println!("🏁 KẾT QUẢ ĐÁNH GIÁ (EVALUATION)");
    println!("========================================");
    println!("Thời gian chạy: {:.2} giây", elapsed.as_secs_f64());
    println!("Tổng số dòng đã duyệt: {}", line_count);
    println!("Tổng số mẫu hợp lệ: {}", total_valid_samples);
    println!("Số mẫu đoán đúng: {}", correct_predictions);
    println!("🎯 ĐỘ CHÍNH XÁC (ACCURACY): {:.2}%", final_acc);
    println!("========================================");

    Ok(())
}
