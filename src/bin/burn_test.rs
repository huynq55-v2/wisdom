use std::io::{self, IsTerminal, Write};
use std::thread;
use std::time::Duration;

fn main() {
    println!("======================================");
    println!("🔍 KIỂM TRA MÔI TRƯỜNG TERMINAL (TTY)");
    println!("======================================");

    // 1. Kiểm tra xem HĐH có cấp TTY cho tiến trình này không
    let is_tty = io::stdout().is_terminal();

    if is_tty {
        println!("✅ TRẠNG THÁI: Nhận diện thành công stdout là TTY chuẩn.");
        println!("✅ LÝ THUYẾT: Các thư viện vẽ UI (như burn) CÓ THỂ hoạt động.\n");

        println!("Đang thử nghiệm vẽ đồ họa ANSI (mô phỏng Dashboard)...");
        thread::sleep(Duration::from_secs(1));

        // 2. Thử vẽ đè lên cùng 1 dòng (cách các CLI Dashboard hoạt động)
        for i in 1..=100 {
            // \x1B[2K : Xóa toàn bộ dòng hiện tại
            // \x1B[1G : Đưa con trỏ chuột về đầu dòng
            print!("\x1B[2K\x1B[1G");

            // Vẽ thanh tiến trình
            print!("🚀 Tiến độ Training: [");
            let filled = i / 5;
            for _ in 0..filled {
                print!("█");
            }
            for _ in filled..20 {
                print!("-");
            }
            print!("] {}% ", i);

            // Ép buffer in ra ngay lập tức
            let _ = io::stdout().flush();

            thread::sleep(Duration::from_millis(40));
        }
        println!(
            "\n\n✅ Kiểm tra hoàn tất! Nếu bạn thấy thanh tiến trình chạy mượt mà trên 1 dòng, terminal của bạn hoàn toàn bình thường."
        );
    } else {
        println!("❌ TRẠNG THÁI: stdout KHÔNG PHẢI là TTY.");
        println!("❌ NGUYÊN NHÂN THƯỜNG GẶP:");
        println!(
            "   1. Bạn đang chạy code qua một nút bấm của IDE (như 'Run Code' extension) chứ không gõ 'cargo run' trực tiếp."
        );
        println!("   2. Bạn đang xuất file (vd: cargo run > log.txt).");
        println!("   3. Bạn đang dùng tmux/screen cấu hình sai.");
        println!("💡 KẾT LUẬN: Burn TUI tự động tắt để lùi về chế độ in text thô.");
    }
}
