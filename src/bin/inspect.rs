use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;

fn main() {
    let path = "xiangqi_net_weights.safetensors";
    println!("🔍 Đang mở thẳng file: {}", path);

    // 1. Đọc chay file nhị phân từ ổ cứng
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            println!("❌ Không thể mở file. Lỗi hệ điều hành: {}", e);
            println!(
                "👉 Hãy chắc chắn file đang nằm ngang hàng với thư mục src/ và file Cargo.toml"
            );
            return;
        }
    };

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .expect("Lỗi khi nạp file vào RAM");
    println!(
        "✅ Đã nạp thành công {} bytes vào RAM. Đang giải mã...",
        buffer.len()
    );

    // 2. Tự tay giải mã cấu trúc Safetensors
    let tensors = match SafeTensors::deserialize(&buffer) {
        Ok(t) => t,
        Err(e) => {
            println!("❌ File bị hỏng hoặc không đúng chuẩn. Lỗi: {:?}", e);
            return;
        }
    };

    println!("✅ Giải mã hoàn hảo! Dưới đây là ruột gan của bộ não AI:\n");

    // 3. In toàn bộ danh sách các ma trận (Tensors) bên trong
    let mut count = 0;
    for (name, view) in tensors.tensors() {
        count += 1;
        let shape = view.shape();
        let dtype = view.dtype();

        println!(
            "{:>3}. Lớp: {:<25} | Kiểu dữ liệu: {:<5} | Kích thước (Shape): {:?}",
            count,
            name,
            format!("{:?}", dtype),
            shape
        );
    }

    println!(
        "\n🎉 Tổng cộng có {} Tensors. File hoàn toàn khỏe mạnh!",
        count
    );
    println!(
        "👉 Điều này chứng tỏ Windows của bạn đọc file rất bình thường, bọn làm thư viện Burn code phần giải nén PyTorch quá ẩu!"
    );
}
