use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, PrettyJsonFileRecorder, Recorder};
use safetensors::SafeTensors;
use serde_json::Value;
use std::fs::File;
use std::io::{Read, Write};
use wisdom::nn::XiangqiNetConfig;

// Chuyển byte nhị phân thành số thực f32
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut f32_data = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        f32_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    f32_data
}

// Lật ngược ma trận cho các lớp Linear (Burn ngược chiều với PyTorch)
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }
    transposed
}

fn main() {
    let device = NdArrayDevice::Cpu;
    let config = XiangqiNetConfig::new();
    let model: wisdom::nn::XiangqiNet<NdArray<f32>> = config.init(&device);

    let input_file = "xiangqi_net_weights.safetensors";
    let output_file = "xiangqi_net_weights.mpk";
    let temp_json = "temp_convert.json";

    println!("📥 BƯỚC 1: Xuất cấu trúc (Khuôn mẫu) của Burn ra file tạm JSON...");
    PrettyJsonFileRecorder::<FullPrecisionSettings>::default()
        .record(model.into_record(), temp_json.into())
        .expect("Lỗi tạo file JSON tạm");

    println!("🔍 BƯỚC 2: Mở file khuôn mẫu và file safetensors...");
    let mut file = File::open(temp_json).unwrap();
    let mut json_str = String::new();
    file.read_to_string(&mut json_str).unwrap();
    let mut json_record: Value = serde_json::from_str(&json_str).unwrap();

    let mut st_file = File::open(input_file).expect("Không tìm thấy file safetensors");
    let mut buffer = Vec::new();
    st_file.read_to_end(&mut buffer).unwrap();
    let tensors = SafeTensors::deserialize(&buffer).unwrap();

    println!("💉 BƯỚC 3: Tiêm trực tiếp byte nhị phân vào khuôn mẫu...");
    for (name, view) in tensors.tensors() {
        if name.contains("num_batches_tracked") {
            continue; // Bỏ qua thông số thừa
        }

        let mut parts = name.split('.');
        let module = parts.next().unwrap();
        let mut param = parts.next().unwrap();

        // Burn đổi tên layer BatchNorm
        if module.contains("bn") {
            if param == "weight" {
                param = "gamma";
            }
            if param == "bias" {
                param = "beta";
            }
        }

        let mut data = bytes_to_f32(view.data());
        let mut shape = view.shape().to_vec();

        // Transpose Linear Layer (Từ [out, in] của PyTorch thành [in, out] của Burn)
        if (module == "policy_head" || module == "fc1" || module == "value_head")
            && param == "weight"
        {
            let out_features = shape[0];
            let in_features = shape[1];
            data = transpose_2d(&data, out_features, in_features);
            shape = vec![in_features, out_features];
        }

        // Cập nhật JSON tree
        let node = &mut json_record[module][param];
        if node.get("item").is_some() {
            node["item"]["value"] = serde_json::json!(data);
            node["item"]["shape"] = serde_json::json!(shape);
        } else {
            node["value"] = serde_json::json!(data);
            node["shape"] = serde_json::json!(shape);
        }
    }

    println!("📝 BƯỚC 4: Ghi khuôn JSON đã tiêm xong xuống đĩa...");
    let mut file = File::create(temp_json).unwrap();
    file.write_all(
        serde_json::to_string_pretty(&json_record)
            .unwrap()
            .as_bytes(),
    )
    .unwrap();

    println!("🧠 BƯỚC 5: Đọc lại JSON và chuyển hóa thành Mpk Native...");
    let loaded_record = PrettyJsonFileRecorder::<FullPrecisionSettings>::default()
        .load(temp_json.into(), &device)
        .unwrap();

    let final_model = config
        .init::<NdArray<f32>>(&device)
        .load_record(loaded_record);

    NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .record(final_model.into_record(), output_file.into())
        .expect("Lỗi lưu file mpk");

    // Xóa file rác
    let _ = std::fs::remove_file(temp_json);

    println!(
        "🎉 HOÀN TẤT! Đã tạo thành công file siêu tốc: {}",
        output_file
    );
}
