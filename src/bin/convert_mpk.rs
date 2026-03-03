use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, PrettyJsonFileRecorder, Recorder};
use safetensors::SafeTensors;
use serde_json::Value;
use std::fs::File;
use std::io::{Read, Write};
use wisdom::nn::XiangqiNetConfig;

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    let mut f32_data = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        f32_data.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    f32_data
}

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

    let input_file = "xiangqi_net_weights_latest.safetensors";
    let output_file = "xiangqi_net_latest.mpk";
    let temp_json = "temp_convert.json";

    println!("📥 BƯỚC 1: Xuất khuôn mẫu JSON...");
    PrettyJsonFileRecorder::<FullPrecisionSettings>::default()
        .record(model.into_record(), temp_json.into())
        .expect("Lỗi tạo file JSON tạm");

    let mut file = File::open(temp_json).unwrap();
    let mut json_str = String::new();
    file.read_to_string(&mut json_str).unwrap();
    let mut json_record: Value = serde_json::from_str(&json_str).unwrap();

    let mut st_file = File::open(input_file).expect("Không tìm thấy file safetensors");
    let mut buffer = Vec::new();
    st_file.read_to_end(&mut buffer).unwrap();
    let tensors = SafeTensors::deserialize(&buffer).unwrap();

    println!("💉 BƯỚC 3: Tiêm dữ liệu (Mapping thông minh)...");
    for (name, view) in tensors.tensors() {
        if name.contains("num_batches_tracked") || name.contains("running_") {
            continue; // Burn mặc định không dùng các thông số này trong record cơ bản nếu không cấu hình thêm
        }

        let parts: Vec<&str> = name.split('.').collect();
        let mut data = bytes_to_f32(view.data());
        let mut shape = view.shape().to_vec();

        // Xử lý Linear Transpose
        if (name.contains("policy_head") || name.contains("fc1") || name.contains("value_head"))
            && name.contains("weight")
        {
            let out_f = shape[0];
            let in_f = shape[1];
            data = transpose_2d(&data, out_f, in_f);
            shape = vec![in_f, out_f];
        }

        let mut current_node = &mut json_record;

        for (i, &part) in parts.iter().enumerate() {
            let mut target_key = part.to_string();

            // FIX 1: Map BatchNorm names
            if (target_key == "weight" || target_key == "bias") && i > 0 {
                let parent = parts[i - 1];
                if parent.contains("bn") {
                    target_key = if target_key == "weight" {
                        "gamma".to_string()
                    } else {
                        "beta".to_string()
                    };
                }
            }

            // FIX 2: Xử lý Vec<ResBlock>
            // PyTorch: res_blocks.0.conv1 -> Burn JSON: res_blocks[0].conv1
            if target_key == "res_blocks" {
                current_node = &mut current_node["res_blocks"];
                continue;
            }

            // Nếu part tiếp theo là số (index của Vec)
            if let Ok(idx) = target_key.parse::<usize>() {
                if current_node.is_array() {
                    current_node = &mut current_node[idx];
                } else {
                    // Đôi khi Burn lưu index dưới dạng khóa chuỗi "0", "1"...
                    current_node = &mut current_node[target_key.clone()];
                }
            } else {
                // Truy cập thông thường, nếu không thấy thì thử snake_case (nếu Burn có biến đổi)
                if current_node.get(&target_key).is_some() {
                    current_node = &mut current_node[target_key];
                } else {
                    println!(
                        "⚠️ Cảnh báo: Bỏ qua lớp {} vì không tìm thấy khóa tương ứng trong Burn",
                        name
                    );
                    break;
                }
            }

            // Nếu đã đến phần tử cuối (weight/bias/gamma/beta)
            if i == parts.len() - 1 {
                if let Some(item) = current_node.get_mut("item") {
                    item["value"] = serde_json::json!(data);
                    item["shape"] = serde_json::json!(shape);
                } else {
                    current_node["value"] = serde_json::json!(data);
                    current_node["shape"] = serde_json::json!(shape);
                }
            }
        }
    }

    println!("📝 BƯỚC 4: Đóng gói...");
    let mut file = File::create(temp_json).unwrap();
    file.write_all(
        serde_json::to_string_pretty(&json_record)
            .unwrap()
            .as_bytes(),
    )
    .unwrap();

    let loaded_record = PrettyJsonFileRecorder::<FullPrecisionSettings>::default()
        .load(temp_json.into(), &device)
        .unwrap();

    let final_model = config
        .init::<NdArray<f32>>(&device)
        .load_record(loaded_record);

    NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .record(final_model.into_record(), output_file.into())
        .expect("Lỗi lưu file mpk");

    println!("🎉 THÀNH CÔNG! Đã tạo: {}", output_file);
}
