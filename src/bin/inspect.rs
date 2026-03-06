use burn::backend::NdArray; 
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use wisdom::nn::XiangqiNetConfig;

fn main() {
    let device = Default::default();
    
    // Khởi tạo một mô hình rỗng (Random weights)
    let config = XiangqiNetConfig::new();
    let model = config.init::<NdArray<f32>>(&device);

    // Lưu cấu trúc ra file MPK
    model
        .clone()
        .save_file(
            "dummy_model",
            &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
        )
        .expect("Lỗi khi tạo file dummy");

    println!("✅ Đã sinh thành công file dummy_model.mpk!");
}