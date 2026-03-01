import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from safetensors.torch import save_file
import time

# =====================================================================
# 1. HÀM CHUYỂN ĐỔI FEN SANG TENSOR (Đồng bộ 100% với file nn.rs)
# =====================================================================
def fen_to_tensor(fen_str):
    """
    Chuyển FEN string thành Tensor shape [15, 10, 9]
    """
    parts = fen_str.split()
    board_part = parts[0]
    stm = parts[1] if len(parts) > 1 else 'w' # 'w' là Đỏ, 'b' là Đen

    # Khởi tạo mảng numpy 15 mặt phẳng, kích thước 10x9
    tensor = np.zeros((15, 10, 9), dtype=np.float32)

    # Map quân cờ với index mặt phẳng (0-6 cho Đỏ, 7-13 cho Đen)
    piece_map = {
        'K': 0, 'A': 1, 'E': 2, 'H': 3, 'R': 4, 'C': 5, 'P': 6, # Đỏ (Viết hoa)
        'k': 7, 'a': 8, 'e': 9, 'h': 10, 'r': 11, 'c': 12, 'p': 13 # Đen (Viết thường)
    }

    row = 0
    col = 0
    for char in board_part:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        elif char in piece_map:
            plane = piece_map[char]
            tensor[plane, row, col] = 1.0
            col += 1

    # Mặt phẳng 14: Lượt đi. Nếu Đỏ đi thì toàn bộ là 1.0, Đen đi là 0.0
    if stm.lower() == 'w':
        tensor[14, :, :] = 1.0

    return torch.from_numpy(tensor)

# =====================================================================
# 2. ĐỊNH NGHĨA DATASET ĐỌC TỪ CSV
# =====================================================================
class XiangqiDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Đang nạp dữ liệu từ {csv_file}...")
        self.data = pd.read_csv(csv_file)
        print(f"Đã nạp {len(self.data)} ván cờ.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Lấy Input (FEN)
        fen = row['fen']
        board_tensor = fen_to_tensor(fen)
        
        # Lấy Target Policy (0 - 8099)
        policy_idx = int(row['policy'])
        
        # Lấy Target Value (-1.0 đến 1.0)
        value = np.float32(row['value'])
        
        return board_tensor, policy_idx, value

# =====================================================================
# 3. ĐỊNH NGHĨA MODEL NEURAL NETWORK
# =====================================================================
class XiangqiNet(nn.Module):
    def __init__(self):
        super(XiangqiNet, self).__init__()
        
        # --- Shared Convolutional Blocks ---
        self.conv1 = nn.Conv2d(15, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # --- Policy Head Branch ---
        self.conv_policy = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_head = nn.Linear(2 * 10 * 9, 8100) # 8100 là Action Space
        
        # --- Value Head Branch ---
        self.fc1 = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        batch_size = x.size(0)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_spatial = F.relu(self.bn3(self.conv3(x)))
        
        # Policy Forward
        x_pol = self.conv_policy(x_spatial)
        x_pol = x_pol.view(batch_size, -1) # Flatten thành [Batch, 180]
        logits_policy = self.policy_head(x_pol)
        
        # Value Forward (Global Average Pooling)
        x_val = x_spatial.view(batch_size, 128, -1).mean(dim=2) # Shape: [Batch, 128]
        x_val = F.relu(self.fc1(x_val))
        value = torch.tanh(self.value_head(x_val))
        
        return value, logits_policy

# =====================================================================
# 4. HÀM TRAIN CHÍNH
# =====================================================================
def train_model():
    # Cấu hình thiết bị (Ưu tiên GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # Hyperparameters
    batch_size = 256
    learning_rate = 1e-3
    epochs = 10 # Bạn có thể tăng lên nếu loss vẫn đang giảm tốt

    # Khởi tạo Model, Dataset, DataLoader
    model = XiangqiNet().to(device)
    dataset = XiangqiDataset("./xiangqi_dataset.csv") # Đường dẫn file CSV tạo từ Rust
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, # Dùng đa luồng nạp data trên Kaggle (tùy chỉnh 2-4)
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Định nghĩa Hàm Mất Mát (Loss Functions) và Tối Ưu (Optimizer)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss_fn = nn.MSELoss()             # Cho Value
    ce_loss_fn = nn.CrossEntropyLoss()     # Cho Policy

    print("Bắt đầu huấn luyện...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_v_loss = 0.0
        total_p_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (boards, target_policies, target_values) in enumerate(dataloader):
            boards = boards.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device).unsqueeze(1) # Reshape [Batch] -> [Batch, 1]

            optimizer.zero_dict()

            # Chạy qua mạng
            pred_values, pred_policies = model(boards)

            # Tính Loss
            loss_v = mse_loss_fn(pred_values, target_values)
            loss_p = ce_loss_fn(pred_policies, target_policies)
            loss = loss_v + loss_p

            # Cập nhật trọng số
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_v_loss += loss_v.item()
            total_p_loss += loss_p.item()

            # Cứ mỗi 100 batch thì in ra log một lần
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(dataloader)}] | "
                      f"Loss: {loss.item():.4f} (V_Loss: {loss_v.item():.4f}, P_Loss: {loss_p.item():.4f})")

        # Thống kê Epoch
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"==> KẾT THÚC EPOCH {epoch+1} | Avg Loss: {avg_loss:.4f} | Thời gian: {epoch_time:.2f}s\n")

    # =====================================================================
    # 5. LƯU MODEL RA ĐỊNH DẠNG SAFETENSORS CHO RUST
    # =====================================================================
    output_path = "xiangqi_net_weights.safetensors"
    
    # Để an toàn cho Burn (Rust) đọc, ta thường đẩy mô hình về lại CPU trước khi lưu
    model.eval()
    model.to("cpu")
    
    save_file(model.state_dict(), output_path)
    print(f"🎉 Hoàn tất! Model đã được lưu thành công tại: {output_path}")

if __name__ == "__main__":
    train_model()