import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from safetensors.torch import save_file
import time
import os

# =====================================================================
# 1. HÀM CHUYỂN ĐỔI FEN SANG TENSOR VỚI PERSPECTIVE FIX
# =====================================================================
def fen_to_tensor(fen_str):
    parts = fen_str.split()
    board_part = parts[0]
    stm = parts[1].lower() if len(parts) > 1 else 'w'

    tensor = np.zeros((14, 10, 9), dtype=np.float32)
    piece_map = {
        'K': 0, 'A': 1, 'B': 2, 'N': 3, 'R': 4, 'C': 5, 'P': 6,
        'k': 7, 'a': 8, 'b': 9, 'n': 10, 'r': 11, 'c': 12, 'p': 13
    }

    raw_board = np.full((10, 9), -1, dtype=np.int32)
    row, col = 0, 0
    for char in board_part:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        elif char in piece_map:
            raw_board[row, col] = piece_map[char]
            col += 1

    for r in range(10):
        for c in range(9):
            p = raw_board[r, c]
            if p == -1: continue

            is_red = p <= 6
            piece_type = p if is_red else p - 7

            if stm == 'w':
                plane = piece_type if is_red else piece_type + 7
                tensor[plane, r, c] = 1.0
            else:
                mirrored_r = 9 - r
                mirrored_c = 8 - c
                plane = piece_type if not is_red else piece_type + 7
                tensor[plane, mirrored_r, mirrored_c] = 1.0

    return torch.from_numpy(tensor)

# =====================================================================
# 2. CÁC HÀM TIỆN ÍCH CHO ACTION SPACE 8100
# =====================================================================
def flip_sq90(sq90):
    # Lật tọa độ 180 độ trong mảng 1D (0..89)
    return 89 - sq90

def mirror_sq90_left_right(sq90):
    r, c = sq90 // 9, sq90 % 9
    return r * 9 + (8 - c)

# =====================================================================
# 3. DATASET
# =====================================================================
class XiangqiDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Đang nạp dữ liệu từ {csv_file}...")
        self.data = pd.read_csv(
            csv_file,
            header=None,
            names=['fen', 'value', 'policy'],
            dtype={'fen': str, 'value': np.float32, 'policy': np.int32}
        )
        # Lọc bỏ và không đưa vào training các mẫu có index < 0 hoặc >= 8100 
        valid_mask = (self.data['policy'] >= 0) & (self.data['policy'] < 8100)
        invalid_count = len(self.data) - valid_mask.sum()
        self.data = self.data[valid_mask].reset_index(drop=True)
        if invalid_count > 0:
            print(f"Bỏ qua hoàn toàn {invalid_count} vị trí có action index không hợp lệ.")
            
        print(f"Đã nạp {len(self.data)} positions hợp lệ.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen = row['fen']
        stm = fen.split()[1].lower() if len(fen.split()) > 1 else 'w'

        try:
            absolute_idx = int(row['policy'])
        except (ValueError, OverflowError):
            absolute_idx = -1

        # Tránh lỗi nếu bằng cách nào đó vẫn lọt vào __getitem__
        if absolute_idx < 0 or absolute_idx >= 8100:
            import random
            return self.__getitem__(random.randint(0, len(self) - 1))

        board_tensor = fen_to_tensor(fen)

        # GIẢI NÉN TỌA ĐỘ TUYỆT ĐỐI (Dựa trên mảng 90 ô)
        from_sq90 = absolute_idx // 90
        to_sq90 = absolute_idx % 90

        # Nếu là Đen -> Lật mặt thành "Nước đi ảo"
        if stm == 'b':
            from_sq90 = flip_sq90(from_sq90)
            to_sq90 = flip_sq90(to_sq90)

        # 2. DATA AUGMENTATION: 50% cơ hội lật đối xứng Trái - Phải
        if torch.rand(1).item() < 0.5:
            board_tensor = torch.flip(board_tensor, dims=[2])
            from_sq90 = mirror_sq90_left_right(from_sq90)
            to_sq90 = mirror_sq90_left_right(to_sq90)

        # Tính toán lại Index cho không gian 8100
        policy_idx = from_sq90 * 90 + to_sq90

        value = np.float32(row['value'])
        value = np.clip(value, -1.0, 1.0)

        return board_tensor, np.int64(policy_idx), value

# =====================================================================
# 4. MODEL RESNET (8 BLOCKS, 8100 POLICY)
# =====================================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class XiangqiNet(nn.Module):
    def __init__(self, num_res_blocks=8, channels=128):
        super(XiangqiNet, self).__init__()
        self.conv_input = nn.Conv2d(14, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])

        self.conv_policy = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_head = nn.Linear(2 * 10 * 9, 8100) # ĐÃ TĂNG LÊN 8100

        self.fc1 = nn.Linear(channels, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn_input(self.conv_input(x)))
        x_spatial = self.res_blocks(x)

        x_pol = self.conv_policy(x_spatial).view(batch_size, -1)
        logits_policy = self.policy_head(x_pol)

        x_val = x_spatial.view(batch_size, 128, -1).mean(dim=2)
        x_val = F.relu(self.fc1(x_val))
        value = torch.tanh(self.value_head(x_val))
        return value, logits_policy

# =====================================================================
# 5. HÀM TRAIN 
# =====================================================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    batch_size = 1024
    learning_rate = 2e-3  
    
    epochs = 15

    csv_path = "/kaggle/input/datasets/minhhoang2303/thisisforfun/replay_buffer.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file {csv_path}")
        return

    full_dataset = XiangqiDataset(csv_path)

    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # --- KHỞI TẠO MODEL & OPTIMIZER ---
    model = XiangqiNet(num_res_blocks=8, channels=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # T_max phải tính dựa trên TỔNG số bước của toàn bộ quá trình (epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)

    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0

    # Vòng lặp chạy từ start_epoch đến epochs
    for epoch in range(start_epoch, epochs):
        model.train()
        start_time = time.time()
        total_loss = 0
        total_loss_v = 0
        total_loss_p = 0

        for batch_idx, (boards, target_policies, target_values) in enumerate(train_loader):
            boards, target_policies = boards.to(device), target_policies.to(device)
            target_values = target_values.to(device).unsqueeze(1)

            optimizer.zero_grad()
            pred_values, pred_policies = model(boards)

            loss_v = mse_loss_fn(pred_values, target_values)
            loss_p = ce_loss_fn(pred_policies, target_policies)

            loss = loss_v + loss_p

            loss.backward()
            optimizer.step()
            
            # GIẢM LEARNING RATE MƯỢT MÀ TỪNG BATCH
            scheduler.step()

            total_loss += loss.item()
            total_loss_v += loss_v.item()
            total_loss_p += loss_p.item()

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_loss_v = total_loss_v / (batch_idx + 1)
                avg_loss_p = total_loss_p / (batch_idx + 1)
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch+1}] Batch [{batch_idx+1}/{len(train_loader)}] | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} (V: {avg_loss_v:.4f}, P: {avg_loss_p:.4f})")

        # Validation
        model.eval()
        val_loss_v, val_loss_p = 0, 0
        correct_policy = 0
        total_samples = 0

        with torch.no_grad():
            for boards, target_policies, target_values in val_loader:
                boards, target_policies = boards.to(device), target_policies.to(device)
                target_values = target_values.to(device).unsqueeze(1)

                pred_values, pred_policies = model(boards)

                val_loss_v += mse_loss_fn(pred_values, target_values).item() * boards.size(0)
                val_loss_p += ce_loss_fn(pred_policies, target_policies).item() * boards.size(0)

                _, predicted = torch.max(pred_policies, 1)
                correct_policy += (predicted == target_policies).sum().item()
                total_samples += boards.size(0)

        avg_val_v = val_loss_v / total_samples
        avg_val_p = val_loss_p / total_samples
        accuracy = (correct_policy / total_samples) * 100

        epoch_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1} HOÀN TẤT")
        print(f"{'='*60}")
        print(f"Train Time: {epoch_time:.2f}s")
        print(f"Val Value Loss: {avg_val_v:.4f} | Val Policy Loss: {avg_val_p:.4f}")
        print(f"Policy Accuracy (Top-1): {accuracy:.2f}%")
            
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), # Lưu cả lịch trình giảm LR
            'loss': loss,
        }

        checkpoint_path = f"./wisdom_net_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Đã lưu PyTorch checkpoint: {checkpoint_path}")

        state_dict_cpu = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
        safetensors_path = f"./wisdom_net_{epoch+1}.safetensors"
        save_file(state_dict_cpu, safetensors_path)
        print(f"✅ Đã lưu safetensors: {safetensors_path}")

        print(f"{'='*60}\n")

    print("🎉 Hoàn tất quá trình huấn luyện!")

if __name__ == "__main__":
    train_model()