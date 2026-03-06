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

# XÓA torch_directml, DÙNG ipex
import intel_extension_for_pytorch as ipex

# =====================================================================
# 1. HÀM CHUYỂN ĐỔI FEN SANG TENSOR VỚI PERSPECTIVE FIX
# =====================================================================
def fen_to_tensor(fen_str):
    parts = fen_str.split()
    board_part = parts[0]
    stm = parts[1].lower() if len(parts) > 1 else 'w'

    tensor = np.zeros((14, 10, 9), dtype=np.float32)
    piece_map = {
        'K': 0, 'A': 1, 'E': 2, 'H': 3, 'R': 4, 'C': 5, 'P': 6,
        'k': 7, 'a': 8, 'e': 9, 'h': 10, 'r': 11, 'c': 12, 'p': 13
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
# 2. CÁC HÀM TIỆN ÍCH CHO ACTION SPACE 4500
# =====================================================================
def flip_dense_sq(dense_sq):
    return 89 - dense_sq

def mirror_left_right(dense_sq):
    r, c = dense_sq // 9, dense_sq % 9
    return r * 9 + (8 - c)

def get_action_index(from_dense, to_dense):
    from_r, from_c = from_dense // 9, from_dense % 9
    to_r, to_c = to_dense // 9, to_dense % 9

    dr = to_r - from_r
    dc = to_c - from_c

    plane = 0
    if dr < 0 and dc == 0: plane = -dr - 1
    elif dr > 0 and dc == 0: plane = dr + 8
    elif dr == 0 and dc < 0: plane = -dc + 17
    elif dr == 0 and dc > 0: plane = dc + 25
    elif abs(dr) == 2 and abs(dc) == 1:
        if dr == -2 and dc == -1: plane = 34
        elif dr == -2 and dc == 1: plane = 35
        elif dr == 2 and dc == -1: plane = 36
        else: plane = 37
    elif abs(dr) == 1 and abs(dc) == 2:
        if dr == -1 and dc == -2: plane = 38
        elif dr == 1 and dc == -2: plane = 39
        elif dr == -1 and dc == 2: plane = 40
        else: plane = 41
    elif abs(dr) == 2 and abs(dc) == 2:
        if dr == -2 and dc == -2: plane = 42
        elif dr == -2 and dc == 2: plane = 43
        elif dr == 2 and dc == -2: plane = 44
        else: plane = 45
    elif abs(dr) == 1 and abs(dc) == 1:
        if dr == -1 and dc == -1: plane = 46
        elif dr == -1 and dc == 1: plane = 47
        elif dr == 1 and dc == -1: plane = 48
        else: plane = 49

    return from_dense * 50 + plane

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
        print(f"Đã nạp {len(self.data)} positions.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen = row['fen']
        stm = fen.split()[1].lower() if len(fen.split()) > 1 else 'w'

        board_tensor = fen_to_tensor(fen)

        try:
            absolute_idx = int(row['policy'])
        except (ValueError, OverflowError):
            absolute_idx = 0

        if absolute_idx < 0 or absolute_idx >= 8100:
            absolute_idx = 0

        from_dense = absolute_idx // 90
        to_dense = absolute_idx % 90

        if stm == 'b':
            from_dense = flip_dense_sq(from_dense)
            to_dense = flip_dense_sq(to_dense)

        if torch.rand(1).item() < 0.5:
            board_tensor = torch.flip(board_tensor, dims=[2])
            from_dense = mirror_left_right(from_dense)
            to_dense = mirror_left_right(to_dense)

        compact_policy_idx = get_action_index(from_dense, to_dense)

        value = np.float32(row['value'])
        value = np.clip(value, -1.0, 1.0)

        return board_tensor, np.int64(compact_policy_idx), value

# =====================================================================
# 4. MODEL RESNET (8 BLOCKS, 4500 POLICY)
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
        self.policy_head = nn.Linear(2 * 10 * 9, 4500)

        self.fc1 = nn.Linear(channels, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn_input(self.conv_input(x)))
        x_spatial = self.res_blocks(x)

        # Đổi .view thành .reshape
        x_pol = self.conv_policy(x_spatial).reshape(batch_size, -1)
        logits_policy = self.policy_head(x_pol)

        # Đổi .view thành .reshape
        x_val = x_spatial.reshape(batch_size, 128, -1).mean(dim=2)
        x_val = F.relu(self.fc1(x_val))
        value = torch.tanh(self.value_head(x_val))
        return value, logits_policy

# =====================================================================
# 5. CONVERT TO MPK FORMAT (Burn MessagePack)
# =====================================================================
def save_to_mpk(model, output_path):
    import msgpack
    state_dict = model.state_dict()
    burn_dict = {}
    for name, tensor in state_dict.items():
        data = tensor.cpu().numpy()
        burn_dict[name] = {
            'data': data.flatten().tolist(),
            'shape': list(data.shape),
            'dtype': 'f32' 
        }

    with open(output_path, 'wb') as f:
        packed = msgpack.packb(burn_dict, use_bin_type=True)
        f.write(packed)
    print(f"✅ Đã lưu model sang định dạng .mpk: {output_path}")

# =====================================================================
# 6. HÀM TRAIN 
# =====================================================================
def train_model():
    # === THIẾT LẬP THIẾT BỊ BẰNG IPEX (XPU) ===
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"🚀 Tuyệt vời! Đang sử dụng Intel GPU (XPU): {torch.xpu.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ Không tìm thấy Intel XPU, đang chuyển sang cày bằng CPU.")

    batch_size = 64 # Giảm Batch size xuống cho máy local đỡ ngộp
    learning_rate = 1e-3  
    epochs = 3 # Local thì test nhẹ nhàng thôi

    # === LƯU Ý: ĐƯỜNG DẪN LOCAL ===
    # Hãy trỏ về file csv có trên máy tính của bạn
    csv_path = "replay_buffer.csv" 
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file {csv_path}. Bạn nhớ copy file CSV về thư mục này nhé.")
        return

    full_dataset = XiangqiDataset(csv_path)

    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = XiangqiNet(num_res_blocks=8, channels=128).to(device)

    # === LOAD TRƯỚC KHI OPTIMIZE ===
    latest_ckpt = "./wisdom_models/xiangqi_net_v3_python_latest.pth"
    start_epoch = 0
    checkpoint = None
    if os.path.exists(latest_ckpt):
        print(f"📂 Tìm thấy checkpoint V3, đang load từ {latest_ckpt}...")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"✅ Đã load model từ epoch {start_epoch}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if checkpoint and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # === TỐI ƯU HÓA BẰNG IPEX ===
    # Đây là dòng thần chú bắt buộc để kích hoạt sức mạnh của Intel
    model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=1e-5)

    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, start_epoch + epochs):
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

            value_weight = 2.0
            loss = (value_weight * loss_v) + loss_p

            loss.backward()
            optimizer.step()
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

        if not os.path.exists("./wisdom_models"):
            os.makedirs("./wisdom_models")
            
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        checkpoint_path = f"./wisdom_models/xiangqi_net_v3_python_epoch_{epoch+1}.pth"
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, latest_ckpt)
        print(f"✅ Đã lưu PyTorch checkpoint: {checkpoint_path}")

        # Lấy model state_dict gốc (bỏ qua các lớp bọc của IPEX nếu có)
        state_dict_cpu = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
        safetensors_path = f"./wisdom_models/xiangqi_net_v3_python_epoch_{epoch+1}.safetensors"
        save_file(state_dict_cpu, safetensors_path)
        print(f"✅ Đã lưu safetensors: {safetensors_path}")

        mpk_path = f"./wisdom_models/xiangqi_net_v3_python_{epoch+1}.mpk"
        save_to_mpk(model, mpk_path)

        print(f"{'='*60}\n")

    print("🎉 Hoàn tất quá trình huấn luyện V3 (3 Epoch)!")

if __name__ == "__main__":
    train_model()