import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from safetensors.torch import save_file
import time

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
    
    # Render raw board first
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

    # Map to tensor based on perspective
    for r in range(10):
        for c in range(9):
            p = raw_board[r, c]
            if p == -1: continue
            
            is_red = p <= 6
            piece_type = p if is_red else p - 7
            
            if stm == 'w':
                # Red to move: Red pieces on 0-6, Black on 7-13. No rotation.
                plane = piece_type if is_red else piece_type + 7
                tensor[plane, r, c] = 1.0
            else:
                # Black to move: Black pieces on 0-6, Red on 7-13. Rotate 180 degrees.
                mirrored_r = 9 - r
                mirrored_c = 8 - c
                plane = piece_type if not is_red else piece_type + 7
                tensor[plane, mirrored_r, mirrored_c] = 1.0

    return torch.from_numpy(tensor)

# =====================================================================
# 2. DATASET
# =====================================================================
class XiangqiDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Đang nạp dữ liệu từ {csv_file}...")
        self.data = pd.read_csv(csv_file)
        self.data['value'] = self.data['value'].astype(np.float32) + 0.0
        print(f"Đã nạp {len(self.data)} ván cờ.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen = row['fen']
        stm = fen.split()[1].lower() if len(fen.split()) > 1 else 'w'
        
        board_tensor = fen_to_tensor(fen)
        
        # Policy Fix: If Black to move, mirror the recorded move policy as well!
        policy_idx = int(row['policy'])
        if stm == 'b':
            from_dense = policy_idx // 90
            to_dense = policy_idx % 90
            
            from_r, from_c = from_dense // 9, from_dense % 9
            to_r, to_c = to_dense // 9, to_dense % 9
            
            # Mirror
            from_r, from_c = 9 - from_r, 8 - from_c
            to_r, to_c = 9 - to_r, 8 - to_c
            
            mirrored_from = from_r * 9 + from_c
            mirrored_to = to_r * 9 + to_c
            
            policy_idx = mirrored_from * 90 + mirrored_to
            
        value = np.float32(row['value'])
        return board_tensor, policy_idx, value

# =====================================================================
# 3. MODEL RESNET
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
    def __init__(self, num_res_blocks=7, channels=128):
        super(XiangqiNet, self).__init__()
        self.conv_input = nn.Conv2d(14, channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])
        
        self.conv_policy = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_head = nn.Linear(2 * 10 * 9, 8100)
        
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
# 4. HÀM TRAIN & VALIDATION
# =====================================================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    batch_size = 512
    learning_rate = 5e-4
    epochs = 10

    full_dataset = XiangqiDataset("/kaggle/input/datasets/huyquang2309/xiangqi-mcts/xiangqi_dataset_augmented_1.csv")

    val_size = int(0.05 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = XiangqiNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
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

            if (batch_idx + 1) % 500 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

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
        print(f"==> KẾT THÚC EPOCH {epoch+1}")
        print(f"Train Time: {epoch_time:.2f}s")
        print(f"Val Value Loss: {avg_val_v:.4f} | Val Policy Loss: {avg_val_p:.4f}")
        print(f"Policy Accuracy (Top-1): {accuracy:.2f}%")
        print("-" * 50)

    model.to("cpu")
    save_file(model.state_dict(), "xiangqi_net_weights.safetensors")
    print("🎉 Hoàn tất! Đã lưu model.")

if __name__ == "__main__":
    train_model()