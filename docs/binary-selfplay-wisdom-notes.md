# Notes cho 2 binary chính: `wisdom` và `selfplay`

## A) `src/bin/wisdom.rs`

### Vai trò

- Là binary engine chính để chạy UCCI.
- Có 2 mode phần cứng:
  - CPU (`NdArray`)
  - GPU (`Wgpu`)
- Có mode phụ `perft` để đếm node kiểm tra movegen.

### Luồng chạy

1. Parse args.
2. Nếu `perft`: tạo `Board` ban đầu, gọi `perft(depth)` rồi thoát.
3. Nếu engine mode:
   - tạo `XiangqiNetConfig`
   - load model từ `xiangqi_net_weights.mpk` nếu có, nếu không init random
   - gọi `ucci_loop_generic`.

### Ý nghĩa

`wisdom` là binary phục vụ chơi cờ/thi đấu với GUI UCCI, không phải pipeline train.

---

## B) `src/bin/selfplay.rs`

### Vai trò

- Binary train tổng hợp: vừa tự chơi vừa huấn luyện.
- Vòng lặp iteration:
  1. Self-play sinh dữ liệu mới
  2. cập nhật replay buffer (RAM + CSV)
  3. train model 1 epoch
  4. lưu checkpoint + mpk

### Điểm chính trong self-play

- Mỗi ván dùng MCTS để chọn nước.
- Trước mỗi lượt có check repetition từ `history`.
- Khi đánh nước, code tạo đầy đủ `HistoryEntry` (`hash/is_check/chased_set/is_reversible`).
- Kết thúc ván: blend nhãn `search_value` và ground-truth thắng/thua/hòa để tạo `SelfPlayItem`.

### Hạ tầng hiệu năng

- Dùng `EvalQueue` để batch inference NN.
- Chạy nhiều game đồng thời bằng scoped threads.
- Dùng `TranspositionTable` dùng chung.

### Persistence

- replay buffer tại `./wisdom_models/replay_buffer.csv`.
- model checkpoint dạng compact: `xiangqi_net_ckpt_{iter}`.
- model export dùng engine/GUI: `xiangqi_net_{iter}.mpk`.

---

## C) Liên hệ giữa 2 binary

- `selfplay` tạo/huấn luyện model.
- `wisdom` dùng model đã train để chơi qua UCCI.
- Cả hai dùng chung phần lõi ở `src/` (`board`, `mcts`, `nn`, `ucci`, `tt`, ...).
