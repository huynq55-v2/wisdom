# Project Overview (wisdom)

## 1) Cấu trúc chính

- `src/lib.rs`: khai báo các module lõi của engine.
- `src/board.rs`: biểu diễn bàn cờ Xiangqi, luật hợp lệ, check/repetition, FEN.
- `src/move.rs`, `src/movegen.rs`: định nghĩa nước đi + sinh nước đi.
- `src/mcts.rs`: tìm kiếm MCTS (multi-thread, virtual loss, TT, lazy repetition check).
- `src/nn.rs`: ánh xạ bàn cờ -> tensor, model `XiangqiNet`, mapping move-index.
- `src/eval_queue.rs`: hàng đợi batch inference NN.
- `src/tt.rs`: transposition table.
- `src/ucci.rs`: loop giao thức UCCI để chạy engine với GUI.

## 2) Binary hiện có

- `src/bin/wisdom.rs`: entrypoint engine UCCI (CPU/GPU + perft mode).
- `src/bin/selfplay.rs`: pipeline tự chơi + train lặp (self-play -> replay buffer -> training).
- `src/bin/gui.rs`: GUI local.
- các binary dữ liệu/phụ trợ: `build_dataset`, `augment`, `inspect`, `convert_mpk`, `clean_dataset`.

## 3) Luồng tổng quát

1. `wisdom` nạp model (`.mpk`) nếu có.
2. vào `ucci_loop_generic` để nhận lệnh `position`, `go`, `quit`.
3. `go` gọi MCTS + NN batch inference để chọn `bestmove`.

Song song đó, `selfplay` dùng MCTS để sinh dữ liệu game mới, ghi replay buffer, rồi train model theo iteration.

## 4) Điểm kỹ thuật đáng chú ý

- Board dùng mảng 256 theo kiểu 0x88 biến thể cho bàn 10x9.
- `HistoryEntry` được dùng xuyên suốt để judge repetition/chasing.
- MCTS đã seed lịch sử thật (`game_history`) vào playout qua `relevant_history` để tránh lệch luật lặp.
- Có tách hạ tầng inference (EvalQueue) khỏi search để tận dụng batch GPU.

## 5) Gợi ý đọc code theo thứ tự

1. `src/bin/wisdom.rs`
2. `src/ucci.rs`
3. `src/mcts.rs`
4. `src/board.rs`
5. `src/bin/selfplay.rs`

Thứ tự này giúp nắm nhanh: cách engine chạy thực tế -> cách chọn nước -> cách sinh dữ liệu/train.
