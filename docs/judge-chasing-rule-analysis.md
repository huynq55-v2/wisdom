# Phân tích lưu trữ lịch sử moves cho Judge Chasing Rule

## 1) Dữ liệu lịch sử mà thuật toán `judge_repetition` cần

`judge_repetition` không chỉ cần trạng thái bàn cờ hiện tại, mà cần **ngữ cảnh theo thời gian** của các nước đi gần nhất. Dữ liệu này nằm trong `HistoryEntry`:

- `hash`: Zobrist key của vị trí sau nước đi.
- `is_check`: nước đi vừa thực hiện có tạo chiếu tướng hay không.
- `chased_set`: tập quân bị truy đuổi (bitset `u128`), tính từ chênh lệch “đe doạ không được bảo vệ” trước/sau nước đi.
- `is_reversible`: nước đi có thể hoàn nguyên theo nghĩa dùng cho vòng lặp (không bắt quân, và không phải tốt đi thẳng).

Vị trí code: `src/board.rs` (định nghĩa `HistoryEntry` và logic judge).

---

## 2) `judge_repetition` đang đánh giá chasing/check như thế nào

Quy trình trong `Board::judge_repetition`:

1. Lùi ngược theo bước 2 ply để tìm cùng `hash` cùng phía đi.
2. Dừng quét nếu gặp entry `!is_reversible`.
3. Khi đủ ngưỡng lặp (`rep_threshold`), xét đoạn vòng lặp:
   - Bên mình: có phải **toàn check** hoặc **toàn chase** không.
   - Bên đối thủ: tương tự.
4. So mức vi phạm:
   - `PerpetualCheck` > `PerpetualChase` > `PerpetualIdle`
   - Cao hơn thì thua, bằng nhau thì hòa.

Vì vậy, nếu thiếu `is_check/chased_set/is_reversible` theo từng ply, thuật toán sẽ không phán đúng chasing rule.

---

## 3) Lịch sử moves hiện được lưu ở đâu

### A. GUI (`src/bin/gui.rs`)

GUI có hàm `apply_move_to_game(...)` tính đầy đủ `HistoryEntry` mỗi khi có nước đi thực:

- Tính `is_reversible`
- Nếu reversible: lấy `pre_threats`
- `make_move`
- Tính `gives_check`
- Nếu cần: `chased_set = post_threats & !pre_threats`
- `push HistoryEntry`

Sau đó gọi `board.judge_repetition(&game_history, game_history.len(), 2)` để xử lý luật lặp/chasing trong trận GUI.

=> Luồng GUI đang **đúng hướng** và có đủ dữ liệu cho judge.

### B. Self-play (`src/bin/selfplay.rs`)

Trong `play_game(...)`, sau khi chọn nước đi từ MCTS, code cũng tạo `HistoryEntry` với logic tương tự GUI và `push` vào `history`.

=> Self-play cũng có ngữ cảnh lịch sử đầy đủ để gọi `judge_repetition`.

### C. MCTS (`src/mcts.rs`)

Trong `playout(...)`, MCTS dùng chiến lược **lazy history**:

- Khi đi xuống cây, chỉ đẩy entry tạm (`is_check=false`, `chased_set=0`, có `hash`, `is_reversible`).
- Lý do: nếu tính `is_check` + `chased_set` đầy đủ cho mọi playout thì rất tốn CPU (MCTS gọi playout hàng trăm/hàng nghìn lần).

#### C.1) “MCTS phát hiện khả năng lặp” nghĩa là gì?

Trong code, sau khi có `local_history.len() >= 4`, MCTS kiểm tra:

1. Lấy `current_hash = local_history.last().hash`.
2. Quét ngược theo bước 2 ply (`h -= 2`) để tìm hash trùng phía đi.
3. Dừng nếu gặp nước `!is_reversible`.
4. Nếu tìm được hash trùng (`rep_count >= 1`) thì coi là **khả năng có chu kỳ** cần xác minh luật check/chase.

Nói ngắn gọn: đây là bước phát hiện “có thể đang lặp vị trí theo cùng side-to-move”, chưa kết luận vi phạm ngay.

##### Vì sao `local_history.len() >= 4`?

- `judge_repetition` trong `board.rs` cũng chặn sớm với điều kiện `history.len() < 4` thì `Undecided`.
- Về mặt thực thi, đây là một **guard bảo thủ** để không kết luận quá sớm khi mới có rất ít ply.
- Với cách cài đặt hiện tại, ngưỡng 4 giúp giảm false positive ở các dao động ngắn đầu chuỗi.

Lưu ý: về mặt lý thuyết, lặp vị trí có thể xuất hiện sớm hơn, nhưng engine này chủ động chỉ bắt đầu xét từ mốc 4 ply.

##### Vì sao quét ngược 2 ply (`h -= 2`)?

Vì history xen kẽ theo lượt đi:

- `idx = 0`: sau nước 1 (bên A đi)
- `idx = 1`: sau nước 2 (bên B đi)
- `idx = 2`: sau nước 3 (bên A đi)
- ...

Muốn kiểm tra “lặp vị trí cùng bên đến lượt”, phải so với các trạng thái cách **2 ply** (cùng parity index). Nếu quét từng 1 ply sẽ trộn lẫn trạng thái của bên còn lại, không đúng tiêu chí lặp cho judge.

#### C.2) “Replay đoạn chu kỳ bằng temp_board” là gì?

Do `local_history` ban đầu chỉ có dữ liệu tạm (`is_check=false`, `chased_set=0`), nên khi nghi lặp MCTS sẽ:

1. Clone lại board gốc của playout: `temp_board = master_board.clone()`.
2. Chạy lại lần lượt các move trên path playout từ đầu đến cuối.
3. Chỉ với đoạn từ `loop_start_index` trở đi (đoạn chu kỳ nghi ngờ), mới tính chính xác:
   - `pre_threats`
   - `gives_check`
   - `chased_set = post_threats & !pre_threats`
4. Ghi ngược các giá trị thật vào `local_history[i]`.
5. Gọi `judge_repetition(&local_history, ...)` để ra kết quả Win/Loss/Draw/Undecided.

=> Đây là cơ chế “tính lười”: chỉ trả giá tính toán khi có dấu hiệu lặp.

#### C.3) Vì sao vừa Self-play vừa MCTS đều gọi `judge_repetition`?

Hai lần gọi phục vụ **hai mục tiêu khác nhau**:

1. **Self-play level** (`src/bin/selfplay.rs`):
   - Check ở ván thật đang được sinh dữ liệu.
   - Dùng để quyết định dừng game và xác định kết quả cuối cùng (thắng/thua/hòa) cho dataset.
   - Lịch sử dùng ở đây là lịch sử thật của cả ván (`history` tích lũy qua các nước đã đánh).

2. **MCTS playout level** (`src/mcts.rs`):
   - Check trong từng đường mô phỏng nội bộ của search.
   - Dùng để gán value terminal sớm cho playout và backprop vào cây, tránh search “lao” vào vòng lặp vi phạm.
   - Lịch sử ở đây là `local_history` của path mô phỏng hiện tại.

#### C.4) Khác biệt quan trọng giữa 2 nơi gọi

- **Phạm vi lịch sử**:
  - Self-play: lịch sử ván thực tế (toàn cục).
  - MCTS: lịch sử cục bộ của path mô phỏng trong 1 playout.
- **Thời điểm gọi**:
  - Self-play: gọi theo từng nước thật trước/sau lượt tìm nước.
  - MCTS: gọi rất nhiều lần bên trong playout.
- **Mục đích**:
  - Self-play: quyết định kết quả game và dữ liệu huấn luyện.
  - MCTS: định giá node/planning trong tìm kiếm.
- **Độ đầy đủ dữ liệu ban đầu**:
  - Self-play: `HistoryEntry` được tính thật ngay khi đánh nước.
  - MCTS: ban đầu là dữ liệu tạm, chỉ khi nghi lặp mới replay để tính chính xác.

=> Tóm lại, hai lần gọi không dư thừa; chúng bổ sung nhau ở 2 tầng khác nhau: **game outcome** và **search quality**.

### D. UCCI position apply moves (`src/ucci.rs`)

Trong lệnh `position ... moves ...`, code hiện chỉ `board.make_move(m)` liên tiếp, **không lưu `HistoryEntry`**.

=> Engine có bàn cờ đúng, nhưng thiếu lịch sử ply-level để phán chasing/repetition theo ngữ cảnh ván thực tế trước đó.

---

## 4) Trả lời câu hỏi: có cần bổ sung history trong `ucci.rs` không?

### Kết luận ngắn

- **Nên bổ sung**, nếu muốn engine tự xử lý đúng chasing rule trong mọi ngữ cảnh UCCI.
- Nếu bạn chấp nhận phụ thuộc hoàn toàn vào GUI/arbiter bên ngoài để xử luật, thì có thể tạm không bắt buộc.

### Lý do

`position` nhận danh sách nước đi quá khứ, nhưng engine đang bỏ mất metadata cần cho `judge_repetition`. Chỉ từ board hiện tại thì không đủ suy ra ai đang perpetual check/chase trong chu kỳ gần nhất.

---

## 5) Đề xuất triển khai (mức thiết kế)

1. Tạo `game_history: Vec<HistoryEntry>` trong vòng lặp UCCI.
2. Khi nhận `position`:
   - reset board + clear history
   - apply từng move bằng helper kiểu `apply_move_with_history(...)` (tương tự GUI/selfplay)
3. Truyền history vào search (MCTS) để seed cho kiểm tra repetition/chasing ở root và trong playout.
4. (Tuỳ chọn) kiểm tra repetition ngay sau `position` hoặc trước `go` để trả kết quả nhất quán.

---

## 6) `rep_threshold` là gì?

`rep_threshold` là ngưỡng số lần gặp lại hash (cùng parity lượt đi) để engine coi là đã đủ điều kiện đánh giá chu kỳ.

Điểm quan trọng: trong code hiện tại, `rep_count` chỉ đếm **các lần trùng trước đó trong history**, tức là **không tính vị trí hiện tại**.

- Trong `judge_repetition`, nếu `rep_count < rep_threshold` thì trả `Undecided`.
- Suy ra:
  - `rep_threshold = 1`: cần 1 lần trùng trước đó → vị trí hiện tại xuất hiện tổng cộng 2 lần.
  - `rep_threshold = 2`: cần 2 lần trùng trước đó → vị trí hiện tại xuất hiện tổng cộng 3 lần (thường tương ứng “lặp 3 lần”).

### 6.1) Vậy GUI để `rep_threshold = 2` có bị xử ngay khi lặp 2 lần không?

Không. Với cách đếm hiện tại, GUI chỉ xử khi đủ **2 lần trùng trong quá khứ + 1 lần hiện tại**.

Nói cách khác:

- Lần quay lại thứ 2 của cùng vị trí: thường mới có `rep_count = 1` → chưa xử.
- Lần quay lại thứ 3 của cùng vị trí: `rep_count = 2` → bắt đầu đủ điều kiện để judge Win/Loss/Draw.

Ngưỡng này ảnh hưởng trực tiếp tới thời điểm engine tuyên bố Win/Loss/Draw do repetition/chasing.

---

## 7) Có bug logic tiềm tàng nào trong self-play và MCTS không?

### A. Điểm quan trọng nhất: history của ván thật chưa được seed vào MCTS

Hiện tại MCTS playout tạo `local_history` từ rỗng cho mỗi search, chỉ chứa các nước mô phỏng kể từ root hiện tại. Nó **không nhận** `history` thật của ván đã diễn ra trước đó.

Hệ quả:

- Self-play level có thể phát hiện repetition/chasing trên lịch sử toàn ván.
- Nhưng MCTS playout level có thể bỏ sót bối cảnh chu kỳ bắt đầu trước root hiện tại.

=> Đây là khoảng lệch logic giữa tầng game và tầng search. Không phải bug crash, nhưng có thể làm search đánh giá thiếu chính xác các nhánh liên quan chu kỳ dài.

### B. `move_count` truyền vào `judge_repetition` trong self-play

Self-play truyền `current_ply = move_count` trước khi đánh nước mới. Trong khi `history.len()` cũng đang bằng số ply đã chơi, nên hiện tại vẫn nhất quán parity.

=> Không phải bug ngay lúc này, nhưng là điểm dễ sai khi refactor sau này. Dùng `history.len()` trực tiếp sẽ an toàn hơn về ý nghĩa.

### C. Khác biệt ngưỡng giữa các mode

Hiện tại các mode dùng `rep_threshold` khác nhau (GUI dùng `2`, selfplay/MCTS có chỗ dùng `1`). Đây có thể là chủ đích tuning, nhưng nếu không cố ý thì sẽ tạo khác biệt hành vi giữa GUI, self-play và engine UCCI.
