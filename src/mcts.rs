use crate::board::{Board, Color, HistoryEntry, PieceType, RepetitionResult};
use crate::eval_queue::EvalRequest;
use crate::r#move::Move;

use crate::tt::TranspositionTable;
use crossbeam_channel::Sender;
use rand_distr::{Distribution, multi::Dirichlet};
use std::sync::atomic::{AtomicI64, AtomicU32, Ordering};

// pub const C_PUCT: f32 = 1.5;
pub const VIRTUAL_LOSS: u32 = 1;

pub struct AtomicMCTSNode {
    pub visits: AtomicU32,      // N - Visit count with Virtual Loss
    pub total_value: AtomicI64, // W - Điểm số DƯỚI GÓC NHÌN CỦA NODE NÀY

    pub children_index: AtomicU32,
    pub num_children: AtomicU32, // u32::MAX nghĩa là Đang Lock (Đang Expansion)

    pub move_from_parent: AtomicU32,
    pub prior_prob: AtomicU32,
}

impl AtomicMCTSNode {
    pub fn new() -> Self {
        Self {
            visits: AtomicU32::new(0),
            total_value: AtomicI64::new(0),
            children_index: AtomicU32::new(0),
            num_children: AtomicU32::new(0),
            move_from_parent: AtomicU32::new(0),
            prior_prob: AtomicU32::new(0.0f32.to_bits()),
        }
    }

    pub fn set_data(&self, move_val: u16, prior_prob: f32) {
        self.move_from_parent
            .store(move_val as u32, Ordering::Release);
        self.prior_prob
            .store(prior_prob.to_bits(), Ordering::Release);
    }

    pub fn get_prior_prob(&self) -> f32 {
        f32::from_bits(self.prior_prob.load(Ordering::Acquire))
    }

    pub fn get_move(&self) -> u16 {
        self.move_from_parent.load(Ordering::Acquire) as u16
    }

    pub fn add_value(&self, val: f32) {
        let scaled = (val * 1_000_000.0) as i64;
        self.total_value.fetch_add(scaled, Ordering::AcqRel);
    }

    pub fn get_value(&self) -> f32 {
        let current = self.total_value.load(Ordering::Acquire);
        (current as f32) / 1_000_000.0
    }
}

pub struct MCTS {
    pub tree: Vec<AtomicMCTSNode>,
    pub next_node_idx: AtomicU32,
    pub max_nodes: usize,
}

pub struct SearchMetrics {
    pub root_visits: u32,
    pub best_child_visits: u32,
    pub win_pct: f32,
    pub eval: f32, // Q value in [-1, 1] from the OPPONENT's perspective (negate engine's Q)
    pub top_moves: Vec<(Move, u32, f32)>, // Move, visits, percentage
}

impl MCTS {
    pub fn new(max_nodes: usize) -> Self {
        let mut tree = Vec::with_capacity(max_nodes);
        for _ in 0..max_nodes {
            tree.push(AtomicMCTSNode::new());
        }

        Self {
            tree,
            next_node_idx: AtomicU32::new(1), // Node 0 is Root
            max_nodes,
        }
    }

    pub fn allocate_children(&self, num: u32) -> Option<u32> {
        let mut current = self.next_node_idx.load(Ordering::Relaxed);
        loop {
            if (current as usize) + (num as usize) > self.max_nodes {
                return None; // Out of memory
            }
            match self.next_node_idx.compare_exchange_weak(
                current,
                current + num,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(current),
                Err(val) => current = val,
            }
        }
    }

    pub fn search_best_move(
        &self,
        root_board: &Board,
        game_history: &[HistoryEntry],
        simulations: usize,
        eval_tx: &Sender<EvalRequest>,
        tt: &TranspositionTable,
        num_threads: usize,
        add_noise: bool,
    ) -> (Move, SearchMetrics) {
        // Reset cây về trạng thái ban đầu
        self.tree[0].visits.store(0, Ordering::Release);
        self.tree[0].total_value.store(0, Ordering::Release);
        self.tree[0].children_index.store(0, Ordering::Release);
        self.tree[0].num_children.store(0, Ordering::Release);
        self.tree[0].set_data(0, 1.0);
        self.next_node_idx.store(1, Ordering::SeqCst);
        let tt_age = tt.next_age();

        // Chỉ giữ phần lịch sử còn có thể ảnh hưởng đến repetition/chasing
        let mut last_irreversible_idx = 0;
        for i in (0..game_history.len()).rev() {
            if !game_history[i].is_reversible {
                last_irreversible_idx = i + 1;
                break;
            }
        }
        let relevant_history = &game_history[last_irreversible_idx..];

        // Mở rộng Nút gốc (Root Expansion)
        let mut pseudo_moves = root_board.generate_captures();
        pseudo_moves.append(&mut root_board.generate_quiets());
        let mut legal_moves = Vec::new();
        let current_side = root_board.side_to_move;

        for &m in &pseudo_moves {
            let mut test_board = root_board.clone();
            test_board.make_move(m);
            if !test_board.kings_facing() && !test_board.is_in_check(current_side) {
                legal_moves.push(m);
            }
        }

        // Đánh giá Root: Lấy mảng Nén (Sparse Policy)
        let p_sparse = if let Some(tt_data) = tt.probe(root_board.zobrist_key) {
            tt_data.policy_slice().to_vec()
        } else {
            let tensor = crate::nn::board_to_tensor(root_board);
            let (tx, rx) = crossbeam_channel::bounded(1);
            eval_tx
                .send(EvalRequest {
                    tensor_data: tensor,
                    response_tx: tx,
                    need_policy: true,
                })
                .unwrap();
            let (val, opt_p) = rx.recv().unwrap();
            let p_full = opt_p.unwrap();

            let sparse: Vec<(u16, f32)> = legal_moves
                .iter()
                .map(|m| {
                    // Sử dụng hàm tổng hợp để lấy index chính xác trong Action Space 8100
                    // Hàm này đã bao gồm: Lật tọa độ 0x88 -> Chuyển sang 90 -> Tính Index
                    let nn_idx =
                        crate::r#move::Move::move_to_nn_index(*m, current_side == Color::Black);

                    // Trả về tuple: (Mã nước đi gốc, xác suất tương ứng từ Policy Head)
                    (m.0, p_full[nn_idx])
                })
                .collect();

            tt.record(root_board.zobrist_key, val, &sparse, tt_age);
            sparse
        };

        if let Some(start_idx) = self.allocate_children(p_sparse.len() as u32) {
            self.tree[0]
                .children_index
                .store(start_idx, Ordering::Release);

            // --- THÊM SOFTMAX TRÊN MẢNG NÉN ---
            let mut max_logit = -f32::MAX;
            for &(_, logit) in &p_sparse {
                if logit > max_logit {
                    max_logit = logit;
                }
            }

            let mut sum_exp = 0.0;
            let mut exps = Vec::with_capacity(p_sparse.len());
            for &(_, logit) in &p_sparse {
                let exp = (logit - max_logit).exp();
                exps.push(exp);
                sum_exp += exp;
            }

            let mut priors: Vec<f32> = exps.into_iter().map(|exp| exp / sum_exp).collect();

            // Thêm Dirichlet Noise
            if add_noise && p_sparse.len() > 1 {
                let alpha = vec![0.3; p_sparse.len()];
                if let Ok(dirichlet) = Dirichlet::new(&alpha) {
                    let mut rng = rand::rng();
                    let noise: Vec<f64> = dirichlet.sample(&mut rng);
                    for i in 0..p_sparse.len() {
                        priors[i] = (1.0 - 0.25) * priors[i] + 0.25 * (noise[i] as f32);
                    }
                }
            }

            // Gán dữ liệu cho con
            for (i, &(move_val, _)) in p_sparse.iter().enumerate() {
                let idx = start_idx as usize + i;
                self.tree[idx].visits.store(0, Ordering::Release);
                self.tree[idx].total_value.store(0, Ordering::Release);
                self.tree[idx].children_index.store(0, Ordering::Release);
                self.tree[idx].num_children.store(0, Ordering::Release);
                self.tree[idx].set_data(move_val, priors[i]);
            }
            self.tree[0]
                .num_children
                .store(p_sparse.len() as u32, Ordering::Release);
        }

        if num_threads == 1 {
            let mut local_board = root_board.clone();
            for _ in 0..simulations {
                self.playout(
                    root_board,
                    &mut local_board,
                    eval_tx,
                    tt,
                    tt_age,
                    relevant_history,
                );
                local_board = root_board.clone();
            }
        } else {
            std::thread::scope(|s| {
                for _ in 0..num_threads {
                    s.spawn(|| {
                        let mut local_board = root_board.clone();
                        for _ in 0..(simulations / num_threads) {
                            self.playout(
                                root_board,
                                &mut local_board,
                                eval_tx,
                                tt,
                                tt_age,
                                relevant_history,
                            );
                            local_board = root_board.clone();
                        }
                    });
                }
            });
        }

        // Chọn nước đi có số lần duyệt (Visits) cao nhất
        let root_visits = self.tree[0].visits.load(Ordering::Acquire);
        let mut best_move = Move(0);
        let mut best_child_visits = 0;
        let mut best_child_q = 0.0f32;
        let start_idx = self.tree[0].children_index.load(Ordering::Acquire);
        let num_children = self.tree[0].num_children.load(Ordering::Acquire);

        let mut children_stats: Vec<(Move, u32)> = Vec::with_capacity(num_children as usize);

        for i in 0..num_children {
            let idx = start_idx as usize + i as usize;
            let node = &self.tree[idx];
            let v = node.visits.load(Ordering::Acquire);

            children_stats.push((Move(node.get_move()), v));

            if v > best_child_visits {
                best_child_visits = v;
                best_move = Move(node.get_move());
                if v > 0 {
                    // ĐÃ SỬA: Bỏ dấu '-' để in Win% đúng với phe của Engine
                    best_child_q = node.get_value() / v as f32;
                }
            }
        }

        children_stats.sort_by(|a, b| b.1.cmp(&a.1));

        let mut top_moves = Vec::new();
        let total_visits_f32 = std::cmp::max(1, root_visits) as f32;
        for &(mv, nv) in children_stats.iter().take(5) {
            if nv > 0 {
                let pct = (nv as f32 / total_visits_f32) * 100.0;
                top_moves.push((mv, nv, pct));
            }
        }

        let win_pct = ((best_child_q + 1.0) / 2.0 * 100.0).clamp(0.0, 100.0);
        // Negate Q so eval is from the OPPONENT's (player's) perspective
        let eval = (-best_child_q).clamp(-1.0, 1.0);

        let metrics = SearchMetrics {
            root_visits,
            best_child_visits,
            win_pct,
            eval,
            top_moves,
        };

        (best_move, metrics)
    }

    fn playout(
        &self,
        master_board: &Board,
        board: &mut Board,
        eval_tx: &Sender<EvalRequest>,
        tt: &TranspositionTable,
        tt_age: u16,
        relevant_history: &[HistoryEntry],
    ) {
        let mut path = Vec::with_capacity(64);
        let mut current_idx = 0;
        path.push(current_idx);

        // Track history for repetition detection during playout
        let mut local_history: Vec<HistoryEntry> = Vec::with_capacity(relevant_history.len() + 64);
        local_history.extend_from_slice(relevant_history);
        let seed_len = relevant_history.len();

        // 1. SELECT (Đi từ Root xuống Leaf)
        loop {
            let node = &self.tree[current_idx];

            // Phạt Ảo (Virtual Loss)
            let node_visits = node.visits.fetch_add(VIRTUAL_LOSS, Ordering::AcqRel);
            node.total_value.fetch_sub(1_000_000, Ordering::AcqRel);

            let num_children = node.num_children.load(Ordering::Acquire);

            // Nếu gặp Leaf Node hoặc Node đang bị khóa -> Dừng Select
            if num_children == 0 || num_children == u32::MAX {
                break;
            }

            let start_idx = node.children_index.load(Ordering::Acquire);
            let mut best_score = -1000000.0;
            let mut best_child = start_idx as usize;
            let mut best_move_int = 0;

            let parent_visits = std::cmp::max(1, node_visits) as f32;
            let sqrt_parent_visits = parent_visits.sqrt();

            // Tối ưu: Dùng hằng số tĩnh tĩnh nếu nghĩ ít, dùng công thức động nếu nghĩ sâu
            let c_puct = if parent_visits < 10000.0 {
                // Tiết kiệm CPU cho quá trình Self-play (400 sim)
                1.5
            } else {
                // Bung sức mạnh AlphaZero khi thi đấu / Inference
                let c_init = 1.25f32;
                let c_base = 19652.0f32;
                c_init + ((parent_visits + c_base + 1.0) / c_base).ln()
            };

            for i in 0..num_children {
                let child_idx = start_idx as usize + i as usize;
                let child = &self.tree[child_idx];
                let cv = child.visits.load(Ordering::Acquire) as f32;

                let mut q = 0.0;
                if cv > 0.0 {
                    let total_val = child.get_value();
                    // ĐÃ SỬA: Bỏ dấu trừ đi. Giá trị đã được đảo chiều đúng ở Backprop rồi!
                    q = total_val / cv;
                }

                let prior = child.get_prior_prob();
                let u = c_puct * prior * sqrt_parent_visits / (1.0 + cv);
                let score = q + u;

                if score > best_score {
                    best_score = score;
                    best_child = child_idx;
                    best_move_int = child.get_move();
                }
            }

            path.push(best_child);
            current_idx = best_child;

            let m = Move(best_move_int);
            let is_capture = board.piece_at(m.to_sq()).is_some();
            let piece_opt = board.piece_at(m.from_sq());
            let is_reversible_check = if let Some(p) = piece_opt {
                !is_capture
                    && (p.piece_type != PieceType::Pawn || {
                        let (from_row, _) = crate::board::Board::square_to_coord(m.from_sq());
                        let (to_row, _) = crate::board::Board::square_to_coord(m.to_sq());
                        from_row == to_row
                    })
            } else {
                false
            };

            board.make_move(m);

            local_history.push(HistoryEntry {
                hash: board.zobrist_key,
                is_check: false, // LAZY EVALUATION
                chased_set: 0,   // LAZY EVALUATION
                is_reversible: is_reversible_check,
            });
        }

        // 2. EXPAND & EVALUATE
        let leaf_node = &self.tree[current_idx];
        let value;

        // === REPETITION CHECK ===
        // Before evaluating with NN, check if current position is a repetition
        if local_history.len() >= 4 {
            let current_hash = local_history.last().unwrap().hash;
            let mut rep_count = 0;
            let mut loop_start_index = 0;

            let mut h = local_history.len() as isize - 3;
            while h >= 0 {
                let entry = &local_history[h as usize];
                if !entry.is_reversible {
                    break;
                }
                if entry.hash == current_hash {
                    rep_count += 1;
                    loop_start_index = h as usize;
                    if rep_count >= 1 {
                        break;
                    }
                }
                h -= 2;
            }

            if rep_count >= 1 {
                // LAZY EVALUATION TRIGGERED: Recalculate chased_set & is_check for the cycle
                let mut temp_board = master_board.clone();
                for step in 0..(path.len() - 1) {
                    let m = Move(self.tree[path[step + 1]].get_move());
                    let hist_idx = seed_len + step;

                    if hist_idx >= loop_start_index {
                        let moving_side = temp_board.side_to_move;
                        let pre_threats = temp_board.get_unprotected_threats(moving_side);
                        temp_board.make_move(m);

                        let gives_check = temp_board.is_in_check(temp_board.side_to_move);
                        let chased_set = if local_history[hist_idx].is_reversible && !gives_check {
                            let post_threats = temp_board.get_unprotected_threats(moving_side);
                            post_threats & !pre_threats
                        } else {
                            0
                        };

                        local_history[hist_idx].is_check = gives_check;
                        local_history[hist_idx].chased_set = chased_set;
                    } else {
                        temp_board.make_move(m);
                    }
                }

                match temp_board.judge_repetition(&local_history, local_history.len(), 1) {
                    RepetitionResult::Loss => {
                        value = -1.0;
                        let mut current_val_i64 = (-value * 1_000_000.0) as i64;
                        for &idx in path.iter().rev() {
                            let node = &self.tree[idx];
                            node.total_value
                                .fetch_add(1_000_000 + current_val_i64, Ordering::AcqRel);
                            current_val_i64 = -current_val_i64;
                        }
                        return;
                    }
                    RepetitionResult::Win => {
                        value = 1.0;
                        let mut current_val_i64 = (-value * 1_000_000.0) as i64;
                        for &idx in path.iter().rev() {
                            let node = &self.tree[idx];
                            node.total_value
                                .fetch_add(1_000_000 + current_val_i64, Ordering::AcqRel);
                            current_val_i64 = -current_val_i64;
                        }
                        return;
                    }
                    RepetitionResult::Draw => {
                        value = 0.0;
                        let mut current_val_i64 = (-value * 1_000_000.0) as i64;
                        for &idx in path.iter().rev() {
                            let node = &self.tree[idx];
                            node.total_value
                                .fetch_add(1_000_000 + current_val_i64, Ordering::AcqRel);
                            current_val_i64 = -current_val_i64;
                        }
                        return;
                    }
                    RepetitionResult::Undecided => {}
                }
            }
        }

        // FIX BUG 2: Lọc các nước cờ hợp lệ thật sự
        let mut pseudo_moves = board.generate_captures();
        pseudo_moves.append(&mut board.generate_quiets());
        let mut legal_moves = Vec::with_capacity(pseudo_moves.len());
        let current_side = board.side_to_move;

        for &m in &pseudo_moves {
            let undo = board.make_move(m);
            if !board.kings_facing() && !board.is_in_check(current_side) {
                legal_moves.push(m);
            }
            board.unmake_move(m, undo);
        }

        if legal_moves.is_empty() {
            // Hết cờ (Bị chiếu bí hoặc hết nước đi) -> Value = -1 (Thua)
            value = -1.0;
        } else {
            // FIX BUG 3: Dùng Spin-lock để ngăn đụng độ bộ nhớ đa luồng
            if leaf_node
                .num_children
                .compare_exchange(0, u32::MAX, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
            {
                let p_sparse;
                if let Some(tt_data) = tt.probe(board.zobrist_key) {
                    value = tt_data.value;
                    p_sparse = tt_data.policy_slice().to_vec();
                } else {
                    // Ta đã khóa được Node! Gọi GPU đánh giá
                    let tensor = crate::nn::board_to_tensor(board);
                    let (tx, rx) = crossbeam_channel::bounded(1);
                    eval_tx
                        .send(EvalRequest {
                            tensor_data: tensor,
                            response_tx: tx,
                            need_policy: true,
                        })
                        .unwrap();

                    let (v, opt_p) = rx.recv().unwrap();
                    value = v;
                    let p_full = opt_p.unwrap();

                    // Lọc và nén Policy bằng cách ánh xạ nước đi thực tế sang Action Space của Neural Network
                    p_sparse = legal_moves
                        .iter()
                        .map(|m| {
                            // 1 & 2 & 3: Gom tất cả logic lấy tọa độ, lật (nếu là Đen) và tính Index 8100 vào một chỗ
                            let nn_idx = crate::r#move::Move::move_to_nn_index(
                                *m,
                                current_side == Color::Black,
                            );

                            // Trả về tuple (mã nước đi gốc của engine, xác suất từ NN)
                            (m.0, p_full[nn_idx])
                        })
                        .collect();

                    tt.record(board.zobrist_key, value, &p_sparse, tt_age);
                }

                // Cấp phát con từ mảng p_sparse
                if let Some(start_idx) = self.allocate_children(p_sparse.len() as u32) {
                    leaf_node.children_index.store(start_idx, Ordering::Release);

                    let mut max_logit = -f32::MAX;
                    for &(_, logit) in &p_sparse {
                        if logit > max_logit {
                            max_logit = logit;
                        }
                    }

                    let mut sum_exp = 0.0;
                    let mut exps = Vec::with_capacity(p_sparse.len());
                    for &(_, logit) in &p_sparse {
                        let exp = (logit - max_logit).exp();
                        exps.push(exp);
                        sum_exp += exp;
                    }

                    for (i, &(move_val, _)) in p_sparse.iter().enumerate() {
                        let prob = exps[i] / sum_exp;
                        let idx = start_idx as usize + i;
                        self.tree[idx].set_data(move_val, prob);
                        self.tree[idx].visits.store(0, Ordering::Release);
                        self.tree[idx].total_value.store(0, Ordering::Release);
                        self.tree[idx].children_index.store(0, Ordering::Release);
                        self.tree[idx].num_children.store(0, Ordering::Release);
                    }
                    // Mở khóa
                    leaf_node
                        .num_children
                        .store(p_sparse.len() as u32, Ordering::Release);
                }
            } else {
                // Có một luồng khác đang mở rộng Node này. Ta Spin-wait.
                while leaf_node.num_children.load(Ordering::Acquire) == u32::MAX {
                    std::hint::spin_loop();
                }
                let cv = std::cmp::max(1, leaf_node.visits.load(Ordering::Acquire)) as f32;
                value = -leaf_node.get_value() / cv;
            }
        }

        // 3. BACKPROPAGATION
        let mut current_val_i64 = (-value * 1_000_000.0) as i64;

        for &idx in path.iter().rev() {
            let node = &self.tree[idx];
            node.total_value
                .fetch_add(1_000_000 + current_val_i64, Ordering::AcqRel);
            current_val_i64 = -current_val_i64;
        }
    }
}
