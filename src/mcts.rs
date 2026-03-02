use crate::board::{Board, HistoryEntry, PieceType, RepetitionResult};
use crate::eval_queue::EvalRequest;
use crate::r#move::Move;
use crate::nn::move_to_index;
use crate::tt::TranspositionTable;
use crossbeam_channel::Sender;
use rand_distr::{Dirichlet, Distribution};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub const C_PUCT: f32 = 1.5;
pub const VIRTUAL_LOSS: u32 = 1;

pub struct AtomicMCTSNode {
    pub visits: AtomicU32,      // N - Visit count with Virtual Loss
    pub total_value: AtomicU64, // W - Điểm số DƯỚI GÓC NHÌN CỦA NODE NÀY

    pub children_index: AtomicU32,
    pub num_children: AtomicU32, // u32::MAX nghĩa là Đang Lock (Đang Expansion)

    pub move_from_parent: AtomicU32,
    pub prior_prob: AtomicU32,
}

impl AtomicMCTSNode {
    pub fn new() -> Self {
        Self {
            visits: AtomicU32::new(0),
            total_value: AtomicU64::new(0),
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
        let mut current = self.total_value.load(Ordering::Relaxed);
        loop {
            let current_i64 = current as i64;
            let new_i64 = current_i64 + scaled;
            let new_u64 = new_i64 as u64;
            match self.total_value.compare_exchange_weak(
                current,
                new_u64,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(val) => current = val,
            }
        }
    }

    pub fn get_value(&self) -> f32 {
        let current = self.total_value.load(Ordering::Relaxed) as i64;
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
        simulations: usize,
        eval_tx: &Sender<EvalRequest>,
        _tt: &TranspositionTable,
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

        // Đánh giá Root: Thử lấy từ TT trước, nếu trượt mới gọi NN
        let (policy, _v) = if let Some(tt_data) = _tt.probe(root_board.zobrist_key) {
            (tt_data.policy.clone(), tt_data.value)
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
            let pol = opt_p.unwrap();

            // Lưu vào TT
            _tt.record(root_board.zobrist_key, val, pol.clone());
            (pol, val)
        };

        if let Some(start_idx) = self.allocate_children(legal_moves.len() as u32) {
            self.tree[0]
                .children_index
                .store(start_idx, Ordering::Release);

            // --- THÊM SOFTMAX OVER LEGAL MOVES ---
            let mut max_logit = -f32::MAX;
            for m in &legal_moves {
                let nn_idx = move_to_index(*m);
                if policy[nn_idx] > max_logit {
                    max_logit = policy[nn_idx];
                }
            }
            let mut sum_exp = 0.0;
            let mut exps = Vec::with_capacity(legal_moves.len());
            for m in &legal_moves {
                let nn_idx = move_to_index(*m);
                let exp = (policy[nn_idx] - max_logit).exp();
                exps.push(exp);
                sum_exp += exp;
            }
            // -------------------------------------

            // Gán xác suất chuẩn
            let mut priors: Vec<f32> = exps.into_iter().map(|exp| exp / sum_exp).collect();

            // --- THÊM DIRICHLET NOISE CHO SELF-PLAY ---
            // AlphaZero dùng alpha = 0.3 cho Cờ vua (nhánh ~30), Cờ tướng nhánh ~40 dùng 0.3 là đẹp.
            // Epsilon = 0.25 (25% tò mò, 75% nghe theo Mạng Neural)
            if add_noise && legal_moves.len() > 1 {
                let alpha = vec![0.3; legal_moves.len()];
                if let Ok(dirichlet) = Dirichlet::new(&alpha) {
                    let mut rng = rand::thread_rng();
                    let noise = dirichlet.sample(&mut rng);
                    let epsilon = 0.25;
                    for i in 0..legal_moves.len() {
                        priors[i] = (1.0 - epsilon) * priors[i] + epsilon * (noise[i] as f32);
                    }
                }
            }
            // ------------------------------------------

            for (i, m) in legal_moves.iter().enumerate() {
                let idx = start_idx as usize + i;
                let p = priors[i]; // Gán Priors đã được bơm nhiễu

                self.tree[idx].visits.store(0, Ordering::Release);
                self.tree[idx].total_value.store(0, Ordering::Release);
                self.tree[idx].children_index.store(0, Ordering::Release);
                self.tree[idx].num_children.store(0, Ordering::Release);
                self.tree[idx].set_data(m.0, p);
            }
            self.tree[0]
                .num_children
                .store(legal_moves.len() as u32, Ordering::Release);
        }

        // Khởi chạy Đa luồng (Tree Parallelism)
        std::thread::scope(|s| {
            for _ in 0..num_threads {
                s.spawn(|| {
                    let mut local_board = root_board.clone();
                    for _ in 0..(simulations / num_threads) {
                        self.playout(&mut local_board, eval_tx);
                        local_board = root_board.clone();
                    }
                });
            }
        });

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

        let metrics = SearchMetrics {
            root_visits,
            best_child_visits,
            win_pct,
            top_moves,
        };

        (best_move, metrics)
    }

    fn playout(&self, board: &mut Board, eval_tx: &Sender<EvalRequest>) {
        let mut path = Vec::with_capacity(64);
        let mut current_idx = 0;
        path.push(current_idx);

        // Track history for repetition detection during playout
        let mut local_history: Vec<HistoryEntry> = Vec::with_capacity(64);

        // 1. SELECT (Đi từ Root xuống Leaf)
        loop {
            let node = &self.tree[current_idx];

            // Phạt Ảo (Virtual Loss)
            let node_visits = node.visits.fetch_add(VIRTUAL_LOSS, Ordering::AcqRel);
            node.add_value(-1.0);

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
                let u = C_PUCT * prior * sqrt_parent_visits / (1.0 + cv);
                let score = q + u;

                if score > best_score {
                    best_score = score;
                    best_child = child_idx;
                    best_move_int = child.get_move();
                }
            }

            path.push(best_child);
            current_idx = best_child;

            // Track history: compute HistoryEntry for this move
            let m = Move(best_move_int);
            let moving_side = board.side_to_move;
            let is_capture = m.is_capture();
            let piece_opt = board.piece_at(m.from_sq());
            let is_reversible_check = if let Some(p) = piece_opt {
                !is_capture
                    && (p.piece_type != PieceType::Pawn || {
                        let (from_row, _) = Board::square_to_coord(m.from_sq());
                        let (to_row, _) = Board::square_to_coord(m.to_sq());
                        from_row == to_row
                    })
            } else {
                false
            };

            let pre_threats = if is_reversible_check {
                board.get_unprotected_threats(moving_side)
            } else {
                0
            };

            board.make_move(m);

            let gives_check = board.is_in_check(board.side_to_move);
            let chased_set = if is_reversible_check && !gives_check {
                let post_threats = board.get_unprotected_threats(moving_side);
                post_threats & !pre_threats
            } else {
                0
            };

            local_history.push(HistoryEntry {
                hash: board.zobrist_key,
                is_check: gives_check,
                chased_set,
                is_reversible: is_reversible_check,
            });
        }

        // 2. EXPAND & EVALUATE
        let leaf_node = &self.tree[current_idx];
        let value;

        // === REPETITION CHECK ===
        // Before evaluating with NN, check if current position is a repetition
        if local_history.len() >= 4 {
            match board.judge_repetition(&local_history, local_history.len(), 1) {
                RepetitionResult::Loss => {
                    // Current side to move is perpetually chasing/checking → LOSE
                    value = -1.0;
                    // Skip expansion, go straight to backprop
                    let mut current_val = -value;
                    for &idx in path.iter().rev() {
                        let node = &self.tree[idx];
                        node.add_value(1.0 + current_val);
                        current_val = -current_val;
                    }
                    return;
                }
                RepetitionResult::Win => {
                    value = 1.0;
                    let mut current_val = -value;
                    for &idx in path.iter().rev() {
                        let node = &self.tree[idx];
                        node.add_value(1.0 + current_val);
                        current_val = -current_val;
                    }
                    return;
                }
                RepetitionResult::Draw => {
                    value = 0.0;
                    let mut current_val = -value;
                    for &idx in path.iter().rev() {
                        let node = &self.tree[idx];
                        node.add_value(1.0 + current_val);
                        current_val = -current_val;
                    }
                    return;
                }
                RepetitionResult::Undecided => {
                    // Continue with normal evaluation
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
                let p = opt_p.unwrap();

                // Cấp phát con
                if let Some(start_idx) = self.allocate_children(legal_moves.len() as u32) {
                    leaf_node.children_index.store(start_idx, Ordering::Release);

                    // --- THÊM SOFTMAX OVER LEGAL MOVES ---
                    let mut max_logit = -f32::MAX;
                    for m in &legal_moves {
                        let nn_idx = move_to_index(*m);
                        if p[nn_idx] > max_logit {
                            max_logit = p[nn_idx];
                        }
                    }
                    let mut sum_exp = 0.0;
                    let mut exps = Vec::with_capacity(legal_moves.len());
                    for m in &legal_moves {
                        let nn_idx = move_to_index(*m);
                        let exp = (p[nn_idx] - max_logit).exp();
                        exps.push(exp);
                        sum_exp += exp;
                    }
                    // -------------------------------------

                    for (i, m) in legal_moves.iter().enumerate() {
                        let idx = start_idx as usize + i;
                        let prob = exps[i] / sum_exp; // Gán Xác suất chuẩn [0, 1]

                        self.tree[idx].set_data(m.0, prob);
                        self.tree[idx].visits.store(0, Ordering::Release);
                        self.tree[idx].total_value.store(0, Ordering::Release);
                        self.tree[idx].children_index.store(0, Ordering::Release);
                        self.tree[idx].num_children.store(0, Ordering::Release);
                    }
                    // Mở khóa
                    leaf_node
                        .num_children
                        .store(legal_moves.len() as u32, Ordering::Release);
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
        let mut current_val = -value;

        for &idx in path.iter().rev() {
            let node = &self.tree[idx];
            node.add_value(1.0 + current_val);
            current_val = -current_val;
        }
    }
}
