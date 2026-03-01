use crate::board::Board;
use crate::eval_queue::EvalRequest;
use crate::r#move::Move;
use crate::nn::move_to_index;
use crate::tt::TranspositionTable;
use crossbeam_channel::Sender;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub const C_PUCT: f32 = 1.5;
pub const VIRTUAL_LOSS: u32 = 1;

pub struct AtomicMCTSNode {
    pub visits: AtomicU32,      // N - Visit count with Virtual Loss
    pub total_value: AtomicU64, // W - Scaled by 1_000_000 for atomic integer ops

    pub children_index: AtomicU32, // Pointer to children array start
    pub num_children: AtomicU32,

    pub move_from_parent: AtomicU32, // Move integer casted to u32
    pub prior_prob: AtomicU32,       // f32 bits
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
        tt: &TranspositionTable,
        num_threads: usize,
    ) -> Move {
        // Init Root Node
        self.tree[0].visits.store(0, Ordering::Release);
        self.tree[0].total_value.store(0, Ordering::Release);
        self.tree[0].children_index.store(0, Ordering::Release);
        self.tree[0].num_children.store(0, Ordering::Release);
        self.tree[0].set_data(0, 1.0);
        self.next_node_idx.store(1, Ordering::SeqCst);

        // Expand root manually
        let mut legal_moves = root_board.generate_captures();
        legal_moves.append(&mut root_board.generate_quiets());

        let root_p;
        if let Some(data) = tt.probe_mcts(root_board.zobrist_key) {
            root_p = data.policy.clone();
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
            let (v, opt_p) = rx.recv().unwrap();
            let p = opt_p.unwrap();
            tt.record_mcts(root_board.zobrist_key, v, p.clone());
            root_p = p;
        }

        let policy = root_p;

        if let Some(start_idx) = self.allocate_children(legal_moves.len() as u32) {
            self.tree[0]
                .children_index
                .store(start_idx, Ordering::Release);
            for (i, m) in legal_moves.iter().enumerate() {
                let idx = start_idx as usize + i;
                let nn_idx = move_to_index(*m);
                let p = policy[nn_idx];
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

        // Multithreaded Search Loop
        std::thread::scope(|s| {
            for _ in 0..num_threads {
                s.spawn(|| {
                    let mut local_board = root_board.clone();
                    for _ in 0..(simulations / num_threads) {
                        self.playout(&mut local_board, eval_tx, tt);
                        local_board = root_board.clone();
                    }
                });
            }
        });

        // Pick best move based on visits
        let mut best_move = Move(0);
        let mut max_visits = 0;
        let start_idx = self.tree[0].children_index.load(Ordering::Acquire);
        let num_children = self.tree[0].num_children.load(Ordering::Acquire);

        for i in 0..num_children {
            let idx = start_idx as usize + i as usize;
            let node = &self.tree[idx];
            let v = node.visits.load(Ordering::Acquire);
            if v > max_visits {
                max_visits = v;
                best_move = Move(node.get_move());
            }
        }

        best_move
    }

    fn playout(&self, board: &mut Board, eval_tx: &Sender<EvalRequest>, tt: &TranspositionTable) {
        let mut path = Vec::with_capacity(64);
        let mut current_idx = 0;
        path.push(current_idx);

        // 1. SELECT
        loop {
            let node = &self.tree[current_idx];
            let node_visits = node.visits.fetch_add(VIRTUAL_LOSS, Ordering::AcqRel);
            // Apply Virtual Loss to Value as well
            node.add_value(-1.0);

            let num_children = node.num_children.load(Ordering::Acquire);
            if num_children == 0 {
                break; // Reached leaf
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
                    q = total_val / cv; // Mean action value
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
            board.make_move(Move(best_move_int));
        }

        // 2. EXPAND & EVALUATE
        let leaf_node = &self.tree[current_idx];

        let mut value = 0.0;
        let mut is_game_over = false;

        let moving_side = board.side_to_move;
        if board.kings_facing() || board.is_in_check(moving_side) {
            value = -1.0;
            is_game_over = true;
        }

        if !is_game_over {
            let mut legal_moves = board.generate_captures();
            legal_moves.append(&mut board.generate_quiets());

            if legal_moves.is_empty() {
                value = -1.0;
            } else {
                let p: Vec<f32>;
                if let Some(data) = tt.probe_mcts(board.zobrist_key) {
                    value = data.value;
                    p = data.policy.clone();
                } else {
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
                    p = opt_p.unwrap();
                    tt.record_mcts(board.zobrist_key, value, p.clone());
                }

                if let Some(start_idx) = self.allocate_children(legal_moves.len() as u32) {
                    leaf_node.children_index.store(start_idx, Ordering::Release);
                    for (i, m) in legal_moves.iter().enumerate() {
                        let idx = start_idx as usize + i;
                        let nn_idx = move_to_index(*m);
                        let prior = p[nn_idx];

                        self.tree[idx].set_data(m.0, prior);
                        self.tree[idx].visits.store(0, Ordering::Release);
                        self.tree[idx].total_value.store(0, Ordering::Release);
                        self.tree[idx].children_index.store(0, Ordering::Release);
                        self.tree[idx].num_children.store(0, Ordering::Release);
                    }
                    leaf_node
                        .num_children
                        .store(legal_moves.len() as u32, Ordering::Release);
                }
            }
        }

        // 3. BACKPROPAGATION
        let mut current_val = value;
        for &idx in path.iter().rev() {
            let node = &self.tree[idx];

            // Remove Virtual Loss (+1) and Add Real Value
            node.add_value(1.0 + current_val);

            current_val = -current_val; // Alternate turn games
        }
    }
}
