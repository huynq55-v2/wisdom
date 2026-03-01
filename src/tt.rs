use crate::r#move::Move;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub const FLAG_EXACT: u8 = 0;
pub const FLAG_ALPHA: u8 = 1; // Upper bound
pub const FLAG_BETA: u8 = 2; // Lower bound

pub struct TTNodeData {
    pub value: f32,
    pub policy: Vec<f32>,
}

pub struct AtomicTTEntry {
    pub key: AtomicU64,
    // Alpha-beta data
    pub alpha_beta_data: AtomicU64,
    // MCTS data
    pub mcts_data: std::sync::Mutex<Option<Arc<TTNodeData>>>,
}

impl Default for AtomicTTEntry {
    fn default() -> Self {
        Self {
            key: AtomicU64::new(0),
            alpha_beta_data: AtomicU64::new(0),
            mcts_data: std::sync::Mutex::new(None),
        }
    }
}

pub struct TranspositionTable {
    pub entries: Vec<AtomicTTEntry>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        // Approximate size: key (8) + Mutex (varies, ~32 bytes). Let's say 40 bytes.
        let entry_size = 40;
        let num_entries = (size_mb * 1024 * 1024) / entry_size;

        let mut power_of_2 = 1;
        while power_of_2 * 2 <= num_entries {
            power_of_2 *= 2;
        }

        let mut entries = Vec::with_capacity(power_of_2);
        for _ in 0..power_of_2 {
            entries.push(AtomicTTEntry::default());
        }

        Self {
            entries,
            mask: power_of_2 - 1,
        }
    }

    pub fn probe_mcts(&self, key: u64) -> Option<Arc<TTNodeData>> {
        let index = (key as usize) & self.mask;
        let entry = &self.entries[index];

        let k1 = entry.key.load(Ordering::Relaxed);
        if k1 == key {
            if let Ok(guard) = entry.mcts_data.lock() {
                if let Some(ref data) = *guard {
                    return Some(data.clone());
                }
            }
        }
        None
    }

    pub fn record_mcts(&self, key: u64, value: f32, policy: Vec<f32>) {
        let index = (key as usize) & self.mask;
        let entry = &self.entries[index];

        if let Ok(mut guard) = entry.mcts_data.lock() {
            *guard = Some(Arc::new(TTNodeData { value, policy }));
            entry.key.store(key, Ordering::Relaxed);
        }
    }

    // --- Legacy Alpha Beta Functions to avoid breaking search.rs ---
    pub fn probe(
        &self,
        key: u64,
        depth: u8,
        ply: u8,
        alpha: i32,
        beta: i32,
    ) -> Option<(i32, Option<Move>)> {
        let index = (key as usize) & self.mask;
        let entry = &self.entries[index];

        let k1 = entry.key.load(Ordering::Relaxed);
        let data = entry.alpha_beta_data.load(Ordering::Relaxed);
        let k2 = entry.key.load(Ordering::Relaxed);

        if k1 == key && k1 == k2 {
            let score16 = (data & 0xFFFF) as i16 as i32;
            let depth8 = ((data >> 32) & 0xFF) as u8;
            let flag8 = ((data >> 40) & 0xFF) as u8;
            let mv16 = ((data >> 48) & 0xFFFF) as u16;

            let best_move = if mv16 != 0 { Some(Move(mv16)) } else { None };

            if depth8 >= depth {
                let mut return_score = score16;
                if return_score > crate::search::MATE_VALUE - 100 {
                    return_score -= ply as i32;
                } else if return_score < -crate::search::MATE_VALUE + 100 {
                    return_score += ply as i32;
                }

                match flag8 {
                    crate::tt::FLAG_EXACT => return Some((return_score, best_move)),
                    crate::tt::FLAG_ALPHA => {
                        if return_score <= alpha {
                            return Some((return_score, best_move));
                        }
                    }
                    crate::tt::FLAG_BETA => {
                        if return_score >= beta {
                            return Some((return_score, best_move));
                        }
                    }
                    _ => {}
                }
            }
            return Some((i32::MIN, best_move));
        }
        None
    }

    pub fn record(
        &self,
        key: u64,
        depth: u8,
        ply: u8,
        score: i32,
        flag: u8,
        best_move: Option<Move>,
    ) {
        let index = (key as usize) & self.mask;

        let mut store_score = score;
        if store_score > crate::search::MATE_VALUE - 100 {
            store_score += ply as i32;
        } else if store_score < -crate::search::MATE_VALUE + 100 {
            store_score -= ply as i32;
        }

        let score16 = store_score.clamp(i16::MIN as i32, i16::MAX as i32) as u16 as u64;
        let depth8 = (depth as u64) << 32;
        let flag8 = (flag as u64) << 40;
        let mv16 = match best_move {
            Some(m) => (m.0 as u64) << 48,
            None => 0,
        };

        let data = score16 | depth8 | flag8 | mv16;

        let entry = &self.entries[index];
        entry.alpha_beta_data.store(data, Ordering::Relaxed);
        entry.key.store(key, Ordering::Relaxed);
    }
}
