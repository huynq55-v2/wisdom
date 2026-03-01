use crate::r#move::Move;
use std::sync::atomic::{AtomicU64, Ordering};

pub const FLAG_EXACT: u8 = 0;
pub const FLAG_ALPHA: u8 = 1; // Upper bound
pub const FLAG_BETA: u8 = 2; // Lower bound

// Data packing layout for Atomic u64:
// [16 bits: Best Move] [8 bits: Flag] [8 bits: Depth] [16 bits: Unused] [16 bits: Score]
// We cast i32 score to i16 (sufficient since max is +/-20000).

pub struct TTEntryLayout {
    pub key: u64,
    pub score: i32,
    pub depth: u8,
    pub flag: u8,
    pub best_move: Option<Move>,
}

pub struct AtomicTTEntry {
    pub key: AtomicU64,
    pub data: AtomicU64,
}

impl Default for AtomicTTEntry {
    fn default() -> Self {
        Self {
            key: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }
}

pub struct TranspositionTable {
    pub entries: Vec<AtomicTTEntry>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<AtomicTTEntry>();
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

        // Read key, then data, then key again to ensure consistency
        let k1 = entry.key.load(Ordering::Relaxed);
        let data = entry.data.load(Ordering::Relaxed);
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
                    FLAG_EXACT => return Some((return_score, best_move)),
                    FLAG_ALPHA => {
                        if return_score <= alpha {
                            return Some((return_score, best_move));
                        }
                    }
                    FLAG_BETA => {
                        if return_score >= beta {
                            return Some((return_score, best_move));
                        }
                    }
                    _ => {}
                }
            }
            // Always return move for move ordering
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

        // Clip to i16 bounds
        let score16 = store_score.clamp(i16::MIN as i32, i16::MAX as i32) as u16 as u64;
        let depth8 = (depth as u64) << 32;
        let flag8 = (flag as u64) << 40;
        let mv16 = match best_move {
            Some(m) => (m.0 as u64) << 48,
            None => 0,
        };

        let data = score16 | depth8 | flag8 | mv16;

        let entry = &self.entries[index];
        // Note: Using Relaxed is typically fine for chess engine TT entries.
        entry.data.store(data, Ordering::Relaxed);
        entry.key.store(key, Ordering::Relaxed);
    }
}
