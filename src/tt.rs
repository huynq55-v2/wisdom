use crate::r#move::Move;

pub const FLAG_EXACT: u8 = 0;
pub const FLAG_ALPHA: u8 = 1; // Upper bound
pub const FLAG_BETA: u8  = 2; // Lower bound

#[derive(Copy, Clone)]
pub struct TTEntry {
    pub key: u64,
    pub score: i32,
    pub depth: u8,
    pub flag: u8,
    pub best_move: Option<Move>,
}

pub struct TranspositionTable {
    pub entries: Vec<TTEntry>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<TTEntry>();
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        
        // Find next power of 2 for fast modulo using bitwise AND
        let mut power_of_2 = 1;
        while power_of_2 * 2 <= num_entries {
            power_of_2 *= 2;
        }
        
        Self {
            entries: vec![
                TTEntry {
                    key: 0,
                    score: 0,
                    depth: 0,
                    flag: FLAG_EXACT,
                    best_move: None,
                };
                power_of_2
            ],
            mask: power_of_2 - 1,
        }
    }

    pub fn probe(&self, key: u64, depth: u8, alpha: i32, beta: i32) -> Option<(i32, Option<Move>)> {
        let index = (key as usize) & self.mask;
        let entry = &self.entries[index];

        if entry.key == key {
            if entry.depth >= depth {
                match entry.flag {
                    FLAG_EXACT => return Some((entry.score, entry.best_move)),
                    FLAG_ALPHA => {
                        if entry.score <= alpha {
                            return Some((alpha, entry.best_move));
                        }
                    }
                    FLAG_BETA => {
                        if entry.score >= beta {
                            return Some((beta, entry.best_move));
                        }
                    }
                    _ => {}
                }
            }
            // Return Move for Move Ordering even if depth is insufficient or type is bound
            return Some((i32::MIN, entry.best_move));
        }

        None
    }

    pub fn record(&mut self, key: u64, depth: u8, score: i32, flag: u8, best_move: Option<Move>) {
        let index = (key as usize) & self.mask;
        // Simple replace scheme
        self.entries[index] = TTEntry {
            key,
            score,
            depth,
            flag,
            best_move,
        };
    }
}
