use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct TTNodeData {
    pub value: f32,
    pub policy: Vec<f32>,
}

pub struct TTEntry {
    pub key: AtomicU64,
    pub data: RwLock<Option<Arc<TTNodeData>>>,
}

pub struct TranspositionTable {
    pub entries: Vec<TTEntry>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        // Mỗi entry tốn khoảng 16-24 bytes (không tính phần Vec trong Heap)
        let entry_size = std::mem::size_of::<TTEntry>();
        let num_entries = (size_mb * 1024 * 1024) / entry_size;

        let power_of_2 = num_entries.next_power_of_two() / 2;
        let mut entries = Vec::with_capacity(power_of_2);
        for _ in 0..power_of_2 {
            entries.push(TTEntry {
                key: AtomicU64::new(0),
                data: RwLock::new(None),
            });
        }

        Self {
            entries,
            mask: power_of_2 - 1,
        }
    }

    /// Lấy kết quả từ Model đã lưu trước đó
    pub fn probe(&self, key: u64) -> Option<Arc<TTNodeData>> {
        let index = (key as usize) & self.mask;
        let entry = &self.entries[index];

        if entry.key.load(Ordering::Acquire) == key {
            let guard = entry.data.read();
            if entry.key.load(Ordering::Acquire) == key {
                return guard.as_ref().cloned();
            }
        }
        None
    }

    /// Lưu kết quả sau khi Model ResNet chạy xong
    pub fn record(&self, key: u64, value: f32, policy: Vec<f32>) {
        let index = (key as usize) & self.mask;
        let entry = &self.entries[index];

        let mut guard = entry.data.write();
        *guard = Some(Arc::new(TTNodeData { value, policy }));
        entry.key.store(key, Ordering::Release);
    }
}
