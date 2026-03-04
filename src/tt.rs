use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering, fence};

pub const TT_POLICY_CAPACITY: usize = 128;
const TT_BUCKET_SLOTS: usize = 2;

#[derive(Clone, Copy)]
pub struct TTNodeData {
    pub value: f32,
    pub policy_len: u16,
    pub policy: [(u16, f32); TT_POLICY_CAPACITY],
    pub visits: u32,
    pub age: u16,
}

impl Default for TTNodeData {
    fn default() -> Self {
        Self {
            value: 0.0,
            policy_len: 0,
            policy: [(0, 0.0); TT_POLICY_CAPACITY],
            visits: 0,
            age: 0,
        }
    }
}

impl TTNodeData {
    #[inline]
    pub fn from_sparse(value: f32, policy: &[(u16, f32)], visits: u32, age: u16) -> Self {
        let take = policy.len().min(TT_POLICY_CAPACITY);
        let mut packed = [(0u16, 0.0f32); TT_POLICY_CAPACITY];
        packed[..take].copy_from_slice(&policy[..take]);

        Self {
            value,
            policy_len: take as u16,
            policy: packed,
            visits,
            age,
        }
    }

    #[inline]
    pub fn policy_slice(&self) -> &[(u16, f32)] {
        &self.policy[..self.policy_len as usize]
    }
}

#[repr(align(64))]
pub struct TTEntry {
    key: AtomicU64,
    version: AtomicU64,
    data: UnsafeCell<TTNodeData>,
}

// SAFETY: Access to `data` is coordinated by sequence lock protocol (`version`).
unsafe impl Sync for TTEntry {}

impl TTEntry {
    #[inline]
    fn new() -> Self {
        Self {
            key: AtomicU64::new(0),
            version: AtomicU64::new(0),
            data: UnsafeCell::new(TTNodeData::default()),
        }
    }

    #[inline]
    fn load_key(&self) -> u64 {
        self.key.load(Ordering::Relaxed)
    }

    #[inline]
    fn probe_slot(&self, target_key: u64) -> Option<TTNodeData> {
        loop {
            let v1 = self.version.load(Ordering::Acquire);
            if v1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }

            let current_key = self.key.load(Ordering::Relaxed);
            if current_key != target_key {
                let v2 = self.version.load(Ordering::Acquire);
                if v1 == v2 {
                    return None;
                }
                continue;
            }

            // SAFETY: Use volatile read to prevent LLVM from assuming no concurrent writes
            let data_copy = unsafe { std::ptr::read_volatile(self.data.get()) };

            fence(Ordering::Acquire);
            let v2 = self.version.load(Ordering::Acquire);
            if v1 == v2 {
                return Some(data_copy);
            }
        }
    }

    #[inline]
    fn snapshot_meta(&self) -> (u64, u32, u16) {
        loop {
            let v1 = self.version.load(Ordering::Acquire);
            if v1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }

            let key = self.key.load(Ordering::Relaxed);
            // SAFETY: Use raw pointer access to avoid creating reference during concurrent writes
            let data_ptr = self.data.get();
            let visits = unsafe { std::ptr::addr_of!((*data_ptr).visits).read_volatile() };
            let age = unsafe { std::ptr::addr_of!((*data_ptr).age).read_volatile() };

            fence(Ordering::Acquire);
            let v2 = self.version.load(Ordering::Acquire);
            if v1 == v2 {
                return (key, visits, age);
            }
        }
    }

    #[inline]
    fn write_slot(&self, key: u64, new_data: TTNodeData) {
        self.version.fetch_add(1, Ordering::AcqRel);
        unsafe {
            *self.data.get() = new_data;
        }
        self.key.store(key, Ordering::Relaxed);
        self.version.fetch_add(1, Ordering::Release);
    }
}

pub struct TTBucket {
    slots: [TTEntry; TT_BUCKET_SLOTS],
}

impl TTBucket {
    #[inline]
    fn new() -> Self {
        Self {
            slots: [TTEntry::new(), TTEntry::new()],
        }
    }
}

pub struct TranspositionTable {
    buckets: Vec<TTBucket>,
    mask: usize,
    current_age: AtomicU16,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        let target_bytes = size_mb.saturating_mul(1024).saturating_mul(1024);
        let bucket_size = std::mem::size_of::<TTBucket>().max(1);
        let raw_buckets = (target_bytes / bucket_size).max(1);
        let mut num_buckets = raw_buckets.next_power_of_two();

        // FIX: Round down if rounding up exceeds target to prevent OOM
        if (num_buckets * bucket_size) > target_bytes && num_buckets > 1 {
            num_buckets /= 2;
        }

        let mut buckets = Vec::with_capacity(num_buckets);
        for _ in 0..num_buckets {
            buckets.push(TTBucket::new());
        }

        Self {
            buckets,
            mask: num_buckets - 1,
            current_age: AtomicU16::new(0),
        }
    }

    /// Increment and return the current age (generation) for this search
    #[inline]
    pub fn next_age(&self) -> u16 {
        self.current_age.fetch_add(1, Ordering::Relaxed)
    }

    #[inline]
    pub fn probe(&self, key: u64) -> Option<TTNodeData> {
        let index = (key as usize) & self.mask;
        let bucket = &self.buckets[index];

        if let Some(data) = bucket.slots[0].probe_slot(key) {
            return Some(data);
        }

        bucket.slots[1].probe_slot(key)
    }

    #[inline]
    pub fn record(&self, key: u64, value: f32, policy: &[(u16, f32)], current_age: u16) {
        let new_data = TTNodeData::from_sparse(value, policy, 1, current_age);
        self.record_with_meta(key, new_data, current_age);
    }

    #[inline]
    pub fn record_with_meta(&self, key: u64, mut new_data: TTNodeData, current_age: u16) {
        new_data.age = current_age;

        let index = (key as usize) & self.mask;
        let bucket = &self.buckets[index];

        if bucket.slots[0].load_key() == key {
            bucket.slots[0].write_slot(key, new_data);
            return;
        }

        if bucket.slots[1].load_key() == key {
            bucket.slots[1].write_slot(key, new_data);
            return;
        }

        let (_, slot0_visits, slot0_age) = bucket.slots[0].snapshot_meta();
        if slot0_age != current_age || new_data.visits > slot0_visits {
            bucket.slots[0].write_slot(key, new_data);
        } else {
            bucket.slots[1].write_slot(key, new_data);
        }
    }
}
