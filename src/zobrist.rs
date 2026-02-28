pub struct Zobrist {
    pub keys: [[u64; 256]; 14],
    pub side: u64,
}

const fn pcg32_random_r(state: &mut u64, inc: u64) -> u32 {
    let oldstate = *state;
    *state = oldstate.wrapping_mul(6364136223846793005).wrapping_add(inc | 1);
    let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
    let rot = (oldstate >> 59) as u32;
    (xorshifted >> rot) | (xorshifted.wrapping_shl((rot.wrapping_neg()) & 31))
}

const fn next_u64(state: &mut u64) -> u64 {
    let v1 = pcg32_random_r(state, 1442695040888963407) as u64;
    let v2 = pcg32_random_r(state, 1442695040888963407) as u64;
    (v1 << 32) | v2
}

pub const ZOBRIST: Zobrist = {
    let mut state = 1234567890123456789;
    let mut keys = [[0; 256]; 14];
    
    let mut i = 0;
    while i < 14 {
        let mut j = 0;
        while j < 256 {
            keys[i][j] = next_u64(&mut state);
            j += 1;
        }
        i += 1;
    }
    
    let side = next_u64(&mut state);
    
    Zobrist { keys, side }
};
