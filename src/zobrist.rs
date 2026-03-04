pub struct Zobrist {
    // [Color][Piece type][Square 0-89]
    pub pieces: [[[u64; 90]; 7]; 2],
    pub side: u64,
}

const fn pcg32(state: &mut u64) -> u64 {
    let oldstate = *state;
    *state = oldstate
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
    let rot = (oldstate >> 59) as u32;
    let v1 = (xorshifted >> rot) | (xorshifted << ((rot.wrapping_neg()) & 31));

    // Run one more round to assemble a full 64-bit value.
    let oldstate = *state;
    *state = oldstate
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
    let rot = (oldstate >> 59) as u32;
    let v2 = (xorshifted >> rot) | (xorshifted << ((rot.wrapping_neg()) & 31));

    ((v1 as u64) << 32) | (v2 as u64)
}

pub const ZOBRIST: Zobrist = {
    let mut state = 1234567890123456789u64;
    let mut pieces = [[[0u64; 90]; 7]; 2];

    let mut color = 0;
    while color < 2 {
        let mut p_type = 0;
        while p_type < 7 {
            let mut sq = 0;
            while sq < 90 {
                pieces[color][p_type][sq] = pcg32(&mut state);
                sq += 1;
            }
            p_type += 1;
        }
        color += 1;
    }

    let side = pcg32(&mut state);
    Zobrist { pieces, side }
};
