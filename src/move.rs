/// A single move encoded in a `u16` unsigned integer.
/// 
/// Bit Layout (15 bits used):
/// 0-6   (7 bits):  `from` square (0-152) -> 128 is not enough if index goes up to 0x98 (152), so we need 8 bits!
/// Wait, let's recount.
/// Max square index is 152 (0x98). 8 bits can hold up to 255.
/// 
/// Corrected Layout (17 bits needed?? No).
/// Wait, `u16` has 16 bits.
/// 0-7  (8 bits): `from` square (0-255)
/// 8-14 (7 bits): `to` square (0-152) -> Wait, `to` square also needs 8 bits!
/// If both need 8 bits, that's 16 bits. Then we have no room for `is_capture` flag in a `u16`.
/// 
/// Let's use `u32` just to be safe and keep it simple without complex bit trickery.
/// Or, we only store the dense 0-89 index for the positions, which fits in 7 bits!
/// To convert a 256-array index to a dense 0-89 index: 
/// dense = (sq / 16) * 9 + (sq % 16).
/// dense fits in 7 bits (0-127).
/// 
/// Let's use the dense 7-bit index to keep `Move` as a `u16`!
/// Bits 0-6:   dense `from` index (0-89)
/// Bits 7-13:  dense `to` index (0-89)
/// Bit  14:    `is_capture` flag (0 or 1)

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Move(u16);

impl Move {
    /// Creates a new move from two 0x88 board indices and a capture flag.
    pub fn new(from_0x88: usize, to_0x88: usize, is_capture: bool) -> Self {
        let from_dense = Self::to_dense(from_0x88) as u16;
        let to_dense = Self::to_dense(to_0x88) as u16;
        let capture_bit = if is_capture { 1 << 14 } else { 0 };

        Self((to_dense << 7) | from_dense | capture_bit)
    }

    /// Converts a 0x88 index (0..255) to a dense index (0..89)
    fn to_dense(sq: usize) -> usize {
        (sq / 16) * 9 + (sq % 16)
    }

    /// Converts a dense index (0..89) back to a 0x88 index
    fn from_dense(dense: usize) -> usize {
        (dense / 9) * 16 + (dense % 9)
    }

    /// Gets the original 0x88 `from` square
    pub fn from_sq(&self) -> usize {
        let dense = (self.0 & 0x7F) as usize;
        Self::from_dense(dense)
    }

    /// Gets the original 0x88 `to` square
    pub fn to_sq(&self) -> usize {
        let dense = ((self.0 >> 7) & 0x7F) as usize;
        Self::from_dense(dense)
    }

    /// Returns whether this move is a capture
    pub fn is_capture(&self) -> bool {
        (self.0 & (1 << 14)) != 0
    }
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Move {{ from: {}, to: {}, capture: {} }}",
            self.from_sq(),
            self.to_sq(),
            self.is_capture()
        )
    }
}
