use crc::{Crc, CRC_32_ISO_HDLC};
use num_complex::Complex;
use thiserror::Error;

use crate::config::Modulation;

const CRC32: Crc<u32> = Crc::<u32>::new(&CRC_32_ISO_HDLC);

/// Fixed 16-byte / 128-bit training preamble.
///
/// Sits between the chirp and the sync word. Its job is to give the
/// decision-feedback equalizer a known symbol sequence to converge on:
/// recover the absolute phase from the first ~16 symbols, then LMS-train
/// the equalizer taps over the remaining ~112.
///
/// The pattern is balanced (≈64 ones, 64 zeros) and de-correlated against
/// itself by inspection — picked once and frozen so both sides agree.
pub const TRAINING_SEQUENCE: [u8; 16] = [
    0xB8, 0xA0, 0x1B, 0x59, 0xA9, 0xBE, 0x42, 0x73,
    0xC4, 0xF6, 0xD2, 0x9D, 0x37, 0x0E, 0x65, 0x8F,
];

/// 32-bit sync pattern. After the chirp + training preamble, this verifies
/// byte alignment and discards any equalizer that converged on noise.
pub const SYNC_WORD: [u8; 4] = [0x1A, 0xCF, 0xFC, 0x1D];

pub const MAX_PAYLOAD: usize = 65_535;

#[derive(Debug, Error)]
pub enum FrameError {
    #[error("payload too large: {0} bytes (max {})", MAX_PAYLOAD)]
    PayloadTooLarge(usize),
    #[error("frame too short to contain header")]
    TooShort,
    #[error("training preamble mismatch (equalizer likely failed to converge)")]
    BadTraining,
    #[error("sync word mismatch")]
    BadSync,
    #[error("unknown modulation byte: {0}")]
    UnknownModulation(u8),
    #[error("payload truncated: header says {expected} bytes, only {actual} available")]
    Truncated { expected: usize, actual: usize },
    #[error("crc mismatch: expected {expected:08x}, computed {actual:08x}")]
    BadCrc { expected: u32, actual: u32 },
}

/// Serialize frame bytes (everything BPSK-modulated; the chirp is a separate
/// waveform the TX prepends afterwards).
///
/// Layout:
///   [TRAINING: 16][SYNC: 4][LEN: u16 LE][MOD: u8][PAYLOAD: LEN][CRC32: u32 LE]
///
/// CRC covers LEN, MOD, and PAYLOAD — not TRAINING or SYNC, both of which are
/// fixed markers.
pub fn encode(payload: &[u8], modulation: Modulation) -> Result<Vec<u8>, FrameError> {
    if payload.len() > MAX_PAYLOAD {
        return Err(FrameError::PayloadTooLarge(payload.len()));
    }
    let len = payload.len() as u16;
    let cap = TRAINING_SEQUENCE.len() + SYNC_WORD.len() + 2 + 1 + payload.len() + 4;
    let mut buf = Vec::with_capacity(cap);
    buf.extend_from_slice(&TRAINING_SEQUENCE);
    buf.extend_from_slice(&SYNC_WORD);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.push(modulation.to_byte());
    buf.extend_from_slice(payload);

    let crc_input_start = TRAINING_SEQUENCE.len() + SYNC_WORD.len();
    let crc = CRC32.checksum(&buf[crc_input_start..]);
    buf.extend_from_slice(&crc.to_le_bytes());
    Ok(buf)
}

#[derive(Debug)]
pub struct DecodedFrame {
    pub modulation: Modulation,
    pub payload: Vec<u8>,
}

/// Parse a frame from `bytes`. Returns the decoded payload and how many bytes
/// of the input were consumed.
pub fn decode(bytes: &[u8]) -> Result<(DecodedFrame, usize), FrameError> {
    let min_len = TRAINING_SEQUENCE.len() + SYNC_WORD.len() + 2 + 1 + 4;
    if bytes.len() < min_len {
        return Err(FrameError::TooShort);
    }
    let training_end = TRAINING_SEQUENCE.len();
    if bytes[..training_end] != TRAINING_SEQUENCE {
        return Err(FrameError::BadTraining);
    }
    let sync_end = training_end + SYNC_WORD.len();
    if bytes[training_end..sync_end] != SYNC_WORD {
        return Err(FrameError::BadSync);
    }
    let len = u16::from_le_bytes([bytes[sync_end], bytes[sync_end + 1]]) as usize;
    let mod_byte_idx = sync_end + 2;
    let modulation =
        Modulation::from_byte(bytes[mod_byte_idx]).ok_or(FrameError::UnknownModulation(bytes[mod_byte_idx]))?;
    let payload_start = mod_byte_idx + 1;
    let payload_end = payload_start + len;
    let crc_end = payload_end + 4;
    if bytes.len() < crc_end {
        return Err(FrameError::Truncated {
            expected: crc_end,
            actual: bytes.len(),
        });
    }
    let payload = bytes[payload_start..payload_end].to_vec();
    let expected_crc = u32::from_le_bytes([
        bytes[payload_end],
        bytes[payload_end + 1],
        bytes[payload_end + 2],
        bytes[payload_end + 3],
    ]);
    let actual_crc = CRC32.checksum(&bytes[sync_end..payload_end]);
    if expected_crc != actual_crc {
        return Err(FrameError::BadCrc {
            expected: expected_crc,
            actual: actual_crc,
        });
    }
    Ok((DecodedFrame { modulation, payload }, crc_end))
}

/// Return the coherent BPSK symbols of the training preamble (±1 on the real
/// axis). The receiver uses these as known targets when adapting the equalizer.
pub fn training_symbols() -> Vec<Complex<f32>> {
    let bits = bytes_to_bits(&TRAINING_SEQUENCE);
    bits.iter()
        .map(|&b| Complex::new(if b == 0 { 1.0 } else { -1.0 }, 0.0))
        .collect()
}

/// Convert a byte slice into a flat Vec<u8> of bits (MSB-first per byte).
/// Values are 0 or 1.
pub fn bytes_to_bits(bytes: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(bytes.len() * 8);
    for &b in bytes {
        for i in (0..8).rev() {
            bits.push((b >> i) & 1);
        }
    }
    bits
}

/// Inverse of [`bytes_to_bits`]. Drops any trailing bits that don't fill a byte.
pub fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len() / 8);
    for chunk in bits.chunks_exact(8) {
        let mut b = 0u8;
        for &bit in chunk {
            b = (b << 1) | (bit & 1);
        }
        out.push(b);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty() {
        let frame = encode(&[], Modulation::Bpsk).unwrap();
        let (decoded, consumed) = decode(&frame).unwrap();
        assert_eq!(decoded.payload, Vec::<u8>::new());
        assert_eq!(consumed, frame.len());
    }

    #[test]
    fn roundtrip_payload() {
        let payload = b"hello world".to_vec();
        let frame = encode(&payload, Modulation::Bpsk).unwrap();
        let (decoded, _) = decode(&frame).unwrap();
        assert_eq!(decoded.payload, payload);
        assert_eq!(decoded.modulation, Modulation::Bpsk);
    }

    #[test]
    fn detects_crc_error() {
        let mut frame = encode(b"abcd", Modulation::Bpsk).unwrap();
        let last = frame.len() - 1;
        frame[last] ^= 0x01;
        assert!(matches!(decode(&frame), Err(FrameError::BadCrc { .. })));
    }

    #[test]
    fn detects_training_corruption() {
        let mut frame = encode(b"abcd", Modulation::Bpsk).unwrap();
        frame[3] ^= 0x80;
        assert!(matches!(decode(&frame), Err(FrameError::BadTraining)));
    }

    #[test]
    fn training_symbols_are_balanced() {
        let syms = training_symbols();
        assert_eq!(syms.len(), TRAINING_SEQUENCE.len() * 8);
        let sum: f32 = syms.iter().map(|s| s.re).sum();
        // Picked-by-eye balance; tolerate ±16 imbalance.
        assert!(sum.abs() <= 16.0, "imbalanced sum {sum}");
    }

    #[test]
    fn bit_conversion_roundtrips() {
        let data = b"\x00\xff\xa5\x5a\x01";
        let bits = bytes_to_bits(data);
        assert_eq!(bits.len(), 5 * 8);
        let back = bits_to_bytes(&bits);
        assert_eq!(back.as_slice(), data);
    }
}
