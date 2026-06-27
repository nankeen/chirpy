use crc::{Crc, CRC_32_ISO_HDLC};
use thiserror::Error;

use crate::config::Modulation;

const CRC32: Crc<u32> = Crc::<u32>::new(&CRC_32_ISO_HDLC);

/// 32-bit sync pattern. After the chirp gets us close, this gives a known
/// byte-aligned bit pattern at the start of the DBPSK-encoded data so the
/// receiver can verify alignment and discard false chirp triggers cheaply.
pub const SYNC_WORD: [u8; 4] = [0x1A, 0xCF, 0xFC, 0x1D];

pub const MAX_PAYLOAD: usize = 65_535;

#[derive(Debug, Error)]
pub enum FrameError {
    #[error("payload too large: {0} bytes (max {})", MAX_PAYLOAD)]
    PayloadTooLarge(usize),
    #[error("frame too short to contain header")]
    TooShort,
    #[error("sync word mismatch")]
    BadSync,
    #[error("unknown modulation byte: {0}")]
    UnknownModulation(u8),
    #[error("payload truncated: header says {expected} bytes, only {actual} available")]
    Truncated { expected: usize, actual: usize },
    #[error("crc mismatch: expected {expected:08x}, computed {actual:08x}")]
    BadCrc { expected: u32, actual: u32 },
}

/// Serialize frame bytes (everything that gets DBPSK-modulated; the chirp
/// is a separate waveform prepended later).
///
/// Layout: [SYNC: 4][LEN: u16 LE][MOD: u8][PAYLOAD: LEN bytes][CRC32: u32 LE]
/// The CRC covers LEN, MOD, and PAYLOAD (not SYNC, since SYNC is a fixed marker).
pub fn encode(payload: &[u8], modulation: Modulation) -> Result<Vec<u8>, FrameError> {
    if payload.len() > MAX_PAYLOAD {
        return Err(FrameError::PayloadTooLarge(payload.len()));
    }
    let len = payload.len() as u16;
    let mut buf = Vec::with_capacity(4 + 2 + 1 + payload.len() + 4);
    buf.extend_from_slice(&SYNC_WORD);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.push(modulation.to_byte());
    buf.extend_from_slice(payload);

    let crc_input_start = 4; // skip SYNC
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
/// of the input were consumed (so a stream consumer can advance).
pub fn decode(bytes: &[u8]) -> Result<(DecodedFrame, usize), FrameError> {
    if bytes.len() < 4 + 2 + 1 + 4 {
        return Err(FrameError::TooShort);
    }
    if bytes[..4] != SYNC_WORD {
        return Err(FrameError::BadSync);
    }
    let len = u16::from_le_bytes([bytes[4], bytes[5]]) as usize;
    let modulation = Modulation::from_byte(bytes[6]).ok_or(FrameError::UnknownModulation(bytes[6]))?;
    let payload_start = 7;
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
    let actual_crc = CRC32.checksum(&bytes[4..payload_end]);
    if expected_crc != actual_crc {
        return Err(FrameError::BadCrc {
            expected: expected_crc,
            actual: actual_crc,
        });
    }
    Ok((
        DecodedFrame { modulation, payload },
        crc_end,
    ))
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
    fn bit_conversion_roundtrips() {
        let data = b"\x00\xff\xa5\x5a\x01";
        let bits = bytes_to_bits(data);
        assert_eq!(bits.len(), 5 * 8);
        let back = bits_to_bytes(&bits);
        assert_eq!(back.as_slice(), data);
    }
}
