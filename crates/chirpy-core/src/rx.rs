use thiserror::Error;

use crate::{
    bpsk, carrier, chirp, frame, pulse,
    tx::POST_CHIRP_GAP_SYMBOLS,
    Config, FrameError,
};

#[derive(Error, Debug)]
pub enum DecodeError {
    #[error("chirp preamble not found")]
    ChirpNotFound,
    #[error("signal truncated after chirp")]
    Truncated,
    #[error("frame error: {0}")]
    Frame(#[from] FrameError),
}

/// Detection threshold for the chirp matched filter, as a fraction of the
/// chirp's self-energy. 0.25 trades a bit of false-accept tolerance against
/// being too permissive on silent-room noise.
const CHIRP_DETECT_THRESHOLD: f32 = 0.25;

/// Decode `samples` back to the original payload bytes.
///
/// Strategy:
/// 1. Cross-correlate against the known chirp to find the preamble.
/// 2. Skip the chirp and inter-burst gap to land on the data passband.
/// 3. Downconvert to complex baseband.
/// 4. RRC matched-filter the baseband.
/// 5. Sample once per symbol, offset by the total TX+RX filter delay.
/// 6. DBPSK-demodulate, pack bits to bytes, decode the frame.
pub fn decode_samples(samples: &[f32], cfg: &Config) -> Result<Vec<u8>, DecodeError> {
    let preamble = chirp::make_chirp(cfg);
    let hit = chirp::detect_chirp(samples, &preamble, CHIRP_DETECT_THRESHOLD)
        .ok_or(DecodeError::ChirpNotFound)?;

    let sps = cfg.sps();
    let data_start = hit.start + preamble.len() + sps * POST_CHIRP_GAP_SYMBOLS;
    if data_start >= samples.len() {
        return Err(DecodeError::Truncated);
    }
    let data_passband = &samples[data_start..];

    let baseband_mixed = carrier::downconvert(data_passband, cfg.carrier_hz, cfg.sample_rate as f32);
    let taps = pulse::rrc_taps(sps, cfg.rrc_span_symbols, cfg.rrc_beta);
    let filtered = pulse::convolve(&baseband_mixed, &taps);

    // Symbol k's matched-filter peak sits at: TX_delay + RX_delay = 2 * (N-1)/2 = N - 1
    // beyond the start of the symbol stream, where N = taps.len() = sps * span + 1.
    // Then every sps samples for each subsequent symbol.
    let filter_delay = sps * cfg.rrc_span_symbols;
    let max_symbols = if filtered.len() > filter_delay {
        (filtered.len() - filter_delay) / sps + 1
    } else {
        0
    };

    let mut symbols = Vec::with_capacity(max_symbols);
    for k in 0..max_symbols {
        let idx = filter_delay + k * sps;
        if idx >= filtered.len() {
            break;
        }
        symbols.push(filtered[idx]);
    }

    let bits = bpsk::demodulate(&symbols);
    let bytes = frame::bits_to_bytes(&bits);
    let (decoded, _consumed) = frame::decode(&bytes)?;
    Ok(decoded.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tx::encode_samples;

    fn roundtrip(payload: &[u8]) {
        let cfg = Config::default();
        let samples = encode_samples(payload, &cfg).unwrap();
        let out = decode_samples(&samples, &cfg).expect("decode failed");
        assert_eq!(out.as_slice(), payload);
    }

    #[test]
    fn roundtrip_tiny() {
        roundtrip(b"hi");
    }

    #[test]
    fn roundtrip_medium() {
        let payload: Vec<u8> = (0..200).map(|i| (i as u8).wrapping_mul(37)).collect();
        roundtrip(&payload);
    }

    #[test]
    fn roundtrip_empty() {
        roundtrip(b"");
    }

    #[test]
    fn roundtrip_with_leading_silence() {
        let cfg = Config::default();
        let payload = b"hello chirpy".to_vec();
        let modulated = encode_samples(&payload, &cfg).unwrap();
        let mut padded = vec![0.0_f32; 12_345];
        padded.extend_from_slice(&modulated);
        let out = decode_samples(&padded, &cfg).expect("decode failed");
        assert_eq!(out, payload);
    }
}
