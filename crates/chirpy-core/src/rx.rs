use num_complex::Complex;
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
/// chirp's self-energy.
const CHIRP_DETECT_THRESHOLD: f32 = 0.25;

/// Decode `samples` back to the original payload bytes.
///
/// 1. FFT-based matched filter against the chirp preamble; parabolic peak refinement
///    gives sub-sample timing.
/// 2. Downconvert the data passband to complex baseband; RRC matched-filter.
/// 3. Brute-force the integer symbol-phase offset ±sps/2, choosing the one that
///    maximizes total symbol energy. Combined with the chirp's fractional offset,
///    this locks symbol timing through linear interpolation.
/// 4. DBPSK-demodulate, pack bits, frame-decode.
pub fn decode_samples(samples: &[f32], cfg: &Config) -> Result<Vec<u8>, DecodeError> {
    let preamble = chirp::make_chirp(cfg);
    let hit = chirp::detect_chirp(samples, &preamble, CHIRP_DETECT_THRESHOLD)
        .ok_or(DecodeError::ChirpNotFound)?;

    let sps = cfg.sps();

    // Fractional time at which the data passband begins.
    let data_start_f = hit.start as f32
        + hit.sub_sample
        + preamble.len() as f32
        + (sps * POST_CHIRP_GAP_SYMBOLS) as f32;
    let data_start = data_start_f.floor() as usize;
    let data_frac = data_start_f - data_start as f32;

    if data_start >= samples.len() {
        return Err(DecodeError::Truncated);
    }
    let data_passband = &samples[data_start..];

    let baseband_mixed =
        carrier::downconvert(data_passband, cfg.carrier_hz, cfg.sample_rate as f32);
    let taps = pulse::rrc_taps(sps, cfg.rrc_span_symbols, cfg.rrc_beta);
    let filtered = pulse::convolve(&baseband_mixed, &taps);

    // First symbol's matched-filter peak (no offsets): filter_delay samples in.
    let filter_delay = sps * cfg.rrc_span_symbols;

    // How many symbol slots fit in the filtered buffer at zero offset.
    let nominal_symbols = if filtered.len() > filter_delay + 1 {
        (filtered.len() - filter_delay - 1) / sps
    } else {
        0
    };

    // Brute-force the integer symbol-phase offset. Searching ±sps/2 covers any
    // channel-induced delay error up to half a symbol either way.
    let search = (sps / 2) as i32;
    let mut best_offset = 0i32;
    let mut best_metric = f32::MIN;
    for offset in -search..=search {
        let metric = symbol_energy_metric(
            &filtered,
            filter_delay as i32 + offset,
            data_frac,
            sps,
            nominal_symbols,
        );
        if metric > best_metric {
            best_metric = metric;
            best_offset = offset;
        }
    }

    let base_idx = filter_delay as i32 + best_offset;
    let mut symbols = Vec::with_capacity(nominal_symbols);
    for k in 0..nominal_symbols {
        let idx_f = base_idx as f32 + (k * sps) as f32 + data_frac;
        if let Some(s) = sample_linear(&filtered, idx_f) {
            symbols.push(s);
        } else {
            break;
        }
    }

    let bits = bpsk::demodulate(&symbols);
    let bytes = frame::bits_to_bytes(&bits);
    let (decoded, _consumed) = frame::decode(&bytes)?;
    Ok(decoded.payload)
}

/// Sum of |filtered|² at the proposed symbol sampling instants, with linear
/// interpolation at fractional positions. Used to pick the best phase offset.
fn symbol_energy_metric(
    filtered: &[Complex<f32>],
    base_idx: i32,
    frac: f32,
    sps: usize,
    n_symbols: usize,
) -> f32 {
    let mut sum = 0.0_f32;
    for k in 0..n_symbols {
        let idx_f = base_idx as f32 + (k * sps) as f32 + frac;
        if let Some(s) = sample_linear(filtered, idx_f) {
            sum += s.norm_sqr();
        }
    }
    sum
}

/// Linearly interpolate `buf` at the fractional index `idx_f`. Returns `None`
/// if `idx_f` is outside the valid range.
fn sample_linear(buf: &[Complex<f32>], idx_f: f32) -> Option<Complex<f32>> {
    if idx_f < 0.0 {
        return None;
    }
    let i = idx_f.floor() as usize;
    let f = idx_f - i as f32;
    if i + 1 >= buf.len() {
        return None;
    }
    Some(buf[i] * (1.0 - f) + buf[i + 1] * f)
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
