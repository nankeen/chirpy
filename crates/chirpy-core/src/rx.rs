use num_complex::Complex;
use thiserror::Error;

use crate::{
    bpsk, carrier, chirp,
    equalizer::{estimate_channel, Dfe, DEFAULT_FBF_LEN, DEFAULT_FFF_LEN, DEFAULT_MU, PHASE_EST_LEN},
    frame, pulse,
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

#[derive(Debug, Clone, Default)]
pub struct DecodeTrace {
    /// Best chirp matched-filter peak / chirp self-energy.
    pub chirp_peak_ratio: f32,
    /// Sub-sample chirp peak refinement, ~[-0.5, +0.5].
    pub chirp_sub_sample: f32,
    /// Integer symbol-phase offset chosen by the timing search.
    pub best_phase_offset: i32,
    /// Post-equalizer symbols — what the constellation should plot.
    /// On training failure (no chirp lock) this is empty.
    pub symbols: Vec<Complex<f32>>,
    /// Pre-equalizer symbols (post-RRC, pre-DFE). Useful for "before / after"
    /// comparison when debugging the equalizer.
    pub pre_eq_symbols: Vec<Complex<f32>>,
}

/// Decode `samples` back to the original payload bytes.
pub fn decode_samples(samples: &[f32], cfg: &Config) -> Result<Vec<u8>, DecodeError> {
    decode_samples_traced(samples, cfg).0
}

/// Same as [`decode_samples`] but also returns a [`DecodeTrace`] populated as
/// far as the attempt got. The trace's `symbols` field holds the post-DFE
/// constellation; `pre_eq_symbols` holds the input to the DFE.
pub fn decode_samples_traced(
    samples: &[f32],
    cfg: &Config,
) -> (Result<Vec<u8>, DecodeError>, DecodeTrace) {
    let mut trace = DecodeTrace::default();
    let preamble = chirp::make_chirp(cfg);

    let raw_hit = chirp::detect_chirp(samples, &preamble, 0.0);
    if let Some(h) = raw_hit {
        trace.chirp_peak_ratio = if h.self_energy > 0.0 {
            h.peak_correlation / h.self_energy
        } else {
            0.0
        };
        trace.chirp_sub_sample = h.sub_sample;
    }
    let hit = match raw_hit {
        Some(h) if trace.chirp_peak_ratio >= cfg.chirp_detect_threshold => h,
        _ => return (Err(DecodeError::ChirpNotFound), trace),
    };

    let sps = cfg.sps();
    let data_start_f = hit.start as f32
        + hit.sub_sample
        + preamble.len() as f32
        + (sps * POST_CHIRP_GAP_SYMBOLS) as f32;
    let data_start = data_start_f.floor() as usize;
    let data_frac = data_start_f - data_start as f32;
    if data_start >= samples.len() {
        return (Err(DecodeError::Truncated), trace);
    }
    let data_passband = &samples[data_start..];

    let baseband_mixed =
        carrier::downconvert(data_passband, cfg.carrier_hz, cfg.sample_rate as f32);
    let taps = pulse::rrc_taps(sps, cfg.rrc_span_symbols, cfg.rrc_beta);
    let filtered = pulse::convolve(&baseband_mixed, &taps);

    let filter_delay = sps * cfg.rrc_span_symbols;
    let nominal_symbols = if filtered.len() > filter_delay + 1 {
        (filtered.len() - filter_delay - 1) / sps
    } else {
        0
    };

    // Brute-force integer phase offset search (covers ±sps/2).
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
    trace.best_phase_offset = best_offset;

    let base_idx = filter_delay as i32 + best_offset;
    let mut raw_symbols = Vec::with_capacity(nominal_symbols);
    for k in 0..nominal_symbols {
        let idx_f = base_idx as f32 + (k * sps) as f32 + data_frac;
        if let Some(s) = sample_linear(&filtered, idx_f) {
            raw_symbols.push(s);
        } else {
            break;
        }
    }
    trace.pre_eq_symbols = raw_symbols.clone();

    // Joint phase + gain estimate from the head of the training preamble.
    let training_targets = frame::training_symbols();
    let n_training = training_targets.len();
    let head = raw_symbols.len().min(PHASE_EST_LEN).min(n_training);
    let channel_est = estimate_channel(&raw_symbols[..head], &training_targets[..head]);
    let normalized: Vec<Complex<f32>> = if channel_est.norm_sqr() < 1e-12 {
        raw_symbols.clone()
    } else {
        raw_symbols.iter().map(|s| s / channel_est).collect()
    };

    // Equalize.
    let mut dfe = Dfe::new(DEFAULT_FFF_LEN, DEFAULT_FBF_LEN, DEFAULT_MU);
    let equalized = dfe.process(&normalized, &training_targets, n_training);
    trace.symbols = equalized.clone();

    let bits = bpsk::demodulate(&equalized);
    let bytes = frame::bits_to_bytes(&bits);
    let result = frame::decode(&bytes)
        .map(|(decoded, _consumed)| decoded.payload)
        .map_err(DecodeError::from);
    (result, trace)
}

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

    #[test]
    fn traced_decode_matches_plain_on_success() {
        let cfg = Config::default();
        let payload = b"trace consistency".to_vec();
        let samples = encode_samples(&payload, &cfg).unwrap();
        let plain = decode_samples(&samples, &cfg).unwrap();
        let (traced, trace) = decode_samples_traced(&samples, &cfg);
        assert_eq!(traced.unwrap(), plain);
        assert!(trace.chirp_peak_ratio >= cfg.chirp_detect_threshold);
        assert!(!trace.symbols.is_empty());
        assert_eq!(trace.symbols.len(), trace.pre_eq_symbols.len());
    }

    #[test]
    fn trace_populated_on_noise() {
        let cfg = Config::default();
        let noise: Vec<f32> = (0..96_000)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761) ^ 0xfeedface) as f32;
                (x / u32::MAX as f32) - 0.5
            })
            .collect();
        let (result, trace) = decode_samples_traced(&noise, &cfg);
        assert!(matches!(result, Err(DecodeError::ChirpNotFound)));
        assert!(trace.chirp_peak_ratio < cfg.chirp_detect_threshold);
        assert!(trace.symbols.is_empty());
    }
}
