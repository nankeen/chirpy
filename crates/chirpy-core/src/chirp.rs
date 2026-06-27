use crate::config::Config;
use std::f32::consts::{PI, TAU};

/// Generate a Hann-windowed linear FM chirp from f0 to f1 over chirp_dur_ms.
/// The Hann window suppresses spectral sidelobes for a cleaner matched-filter peak.
pub fn make_chirp(cfg: &Config) -> Vec<f32> {
    let n = cfg.chirp_len_samples();
    let fs = cfg.sample_rate as f32;
    let f0 = cfg.chirp_f0_hz;
    let f1 = cfg.chirp_f1_hz;
    let t_total = n as f32 / fs;
    let k = (f1 - f0) / t_total;
    (0..n)
        .map(|i| {
            let t = i as f32 / fs;
            let phase = TAU * (f0 * t + 0.5 * k * t * t);
            let win = 0.5 * (1.0 - (TAU * i as f32 / (n - 1) as f32).cos());
            // Hann: 0.5*(1 - cos(2π i / (N-1)))
            let win = win.max(0.0);
            let _ = PI; // silence unused import warning if Hann form changes
            phase.sin() * win
        })
        .collect()
}

/// Search `signal` for the chirp via sliding cross-correlation.
/// Returns the index in `signal` where the chirp begins, along with the peak
/// correlation magnitude relative to the chirp's autocorrelation peak.
///
/// `threshold_ratio` (0..1) — peak must exceed this fraction of the chirp's
/// self-energy to be accepted. 0.3 is a reasonable default.
pub fn detect_chirp(signal: &[f32], chirp: &[f32], threshold_ratio: f32) -> Option<ChirpHit> {
    let m = chirp.len();
    let n = signal.len();
    if n < m {
        return None;
    }
    let self_energy: f32 = chirp.iter().map(|x| x * x).sum();
    let abs_threshold = self_energy * threshold_ratio;

    let mut best_idx = 0usize;
    let mut best_val = 0.0_f32;
    for i in 0..=(n - m) {
        let mut acc = 0.0_f32;
        for j in 0..m {
            acc += signal[i + j] * chirp[j];
        }
        if acc.abs() > best_val {
            best_val = acc.abs();
            best_idx = i;
        }
    }

    if best_val < abs_threshold {
        None
    } else {
        Some(ChirpHit {
            start: best_idx,
            peak_correlation: best_val,
            self_energy,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ChirpHit {
    /// Index in the input signal where the chirp begins.
    pub start: usize,
    /// Absolute correlation value at the peak.
    pub peak_correlation: f32,
    /// Sum of squares of the chirp template — the autocorrelation peak.
    pub self_energy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_chirp_in_padded_signal() {
        let cfg = Config::default();
        let chirp = make_chirp(&cfg);
        let pre = vec![0.0_f32; 1234];
        let post = vec![0.0_f32; 567];
        let mut signal = Vec::new();
        signal.extend_from_slice(&pre);
        signal.extend_from_slice(&chirp);
        signal.extend_from_slice(&post);

        let hit = detect_chirp(&signal, &chirp, 0.5).expect("chirp not detected");
        assert_eq!(hit.start, pre.len());
    }

    #[test]
    fn rejects_pure_noise() {
        let cfg = Config::default();
        let chirp = make_chirp(&cfg);
        // Deterministic pseudo-noise.
        let signal: Vec<f32> = (0..(chirp.len() * 4))
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761) ^ 0xdeadbeef) as f32;
                (x / u32::MAX as f32) - 0.5
            })
            .collect();
        // Energy of this noise is << chirp self-energy, so a high threshold rejects.
        assert!(detect_chirp(&signal, &chirp, 0.8).is_none());
    }
}
