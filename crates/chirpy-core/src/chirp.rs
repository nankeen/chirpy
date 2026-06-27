use num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::TAU;

use crate::config::Config;

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
            phase.sin() * win
        })
        .collect()
}

/// FFT-accelerated linear cross-correlation. Returns
/// `corr[i] = Σⱼ signal[i+j] · template[j]` for `i ∈ [0, signal.len() - template.len()]`.
///
/// O((N+M) log(N+M)) — replaces O(N·M) sliding correlation.
pub fn fft_correlate(signal: &[f32], template: &[f32]) -> Vec<f32> {
    let n = signal.len();
    let m = template.len();
    if n < m {
        return Vec::new();
    }
    let valid_len = n - m + 1;
    let l = (n + m - 1).next_power_of_two();

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(l);
    let ifft = planner.plan_fft_inverse(l);

    let mut sig: Vec<Complex<f32>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    sig.resize(l, Complex::new(0.0, 0.0));
    let mut tpl: Vec<Complex<f32>> = template.iter().map(|&x| Complex::new(x, 0.0)).collect();
    tpl.resize(l, Complex::new(0.0, 0.0));

    fft.process(&mut sig);
    fft.process(&mut tpl);

    // Cross-correlation r[k] = Σ x[n] · y[n−k]  ↔  X(f) · conj(Y(f)).
    let mut prod: Vec<Complex<f32>> = sig
        .iter()
        .zip(tpl.iter())
        .map(|(s, t)| s * t.conj())
        .collect();
    ifft.process(&mut prod);

    let scale = 1.0 / l as f32;
    prod.iter().take(valid_len).map(|c| c.re * scale).collect()
}

/// Search `signal` for the chirp via FFT-accelerated cross-correlation.
/// Returns the integer-sample peak together with a sub-sample refinement.
///
/// `threshold_ratio` (0..1): peak must exceed this fraction of the chirp's
/// self-energy to be accepted.
pub fn detect_chirp(signal: &[f32], chirp: &[f32], threshold_ratio: f32) -> Option<ChirpHit> {
    let m = chirp.len();
    let n = signal.len();
    if n < m {
        return None;
    }

    let corr = fft_correlate(signal, chirp);
    if corr.is_empty() {
        return None;
    }

    // Locate the integer peak by absolute value (channel polarity may flip sign).
    let (best_idx, best_signed) = corr
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())?;
    let best_abs = best_signed.abs();

    let self_energy: f32 = chirp.iter().map(|x| x * x).sum();
    if best_abs < self_energy * threshold_ratio {
        return None;
    }

    // Parabolic refinement on |corr| in the 3-tap window around the integer peak.
    let sub_sample = if best_idx > 0 && best_idx + 1 < corr.len() {
        parabolic_offset(
            corr[best_idx - 1].abs(),
            corr[best_idx].abs(),
            corr[best_idx + 1].abs(),
        )
    } else {
        0.0
    };

    Some(ChirpHit {
        start: best_idx,
        sub_sample,
        peak_correlation: best_abs,
        self_energy,
    })
}

/// Fit y = a·x² + b·x + c through (−1, y_m1), (0, y_0), (+1, y_p1) and return
/// the x-coordinate of the parabola's vertex — the sub-sample peak offset.
/// Returns 0 if the three points are colinear (peak ill-defined).
fn parabolic_offset(y_m1: f32, y_0: f32, y_p1: f32) -> f32 {
    let den = y_m1 - 2.0 * y_0 + y_p1;
    if den.abs() < 1e-12 {
        return 0.0;
    }
    let delta = 0.5 * (y_m1 - y_p1) / den;
    // Clamp to the bin we already chose; numerical noise can push slightly past ±0.5.
    delta.clamp(-0.5, 0.5)
}

#[derive(Debug, Clone, Copy)]
pub struct ChirpHit {
    /// Integer-aligned index in the input signal where the chirp peak sits.
    pub start: usize,
    /// Sub-sample refinement of the peak position, in ~[−0.5, +0.5].
    /// True peak time ≈ `start + sub_sample`.
    pub sub_sample: f32,
    /// Absolute correlation value at the peak.
    pub peak_correlation: f32,
    /// Sum-of-squares of the chirp template (the autocorrelation peak).
    pub self_energy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference O(N·M) sliding implementation, used only in tests to validate FFT path.
    fn sliding_correlate(signal: &[f32], template: &[f32]) -> Vec<f32> {
        let n = signal.len();
        let m = template.len();
        if n < m {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(n - m + 1);
        for i in 0..=(n - m) {
            let mut acc = 0.0_f32;
            for j in 0..m {
                acc += signal[i + j] * template[j];
            }
            out.push(acc);
        }
        out
    }

    #[test]
    fn fft_matches_sliding_correlation() {
        // Mixed sinusoids: realistic-ish signal without being audio-coupled.
        let signal: Vec<f32> = (0..2048)
            .map(|i| (i as f32 * 0.07).sin() * 0.6 + (i as f32 * 0.013).cos() * 0.3)
            .collect();
        let template: Vec<f32> = (0..128).map(|i| (i as f32 * 0.21).sin()).collect();

        let a = fft_correlate(&signal, &template);
        let b = sliding_correlate(&signal, &template);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-3, "{x} vs {y}");
        }
    }

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
        // Integer-aligned chirp → fractional offset should be ~0.
        assert!(hit.sub_sample.abs() < 0.05, "got {}", hit.sub_sample);
    }

    #[test]
    fn rejects_pure_noise() {
        let cfg = Config::default();
        let chirp = make_chirp(&cfg);
        let signal: Vec<f32> = (0..(chirp.len() * 4))
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761) ^ 0xdeadbeef) as f32;
                (x / u32::MAX as f32) - 0.5
            })
            .collect();
        assert!(detect_chirp(&signal, &chirp, 0.8).is_none());
    }

    #[test]
    fn parabolic_offset_recovers_known_shift() {
        // Place the chirp at a fractional offset (linear-interp resampling) and
        // confirm sub_sample recovers it.
        let cfg = Config::default();
        let chirp = make_chirp(&cfg);
        let pre_int = 500;
        let true_frac = 0.3_f32; // shift the chirp 0.3 samples later

        // Build signal: zeros for pre_int, then a fractionally-delayed chirp.
        let mut signal = vec![0.0_f32; pre_int + chirp.len() + 16];
        for i in 0..chirp.len() {
            // True chirp sample at time t = i is placed at signal index
            // (pre_int + i + true_frac). Splat into two integer bins by linear interp.
            let target = pre_int as f32 + i as f32 + true_frac;
            let lo = target.floor() as usize;
            let f = target - lo as f32;
            signal[lo] += chirp[i] * (1.0 - f);
            signal[lo + 1] += chirp[i] * f;
        }

        let hit = detect_chirp(&signal, &chirp, 0.3).expect("chirp not detected");
        let recovered = hit.start as f32 + hit.sub_sample;
        let expected = pre_int as f32 + true_frac;
        assert!(
            (recovered - expected).abs() < 0.1,
            "expected {expected}, got {recovered} (start={}, sub={})",
            hit.start,
            hit.sub_sample
        );
    }
}
