use num_complex::Complex;
use std::f32::consts::{PI, SQRT_2};

/// Root-raised-cosine filter taps. Length = sps * span + 1.
/// Taps are normalized so that ‖h‖₂ = 1; pairing two of these (matched filter)
/// gives unit peak gain at the symbol instant.
pub fn rrc_taps(sps: usize, span: usize, beta: f32) -> Vec<f32> {
    let n_taps = sps * span + 1;
    let center = (n_taps - 1) as f32 / 2.0;
    let mut taps = vec![0.0_f32; n_taps];
    for (i, tap) in taps.iter_mut().enumerate() {
        let t = (i as f32 - center) / sps as f32;
        *tap = rrc_impulse(t, beta);
    }
    let norm = taps.iter().map(|x| x * x).sum::<f32>().sqrt();
    for tap in &mut taps {
        *tap /= norm;
    }
    taps
}

fn rrc_impulse(t: f32, beta: f32) -> f32 {
    let eps = 1e-6_f32;
    if t.abs() < eps {
        return 1.0 - beta + 4.0 * beta / PI;
    }
    let t_special = 1.0 / (4.0 * beta);
    if (t.abs() - t_special).abs() < eps {
        let k = PI / (4.0 * beta);
        return (beta / SQRT_2) * ((1.0 + 2.0 / PI) * k.sin() + (1.0 - 2.0 / PI) * k.cos());
    }
    let pi_t = PI * t;
    let num = ((1.0 - beta) * pi_t).sin() + 4.0 * beta * t * ((1.0 + beta) * pi_t).cos();
    let den = pi_t * (1.0 - (4.0 * beta * t).powi(2));
    num / den
}

/// Upsample symbols by `sps` (insert zeros) then convolve with `taps`.
/// Output length = symbols.len() * sps + taps.len() - 1.
pub fn upsample_and_filter(
    symbols: &[Complex<f32>],
    sps: usize,
    taps: &[f32],
) -> Vec<Complex<f32>> {
    let upsampled_len = symbols.len() * sps;
    let out_len = upsampled_len + taps.len() - 1;
    let mut out = vec![Complex::new(0.0, 0.0); out_len];

    // Direct convolution exploiting the sparse upsampled signal:
    // only every sps-th input sample is nonzero.
    for (sym_idx, &sym) in symbols.iter().enumerate() {
        let n = sym_idx * sps;
        for (j, &tap) in taps.iter().enumerate() {
            out[n + j] += sym * tap;
        }
    }
    out
}

/// Convolve a complex signal with a real FIR filter. Output length = x.len() + h.len() - 1.
pub fn convolve(x: &[Complex<f32>], h: &[f32]) -> Vec<Complex<f32>> {
    let out_len = x.len() + h.len() - 1;
    let mut y = vec![Complex::new(0.0, 0.0); out_len];
    for i in 0..x.len() {
        for (j, &hj) in h.iter().enumerate() {
            y[i + j] += x[i] * hj;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrc_is_symmetric() {
        let taps = rrc_taps(8, 6, 0.35);
        let n = taps.len();
        for i in 0..n / 2 {
            assert!((taps[i] - taps[n - 1 - i]).abs() < 1e-5);
        }
    }

    #[test]
    fn rrc_peaks_at_center() {
        let taps = rrc_taps(8, 6, 0.35);
        let center = taps.len() / 2;
        for (i, &t) in taps.iter().enumerate() {
            if i != center {
                assert!(t.abs() <= taps[center].abs() + 1e-6);
            }
        }
    }
}
