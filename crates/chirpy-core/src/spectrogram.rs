use num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::TAU;

/// Short-time Fourier transform.
///
/// Splits `samples` into Hann-windowed frames of `window_size` samples each,
/// stepping by `hop` samples. For each frame returns the log-magnitude of the
/// first `window_size / 2 + 1` FFT bins (positive frequencies up to Nyquist).
///
/// Output is shaped as `Vec<column>` where each column is one frame in time.
/// Values are `log10(|X[k]| + ε)` so silent bins map to a finite, very negative
/// number rather than `-inf`.
pub fn stft(samples: &[f32], window_size: usize, hop: usize) -> Vec<Vec<f32>> {
    assert!(window_size > 0 && hop > 0, "window_size and hop must be > 0");
    if samples.len() < window_size {
        return Vec::new();
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(window_size);

    let window: Vec<f32> = (0..window_size)
        .map(|i| 0.5 * (1.0 - (TAU * i as f32 / (window_size - 1) as f32).cos()))
        .collect();

    let n_bins = window_size / 2 + 1;
    let n_frames = (samples.len() - window_size) / hop + 1;
    let mut out = Vec::with_capacity(n_frames);
    let mut buf = vec![Complex::new(0.0_f32, 0.0); window_size];

    const EPS: f32 = 1e-9;

    for frame in 0..n_frames {
        let start = frame * hop;
        for (i, b) in buf.iter_mut().enumerate() {
            *b = Complex::new(samples[start + i] * window[i], 0.0);
        }
        fft.process(&mut buf);
        let column: Vec<f32> = buf[..n_bins].iter().map(|c| (c.norm() + EPS).log10()).collect();
        out.push(column);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stft_locates_known_tone() {
        let fs = 48_000.0_f32;
        let f0 = 1_000.0_f32;
        let n = 4096;
        let samples: Vec<f32> = (0..n).map(|i| (TAU * f0 * i as f32 / fs).sin()).collect();

        let window_size = 1024;
        let hop = 256;
        let spec = stft(&samples, window_size, hop);
        assert!(!spec.is_empty());

        // Bin frequency = k * fs / window_size. For f0 = 1000 at fs = 48000,
        // window 1024, expected bin = 1000 / (48000/1024) ≈ 21.33.
        let bin_hz = fs / window_size as f32;
        let expected_bin = (f0 / bin_hz).round() as usize;

        // Take a middle frame; the tone fills the whole signal.
        let column = &spec[spec.len() / 2];
        let (peak_bin, _) = column
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(
            peak_bin.abs_diff(expected_bin) <= 1,
            "peak at bin {peak_bin}, expected ~{expected_bin}"
        );
    }

    #[test]
    fn stft_handles_short_input() {
        let spec = stft(&[0.0; 10], 64, 16);
        assert!(spec.is_empty());
    }
}
