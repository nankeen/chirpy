use num_complex::Complex;
use std::f32::consts::TAU;

/// Mix a complex baseband signal up to a real passband signal at carrier `fc`.
/// y[n] = Re{ x[n] * e^{j 2π fc n / fs} }
pub fn upconvert(baseband: &[Complex<f32>], fc: f32, fs: f32) -> Vec<f32> {
    let omega = TAU * fc / fs;
    baseband
        .iter()
        .enumerate()
        .map(|(n, s)| {
            let phase = omega * n as f32;
            s.re * phase.cos() - s.im * phase.sin()
        })
        .collect()
}

/// Mix a real passband signal down to complex baseband by multiplying with e^{-j 2π fc n / fs}.
/// The 2·fc image is left for the matched filter to remove.
pub fn downconvert(passband: &[f32], fc: f32, fs: f32) -> Vec<Complex<f32>> {
    let omega = TAU * fc / fs;
    passband
        .iter()
        .enumerate()
        .map(|(n, &s)| {
            let phase = omega * n as f32;
            Complex::new(s * phase.cos(), -s * phase.sin())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_recovers_baseband_after_lowpass() {
        let fs = 48_000.0;
        let fc = 6_000.0;
        let n = 4_096;
        // A constant-amplitude baseband tone at +500 Hz (slow rotation).
        let baseband: Vec<Complex<f32>> = (0..n)
            .map(|k| {
                let p = TAU * 500.0 * k as f32 / fs;
                Complex::new(p.cos(), p.sin())
            })
            .collect();

        let passband = upconvert(&baseband, fc, fs);
        let down = downconvert(&passband, fc, fs);

        // Average a long stretch with explicit phase de-rotation so the residual
        // 2·fc image cancels and the baseband DC term survives. The expected
        // magnitude is 0.5 (downconverting a real cosine yields half-amplitude
        // baseband).
        let omega_b = TAU * 500.0 / fs;
        let mut acc = Complex::new(0.0, 0.0);
        let start = n / 4;
        let end = 3 * n / 4;
        for k in start..end {
            let p = omega_b * k as f32;
            let rot = Complex::new(p.cos(), -p.sin());
            acc += down[k] * rot;
        }
        acc /= (end - start) as f32;
        let mag = acc.norm();
        assert!((mag - 0.5).abs() < 0.02, "expected ~0.5, got {mag}");
    }
}
