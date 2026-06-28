use num_complex::Complex;

/// Default feedforward filter length (in symbol-spaced taps).
pub const DEFAULT_FFF_LEN: usize = 9;

/// Default feedback filter length.
pub const DEFAULT_FBF_LEN: usize = 7;

/// Default LMS step size. Picked for ~50-symbol convergence at SNR ≥ 10 dB.
pub const DEFAULT_MU: f32 = 0.02;

/// Number of leading training symbols used for the one-tap channel estimate
/// (joint phase + gain). Chosen short so the LMS-DFE still has plenty of
/// known-target symbols afterwards to converge multi-tap behaviour.
pub const PHASE_EST_LEN: usize = 16;

/// Estimate a single complex channel coefficient (joint phase + gain) from
/// known training data: `c = E[ received · conj(target) ]`. Dividing the
/// incoming symbol stream by `c` aligns the constellation with the targets.
pub fn estimate_channel(received: &[Complex<f32>], targets: &[Complex<f32>]) -> Complex<f32> {
    let n = received.len().min(targets.len());
    if n == 0 {
        return Complex::new(1.0, 0.0);
    }
    let mut acc = Complex::new(0.0, 0.0);
    for i in 0..n {
        acc += received[i] * targets[i].conj();
    }
    acc / n as f32
}

/// Symbol-spaced decision-feedback equalizer with LMS adaptation.
///
/// Operation per symbol `k`:
///   y_ff[k] = Σⱼ h_ff[j] · input[k − center + j]   (centered window, j = 0..FFF)
///   y_fb[k] = Σⱼ h_fb[j] · past[k − 1 − j]         (causal feedback, j = 0..FBF)
///   y[k]    = y_ff[k] − y_fb[k]
///   d̂[k]    = sign(Re y[k])                        (BPSK slicer, ±1)
///
/// `past` is `targets[k']` while `k' < n_training`, then `d̂[k']` after.
///
/// LMS updates (e = target − y):
///   h_ff[j] += μ · e · conj(input[k − center + j])
///   h_fb[j] −= μ · e · conj(past[k − 1 − j])     (sign-flipped because y_fb subtracts)
pub struct Dfe {
    pub h_ff: Vec<Complex<f32>>,
    pub h_fb: Vec<Complex<f32>>,
    pub mu: f32,
}

impl Dfe {
    pub fn new(fff_len: usize, fbf_len: usize, mu: f32) -> Self {
        assert!(fff_len > 0);
        let mut h_ff = vec![Complex::new(0.0, 0.0); fff_len];
        h_ff[fff_len / 2] = Complex::new(1.0, 0.0); // center tap = passthrough
        let h_fb = vec![Complex::new(0.0, 0.0); fbf_len];
        Self { h_ff, h_fb, mu }
    }

    /// Equalize `input` of length N. The first `n_training` symbols are
    /// adapted against `targets`; the remainder are decision-directed (the
    /// slicer output is used both as the "target" for LMS and as the FBF
    /// feedback source). Returns equalized symbols (length N).
    pub fn process(
        &mut self,
        input: &[Complex<f32>],
        targets: &[Complex<f32>],
        n_training: usize,
    ) -> Vec<Complex<f32>> {
        let n = input.len();
        let n_training = n_training.min(targets.len()).min(n);
        let fff = self.h_ff.len();
        let fbf = self.h_fb.len();
        let center = fff / 2;

        let mut equalized = Vec::with_capacity(n);
        let mut decisions = Vec::with_capacity(n);

        for k in 0..n {
            // Feedforward — centered window around input[k].
            let mut y_ff = Complex::new(0.0, 0.0);
            for j in 0..fff {
                let idx = k as i64 + j as i64 - center as i64;
                if (0..n as i64).contains(&idx) {
                    y_ff += self.h_ff[j] * input[idx as usize];
                }
            }

            // Feedback — past targets (training) or decisions (data).
            let mut y_fb = Complex::new(0.0, 0.0);
            for j in 0..fbf {
                let past_idx = k as i64 - 1 - j as i64;
                if past_idx >= 0 {
                    let p = past_idx as usize;
                    let past = if p < n_training { targets[p] } else { decisions[p] };
                    y_fb += self.h_fb[j] * past;
                }
            }

            let y = y_ff - y_fb;
            equalized.push(y);

            let d = Complex::new(if y.re > 0.0 { 1.0 } else { -1.0 }, 0.0);
            decisions.push(d);

            let target = if k < n_training { targets[k] } else { d };
            let error = target - y;

            // LMS updates.
            for j in 0..fff {
                let idx = k as i64 + j as i64 - center as i64;
                if (0..n as i64).contains(&idx) {
                    self.h_ff[j] += self.mu * error * input[idx as usize].conj();
                }
            }
            for j in 0..fbf {
                let past_idx = k as i64 - 1 - j as i64;
                if past_idx >= 0 {
                    let p = past_idx as usize;
                    let past = if p < n_training { targets[p] } else { decisions[p] };
                    // Sign flipped because y_fb subtracts in the output equation.
                    self.h_fb[j] -= self.mu * error * past.conj();
                }
            }
        }

        equalized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bpsk_targets(bits: &[u8]) -> Vec<Complex<f32>> {
        bits.iter()
            .map(|&b| Complex::new(if b == 0 { 1.0 } else { -1.0 }, 0.0))
            .collect()
    }

    #[test]
    fn estimate_recovers_known_channel() {
        let bits = [0u8, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0];
        let targets = bpsk_targets(&bits);
        let h_ch = Complex::from_polar(2.3, 0.7);
        let received: Vec<Complex<f32>> = targets.iter().map(|t| t * h_ch).collect();
        let est = estimate_channel(&received, &targets);
        assert!(
            (est - h_ch).norm() < 1e-4,
            "estimate {:?} vs channel {:?}",
            est,
            h_ch
        );
    }

    #[test]
    fn dfe_passthrough_when_channel_is_clean() {
        let bits: Vec<u8> = (0..200).map(|i| ((i * 37 + 5) as u8) & 1).collect();
        let targets = bpsk_targets(&bits);
        let mut dfe = Dfe::new(DEFAULT_FFF_LEN, DEFAULT_FBF_LEN, DEFAULT_MU);
        let eq = dfe.process(&targets, &targets, 100);
        let recovered: Vec<u8> = eq
            .iter()
            .map(|s| if s.re > 0.0 { 0 } else { 1 })
            .collect();
        assert_eq!(recovered, bits);
    }

    #[test]
    fn dfe_recovers_through_two_tap_isi() {
        // Channel: received[k] = symbol[k] + 0.4 * symbol[k-2]
        let bits: Vec<u8> = (0..400).map(|i| ((i * 109 + 3) as u8) & 1).collect();
        let targets = bpsk_targets(&bits);
        let mut received = vec![Complex::new(0.0, 0.0); targets.len()];
        for k in 0..targets.len() {
            received[k] = targets[k];
            if k >= 2 {
                received[k] += Complex::new(0.4, 0.0) * targets[k - 2];
            }
        }
        let mut dfe = Dfe::new(DEFAULT_FFF_LEN, DEFAULT_FBF_LEN, DEFAULT_MU);
        let eq = dfe.process(&received, &targets, 128);
        let recovered: Vec<u8> = eq
            .iter()
            .map(|s| if s.re > 0.0 { 0 } else { 1 })
            .collect();
        // Allow a few errors in the very first symbols before LMS converges;
        // after the training portion everything should be clean.
        let payload_errors = recovered[128..]
            .iter()
            .zip(bits[128..].iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(payload_errors, 0, "{} bit errors after training", payload_errors);
    }
}
