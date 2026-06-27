use std::f32::consts::TAU;

use chirpy_core::{decode_samples, encode_samples, wav, Config};

#[test]
fn wav_roundtrip() {
    let cfg = Config::default();
    let payload = b"the quick brown fox jumps over the lazy dog 12345".to_vec();

    let samples = encode_samples(&payload, &cfg).unwrap();
    let path = std::env::temp_dir().join("chirpy_loopback_roundtrip.wav");
    wav::write_wav(&path, &samples, cfg.sample_rate).unwrap();

    let (read_back, sr) = wav::read_wav(&path).unwrap();
    assert_eq!(sr, cfg.sample_rate);
    let decoded = decode_samples(&read_back, &cfg).expect("decode failed");
    assert_eq!(decoded, payload);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn survives_moderate_awgn() {
    // 15 dB Es/N0 — comfortably above the ~9 dB BER=1e-3 threshold for DBPSK,
    // so a successful CRC pass is the expected outcome.
    let cfg = Config::default();
    let payload: Vec<u8> = (0..128).map(|i| (i as u8).wrapping_mul(31) ^ 0x5a).collect();

    let clean = encode_samples(&payload, &cfg).unwrap();
    let signal_power: f32 = clean.iter().map(|s| s * s).sum::<f32>() / clean.len() as f32;
    let snr_db = 15.0_f32;
    let noise_power = signal_power / 10f32.powf(snr_db / 10.0);
    let sigma = noise_power.sqrt();

    let mut rng = LcgGauss::new(0xC0FFEE);
    let noisy: Vec<f32> = clean.iter().map(|s| s + sigma * rng.next()).collect();
    let decoded = decode_samples(&noisy, &cfg).expect("decode failed at 15 dB SNR");
    assert_eq!(decoded, payload);
}

#[test]
fn survives_fractional_timing_offset() {
    // Real acoustic channels never align to integer samples. Shift the entire
    // modulated waveform by a non-integer number of samples (linear interp on
    // the signal itself) and verify the decoder still recovers the payload.
    let cfg = Config::default();
    let payload = b"fractional timing offsets must not break us".to_vec();
    let clean = encode_samples(&payload, &cfg).unwrap();

    for &frac in &[0.13_f32, 0.37, 0.5, 0.71, 0.89] {
        let pre_int = 800usize;
        let mut shifted = vec![0.0_f32; pre_int + clean.len() + 8];
        for i in 0..clean.len() {
            let target = pre_int as f32 + i as f32 + frac;
            let lo = target.floor() as usize;
            let f = target - lo as f32;
            shifted[lo] += clean[i] * (1.0 - f);
            shifted[lo + 1] += clean[i] * f;
        }
        let decoded = decode_samples(&shifted, &cfg)
            .unwrap_or_else(|e| panic!("decode failed at frac={frac}: {e}"));
        assert_eq!(decoded, payload, "payload mismatch at frac={frac}");
    }
}

#[test]
fn rejects_pure_noise() {
    let cfg = Config::default();
    let mut rng = LcgGauss::new(0xDEAD_BEEF);
    let n = 96_000; // 2 seconds at 48 kHz
    let noise: Vec<f32> = (0..n).map(|_| 0.3 * rng.next()).collect();
    assert!(decode_samples(&noise, &cfg).is_err());
}

/// LCG + Box-Muller for deterministic Gaussian samples without pulling in `rand`.
struct LcgGauss {
    state: u64,
    cached: Option<f32>,
}

impl LcgGauss {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407),
            cached: None,
        }
    }

    fn uniform(&mut self) -> f32 {
        // Numerical Recipes LCG constants. 53-bit-ish output via the top 24 bits.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.state >> 40) as u32; // 24 bits
        (bits as f32 + 0.5) / (1u32 << 24) as f32
    }

    fn next(&mut self) -> f32 {
        if let Some(v) = self.cached.take() {
            return v;
        }
        // Box-Muller, clamped away from 0 to keep ln() finite.
        let u1 = self.uniform().max(1e-9);
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = TAU * u2;
        self.cached = Some(r * theta.sin());
        r * theta.cos()
    }
}
