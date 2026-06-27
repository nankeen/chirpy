use crate::{bpsk, carrier, chirp, frame, pulse, Config, FrameError};

/// Inter-burst gap between the chirp tail and the start of the modulated data,
/// in symbols. Gives the chirp's Hann window time to ring out before data begins.
pub const POST_CHIRP_GAP_SYMBOLS: usize = 2;

/// Target peak amplitude for the final waveform (range −1..1).
/// Leaves a little headroom under digital full-scale.
pub const PEAK_TARGET: f32 = 0.9;

/// Encode `payload` into a real-valued audio waveform.
///
/// Pipeline: frame → bits → DBPSK symbols → RRC pulse-shape → upconvert to
/// `cfg.carrier_hz` → prepend the chirp preamble → normalize peak.
pub fn encode_samples(payload: &[u8], cfg: &Config) -> Result<Vec<f32>, FrameError> {
    let frame_bytes = frame::encode(payload, cfg.modulation)?;
    let bits = frame::bytes_to_bits(&frame_bytes);
    let symbols = bpsk::modulate(&bits);
    let taps = pulse::rrc_taps(cfg.sps(), cfg.rrc_span_symbols, cfg.rrc_beta);
    let baseband = pulse::upsample_and_filter(&symbols, cfg.sps(), &taps);
    let passband = carrier::upconvert(&baseband, cfg.carrier_hz, cfg.sample_rate as f32);

    let preamble = chirp::make_chirp(cfg);
    let gap = vec![0.0_f32; cfg.sps() * POST_CHIRP_GAP_SYMBOLS];
    let mut out = Vec::with_capacity(preamble.len() + gap.len() + passband.len());
    out.extend_from_slice(&preamble);
    out.extend_from_slice(&gap);
    out.extend_from_slice(&passband);

    normalize_peak(&mut out, PEAK_TARGET);
    Ok(out)
}

fn normalize_peak(samples: &mut [f32], target: f32) {
    let peak = samples.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
    if peak > 0.0 {
        let scale = target / peak;
        for s in samples {
            *s *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_nonempty_normalized_waveform() {
        let cfg = Config::default();
        let samples = encode_samples(b"hi", &cfg).unwrap();
        assert!(!samples.is_empty());
        let peak = samples.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!((peak - PEAK_TARGET).abs() < 1e-4, "peak was {peak}");
    }
}
