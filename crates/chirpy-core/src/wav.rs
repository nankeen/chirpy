use std::path::Path;

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

/// Write mono samples as 16-bit PCM WAV. Samples should be in roughly [-1, 1];
/// out-of-range values are clamped.
pub fn write_wav(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> hound::Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec)?;
    let max = i16::MAX as f32;
    for &s in samples {
        let v = (s * max).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()
}

/// Read the first channel of a WAV file as f32 samples in [-1, 1].
/// Returns the samples and the sample rate.
pub fn read_wav(path: impl AsRef<Path>) -> hound::Result<(Vec<f32>, u32)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => {
            let scale = 1.0_f32 / ((1u64 << (spec.bits_per_sample - 1)) - 1) as f32;
            let interleaved: Result<Vec<i32>, _> = reader.samples::<i32>().collect();
            interleaved?
                .into_iter()
                .step_by(channels)
                .map(|v| v as f32 * scale)
                .collect()
        }
        SampleFormat::Float => {
            let interleaved: Result<Vec<f32>, _> = reader.samples::<f32>().collect();
            interleaved?.into_iter().step_by(channels).collect()
        }
    };
    Ok((samples, spec.sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_through_wav() {
        let dir = std::env::temp_dir();
        let path = dir.join("chirpy_wav_test.wav");
        let original: Vec<f32> = (0..1024)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect();
        write_wav(&path, &original, 48_000).unwrap();
        let (back, sr) = read_wav(&path).unwrap();
        assert_eq!(sr, 48_000);
        assert_eq!(back.len(), original.len());
        // Quantization tolerance for 16-bit.
        let q = 1.0 / i16::MAX as f32;
        for (a, b) in original.iter().zip(back.iter()) {
            assert!((a - b).abs() <= q * 2.0, "{} vs {}", a, b);
        }
        let _ = std::fs::remove_file(&path);
    }
}
