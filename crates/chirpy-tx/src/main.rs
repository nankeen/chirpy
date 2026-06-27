use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use chirpy_core::{encode_samples, wav, Config, Modulation};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

#[derive(Parser, Debug)]
#[command(name = "chirpy-tx", about = "Acoustic packet modem — transmitter")]
struct Args {
    /// Write modulated samples to this WAV file instead of playing through the speaker.
    #[arg(long)]
    wav: Option<PathBuf>,
    /// Carrier frequency in Hz.
    #[arg(long, default_value_t = 6_000.0)]
    carrier: f32,
    /// Symbol rate in baud.
    #[arg(long, default_value_t = 1_000)]
    baud: u32,
    /// Sample rate in Hz. Must match the playback device when using live audio.
    #[arg(long, default_value_t = 48_000)]
    sample_rate: u32,
    /// Modulation scheme.
    #[arg(long, default_value = "bpsk")]
    modulation: Modulation,
    /// Input file path (use "-" or omit for stdin).
    input: Option<PathBuf>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();

    let payload = read_payload(args.input.as_deref())?;
    let cfg = Config {
        sample_rate: args.sample_rate,
        carrier_hz: args.carrier,
        baud: args.baud,
        modulation: args.modulation,
        ..Default::default()
    };

    let samples = encode_samples(&payload, &cfg)?;
    let seconds = samples.len() as f32 / cfg.sample_rate as f32;
    tracing::info!(
        "encoded {} payload bytes -> {} samples ({:.2}s) at {} Hz, carrier {:.0} Hz, {} baud",
        payload.len(),
        samples.len(),
        seconds,
        cfg.sample_rate,
        cfg.carrier_hz,
        cfg.baud,
    );

    if let Some(path) = args.wav {
        wav::write_wav(&path, &samples, cfg.sample_rate)
            .with_context(|| format!("writing WAV to {}", path.display()))?;
        tracing::info!("wrote WAV: {}", path.display());
    } else {
        play_samples(&samples, cfg.sample_rate)?;
    }
    Ok(())
}

fn read_payload(path: Option<&Path>) -> Result<Vec<u8>> {
    match path {
        Some(p) if p != Path::new("-") => {
            std::fs::read(p).with_context(|| format!("reading {}", p.display()))
        }
        _ => {
            let mut buf = Vec::new();
            std::io::stdin().read_to_end(&mut buf)?;
            Ok(buf)
        }
    }
}

fn play_samples(samples: &[f32], sample_rate: u32) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow!("no default output device"))?;
    let supported = device.default_output_config()?;
    let channels = supported.channels() as usize;
    let device_sr = supported.sample_rate().0;

    if device_sr != sample_rate {
        anyhow::bail!(
            "output device sample rate {} != requested {} — \
             pass --sample-rate {} to match, or use --wav for offline encode",
            device_sr,
            sample_rate,
            device_sr
        );
    }

    let buffer = Arc::new(samples.to_vec());
    let position = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicBool::new(false));

    let buf_cb = buffer.clone();
    let pos_cb = position.clone();
    let done_cb = done.clone();
    let stream = device.build_output_stream(
        &supported.config(),
        move |data: &mut [f32], _info: &cpal::OutputCallbackInfo| {
            let mut cur = pos_cb.load(Ordering::Relaxed);
            let frames = data.len() / channels;
            for i in 0..frames {
                let s = buf_cb.get(cur + i).copied().unwrap_or(0.0);
                let base = i * channels;
                for ch in 0..channels {
                    data[base + ch] = s;
                }
            }
            cur += frames;
            pos_cb.store(cur, Ordering::Relaxed);
            if cur >= buf_cb.len() {
                done_cb.store(true, Ordering::Relaxed);
            }
        },
        |err| tracing::error!("output stream error: {err}"),
        None,
    )?;
    stream.play()?;
    tracing::info!("playing {} samples through default output", samples.len());

    while !done.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    // Let the device drain its buffer.
    std::thread::sleep(std::time::Duration::from_millis(250));
    drop(stream);
    Ok(())
}
