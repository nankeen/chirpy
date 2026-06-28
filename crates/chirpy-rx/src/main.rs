use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use chirpy_core::{decode_samples, wav, Config, DecodeError, Modulation};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

mod debug_tui;

#[derive(Parser, Debug)]
#[command(name = "chirpy-rx", about = "Acoustic packet modem — receiver")]
struct Args {
    /// Read modulated samples from this WAV file instead of the microphone.
    #[arg(long)]
    wav: Option<PathBuf>,
    /// Carrier frequency in Hz.
    #[arg(long, default_value_t = 6_000.0)]
    carrier: f32,
    /// Symbol rate in baud.
    #[arg(long, default_value_t = 1_000)]
    baud: u32,
    /// Sample rate in Hz (live audio only — must match the capture device).
    #[arg(long, default_value_t = 48_000)]
    sample_rate: u32,
    /// Modulation scheme.
    #[arg(long, default_value = "bpsk")]
    modulation: Modulation,
    /// Output file. Defaults to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
    /// In live mode, how long to listen before giving up (seconds).
    #[arg(long, default_value_t = 30.0)]
    timeout: f32,
    /// Live debug TUI: spectrogram, constellation, and status. Live audio only.
    #[arg(long, alias = "debug")]
    debug_tui: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    let args = Args::parse();

    let cfg = Config {
        sample_rate: args.sample_rate,
        carrier_hz: args.carrier,
        baud: args.baud,
        modulation: args.modulation,
        ..Default::default()
    };

    if args.debug_tui {
        if args.wav.is_some() {
            anyhow::bail!("--debug-tui is for live audio; not supported with --wav");
        }
        return debug_tui::run(
            &cfg,
            args.output.as_deref(),
            Some(Duration::from_secs_f32(args.timeout)),
        );
    }

    let payload = if let Some(path) = &args.wav {
        let (samples, sr) = wav::read_wav(path)
            .with_context(|| format!("reading WAV {}", path.display()))?;
        let cfg = Config {
            sample_rate: sr,
            ..cfg
        };
        decode_samples(&samples, &cfg)?
    } else {
        listen_for_frame(&cfg, Duration::from_secs_f32(args.timeout))?
    };

    write_payload(&payload, args.output.as_deref())?;
    Ok(())
}

fn write_payload(payload: &[u8], path: Option<&Path>) -> Result<()> {
    match path {
        Some(p) => {
            std::fs::write(p, payload).with_context(|| format!("writing {}", p.display()))?;
            tracing::info!("wrote {} bytes to {}", payload.len(), p.display());
        }
        None => {
            std::io::stdout().write_all(payload)?;
            std::io::stdout().flush()?;
        }
    }
    Ok(())
}

/// Capacity (in seconds) for the rolling capture buffer before we drop the oldest
/// half. Picked so a typical frame fits comfortably; long enough to absorb a slow
/// receiver while still bounding the matched-filter cost.
const BUFFER_CAPACITY_SECONDS: f32 = 6.0;

/// How often we attempt to decode the accumulated buffer.
const DECODE_INTERVAL: Duration = Duration::from_millis(250);

fn listen_for_frame(cfg: &Config, timeout: Duration) -> Result<Vec<u8>> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no default input device"))?;
    let supported = device.default_input_config()?;
    let device_sr = supported.sample_rate().0;
    let channels = supported.channels() as usize;

    if device_sr != cfg.sample_rate {
        anyhow::bail!(
            "input device sample rate {} != requested {} — pass --sample-rate {}",
            device_sr,
            cfg.sample_rate,
            device_sr
        );
    }
    if supported.sample_format() != cpal::SampleFormat::F32 {
        anyhow::bail!(
            "input device sample format is {:?}; only f32 is supported in v1",
            supported.sample_format()
        );
    }

    let capture: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let capture_cb = capture.clone();
    let stream = device.build_input_stream(
        &supported.config(),
        move |data: &[f32], _info: &cpal::InputCallbackInfo| {
            let mut buf = capture_cb.lock().unwrap();
            // Take only the first channel.
            for frame in data.chunks(channels) {
                buf.push(frame[0]);
            }
        },
        |err| tracing::error!("input stream error: {err}"),
        None,
    )?;
    stream.play()?;
    tracing::info!(
        "listening on default input ({} Hz, {} ch); timeout {:.1}s",
        device_sr,
        channels,
        timeout.as_secs_f32()
    );

    let max_samples = (BUFFER_CAPACITY_SECONDS * cfg.sample_rate as f32) as usize;
    let started = Instant::now();

    loop {
        std::thread::sleep(DECODE_INTERVAL);
        if started.elapsed() > timeout {
            return Err(anyhow!("timed out after {:.1}s without a successful decode", timeout.as_secs_f32()));
        }

        let snapshot: Vec<f32> = {
            let mut buf = capture.lock().unwrap();
            if buf.len() > max_samples {
                // Drop the oldest half so total cost stays bounded.
                let drop = buf.len() - max_samples / 2;
                buf.drain(..drop);
            }
            buf.clone()
        };

        match decode_samples(&snapshot, cfg) {
            Ok(payload) => {
                tracing::info!("decoded {} bytes", payload.len());
                return Ok(payload);
            }
            Err(DecodeError::ChirpNotFound) => {
                // Normal — buffer just doesn't contain a frame yet.
            }
            Err(e) => {
                tracing::debug!("partial decode: {e}");
            }
        }
    }
}
