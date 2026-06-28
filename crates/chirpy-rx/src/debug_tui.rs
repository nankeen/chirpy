use std::collections::VecDeque;
use std::io::stdout;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use chirpy_core::{decode_samples_traced, stft, Config, DecodeTrace};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use num_complex::Complex;
use ratatui::backend::CrosstermBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Points};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Terminal;

const SPECTROGRAM_SECONDS: f32 = 3.0;
const CONSTELLATION_POINTS: usize = 512;
const STFT_WINDOW: usize = 1024;
const STFT_HOP: usize = 256;
const DECODE_INTERVAL: Duration = Duration::from_millis(250);
const RENDER_INTERVAL: Duration = Duration::from_millis(50);

/// Run the debug TUI receiver. Listens on the default input device, decodes
/// continuously, and renders a live spectrogram, constellation, and status
/// sidebar until the user quits (q / Esc / Ctrl-C) or `timeout` elapses.
///
/// If `output` is set, the first decoded payload is written there; subsequent
/// distinct payloads overwrite. The TUI keeps running so the user can keep
/// watching the channel.
pub fn run(cfg: &Config, output: Option<&Path>, timeout: Option<Duration>) -> Result<()> {
    // --- Audio capture setup ---
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no default input device"))?;
    let supported = device.default_input_config()?;
    if supported.sample_rate().0 != cfg.sample_rate {
        anyhow::bail!(
            "input device sample rate {} != requested {}",
            supported.sample_rate().0,
            cfg.sample_rate
        );
    }
    if supported.sample_format() != cpal::SampleFormat::F32 {
        anyhow::bail!(
            "input device format is {:?}; only f32 is supported",
            supported.sample_format()
        );
    }
    let channels = supported.channels() as usize;

    let max_samples = (SPECTROGRAM_SECONDS * cfg.sample_rate as f32) as usize;
    let audio_buf: Arc<Mutex<VecDeque<f32>>> =
        Arc::new(Mutex::new(VecDeque::with_capacity(max_samples)));
    let level_peak = Arc::new(Mutex::new(0.0_f32));
    let shutdown = Arc::new(AtomicBool::new(false));
    let state = Arc::new(Mutex::new(SharedState::default()));

    let audio_cb = audio_buf.clone();
    let level_cb = level_peak.clone();
    let stream = device.build_input_stream(
        &supported.config(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buf = audio_cb.lock().unwrap();
            let mut peak = 0.0_f32;
            for frame in data.chunks(channels) {
                let s = frame[0];
                if s.abs() > peak {
                    peak = s.abs();
                }
                buf.push_back(s);
                if buf.len() > max_samples {
                    buf.pop_front();
                }
            }
            let mut lvl = level_cb.lock().unwrap();
            *lvl = lvl.max(peak);
        },
        |_err| { /* silent in TUI */ },
        None,
    )?;
    stream.play()?;

    // --- Decoder thread ---
    let decoder_audio = audio_buf.clone();
    let decoder_state = state.clone();
    let decoder_shutdown = shutdown.clone();
    let cfg_clone = cfg.clone();
    let output_path: Option<PathBuf> = output.map(PathBuf::from);
    let decoder_thread = thread::spawn(move || {
        let mut last_written: Option<Vec<u8>> = None;
        while !decoder_shutdown.load(Ordering::Relaxed) {
            thread::sleep(DECODE_INTERVAL);
            let snap: Vec<f32> = {
                let buf = decoder_audio.lock().unwrap();
                buf.iter().copied().collect()
            };
            let (result, trace) = decode_samples_traced(&snap, &cfg_clone);
            let mut st = decoder_state.lock().unwrap();
            st.trace = trace;
            st.attempts += 1;
            st.last_attempt_at = Some(Instant::now());
            if let Ok(payload) = result {
                st.successes += 1;
                st.last_payload_len = Some(payload.len());
                let candidate = Some(payload.clone());
                if candidate != last_written {
                    if let Some(p) = &output_path {
                        let _ = std::fs::write(p, &payload);
                    }
                    last_written = candidate;
                }
                drop(st);
                // Clear the capture buffer so we don't keep re-decoding the same frame.
                let mut buf = decoder_audio.lock().unwrap();
                buf.clear();
            }
        }
    });

    // --- Terminal / render loop ---
    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(out);
    let mut terminal = Terminal::new(backend)?;

    let start = Instant::now();
    let render_result = render_loop(
        &mut terminal,
        &audio_buf,
        &level_peak,
        &state,
        cfg,
        &shutdown,
        start,
        timeout,
    );

    // --- Teardown ---
    shutdown.store(true, Ordering::Relaxed);
    drop(stream);
    let _ = decoder_thread.join();
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    render_result
}

#[derive(Clone, Default)]
struct SharedState {
    trace: DecodeTrace,
    attempts: u64,
    successes: u64,
    last_attempt_at: Option<Instant>,
    last_payload_len: Option<usize>,
}

fn render_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    audio_buf: &Arc<Mutex<VecDeque<f32>>>,
    level_peak: &Arc<Mutex<f32>>,
    state: &Arc<Mutex<SharedState>>,
    cfg: &Config,
    shutdown: &Arc<AtomicBool>,
    start: Instant,
    timeout: Option<Duration>,
) -> Result<()> {
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        if let Some(t) = timeout {
            if start.elapsed() > t {
                break;
            }
        }

        if event::poll(Duration::from_millis(0))? {
            if let Event::Key(KeyEvent {
                code, modifiers, ..
            }) = event::read()?
            {
                match (code, modifiers) {
                    (KeyCode::Char('q'), _) | (KeyCode::Esc, _) => break,
                    (KeyCode::Char('c'), m) if m.contains(KeyModifiers::CONTROL) => break,
                    _ => {}
                }
            }
        }

        let audio_snap: Vec<f32> = {
            let buf = audio_buf.lock().unwrap();
            buf.iter().copied().collect()
        };
        let level = {
            let mut l = level_peak.lock().unwrap();
            let v = *l;
            *l *= 0.80; // decay between frames so the meter follows envelope
            v
        };
        let st = state.lock().unwrap().clone();

        let spec = stft(&audio_snap, STFT_WINDOW, STFT_HOP);

        terminal.draw(|frame| {
            let area = frame.area();
            let rows = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
                .split(area);

            // Spectrogram
            let spec_block = Block::default()
                .borders(Borders::ALL)
                .title(format!(
                    " Spectrogram  ·  0–{} Hz  ·  last {:.1}s ",
                    cfg.sample_rate / 2,
                    SPECTROGRAM_SECONDS
                ));
            let spec_area = spec_block.inner(rows[0]);
            frame.render_widget(spec_block, rows[0]);
            render_spectrogram(spec_area, frame.buffer_mut(), &spec, cfg);

            // Bottom: constellation + status
            let bottom = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(58), Constraint::Percentage(42)])
                .split(rows[1]);

            let const_block = Block::default()
                .borders(Borders::ALL)
                .title(format!(
                    " Constellation  ·  {} pts ",
                    st.trace.symbols.len().min(CONSTELLATION_POINTS)
                ));
            let canvas = build_constellation_canvas(&st.trace.symbols, const_block);
            frame.render_widget(canvas, bottom[0]);

            let status_block = Block::default().borders(Borders::ALL).title(" Status ");
            let status_lines = build_status(&st, level, cfg);
            let status = Paragraph::new(status_lines).block(status_block);
            frame.render_widget(status, bottom[1]);
        })?;

        thread::sleep(RENDER_INTERVAL);
    }
    Ok(())
}

/// Render a log-magnitude spectrogram into `area`. Newest column on the right.
/// Each cell uses `█` with a foreground color picked from a viridis-ish 7-step
/// ramp so signal stripes stand out against the noise floor.
fn render_spectrogram(area: Rect, buf: &mut Buffer, columns: &[Vec<f32>], _cfg: &Config) {
    let w = area.width as usize;
    let h = area.height as usize;
    if w == 0 || h == 0 || columns.is_empty() {
        return;
    }

    // Auto-scale the magnitude range to a percentile-ish window so we don't
    // depend on absolute signal level.
    let mut flat: Vec<f32> = columns.iter().flat_map(|c| c.iter().copied()).collect();
    flat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = flat[flat.len() / 20]; // 5th percentile
    let hi = flat[flat.len() * 19 / 20]; // 95th percentile
    let span = (hi - lo).max(1e-6);

    let bins = columns[0].len();
    let n_cols = columns.len();

    for cy in 0..h {
        for cx in 0..w {
            // Map x → column index, newest on the right.
            let col_idx = if n_cols >= w {
                n_cols - w + cx
            } else if cx + n_cols < w {
                continue;
            } else {
                cx + n_cols - w
            };
            let column = &columns[col_idx];
            // Map y → freq bin, top = Nyquist, bottom = DC.
            let bin = ((h - 1 - cy) * (bins - 1)) / (h - 1).max(1);
            let mag = column[bin];
            let t = ((mag - lo) / span).clamp(0.0, 1.0);
            let color = heat_color(t);
            let cell = &mut buf[(area.x + cx as u16, area.y + cy as u16)];
            cell.set_char('█').set_style(Style::default().fg(color));
        }
    }
}

fn heat_color(t: f32) -> Color {
    // 7-step ramp: black → indigo → blue → cyan → green → yellow → red.
    let buckets = [
        Color::Black,
        Color::Indexed(17),  // dark navy
        Color::Indexed(20),  // blue
        Color::Cyan,
        Color::Green,
        Color::Yellow,
        Color::Red,
    ];
    let idx = (t * (buckets.len() - 1) as f32).round() as usize;
    buckets[idx.min(buckets.len() - 1)]
}

fn build_constellation_canvas<'a>(
    symbols: &[Complex<f32>],
    block: Block<'a>,
) -> Canvas<'a, impl Fn(&mut ratatui::widgets::canvas::Context) + 'a> {
    // Snapshot the most recent points; auto-scale axes to the RMS of the data.
    let take = symbols.len().min(CONSTELLATION_POINTS);
    let recent: Vec<Complex<f32>> = symbols.iter().rev().take(take).copied().collect();

    let rms = if recent.is_empty() {
        1.0
    } else {
        let sum: f32 = recent.iter().map(|s| s.norm_sqr()).sum();
        (sum / recent.len() as f32).sqrt().max(1e-6)
    };
    let scale = rms * 2.5; // generous; keeps the cluster centered

    let coords: Vec<(f64, f64)> = recent
        .iter()
        .map(|s| (s.re as f64, s.im as f64))
        .collect();

    Canvas::default()
        .block(block)
        .marker(Marker::Braille)
        .x_bounds([-(scale as f64), scale as f64])
        .y_bounds([-(scale as f64), scale as f64])
        .paint(move |ctx| {
            // I and Q axes
            ctx.draw(&ratatui::widgets::canvas::Line {
                x1: -(scale as f64),
                y1: 0.0,
                x2: scale as f64,
                y2: 0.0,
                color: Color::DarkGray,
            });
            ctx.draw(&ratatui::widgets::canvas::Line {
                x1: 0.0,
                y1: -(scale as f64),
                x2: 0.0,
                y2: scale as f64,
                color: Color::DarkGray,
            });
            ctx.draw(&Points {
                coords: &coords,
                color: Color::Cyan,
            });
        })
}

fn build_status(state: &SharedState, level_peak: f32, cfg: &Config) -> Vec<Line<'static>> {
    let dbfs = if level_peak > 1e-6 {
        20.0 * level_peak.log10()
    } else {
        -120.0
    };
    let chirp_ratio = state.trace.chirp_peak_ratio;
    let threshold = cfg.chirp_detect_threshold;
    let chirp_status = if chirp_ratio >= threshold { "lock ✓" } else { "—" };
    let chirp_color = if chirp_ratio >= threshold {
        Color::Green
    } else if chirp_ratio >= threshold * 0.5 {
        Color::Yellow
    } else {
        Color::DarkGray
    };

    let last_seen = state
        .last_attempt_at
        .map(|t| format!("{:.1}s ago", t.elapsed().as_secs_f32()))
        .unwrap_or_else(|| "—".into());

    let payload_line = match state.last_payload_len {
        Some(n) => format!("{} B", n),
        None => "—".into(),
    };

    vec![
        Line::from(vec![
            Span::styled("Carrier   ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{:.0} Hz", cfg.carrier_hz)),
        ]),
        Line::from(vec![
            Span::styled("Baud      ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{}", cfg.baud)),
        ]),
        Line::from(vec![
            Span::styled("Sample rt ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{} Hz", cfg.sample_rate)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Input    ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{:>6.1} dBFS", dbfs)),
        ]),
        Line::from(vec![
            Span::styled("Chirp    ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:>5.2}  {}", chirp_ratio, chirp_status),
                Style::default().fg(chirp_color).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Sub-samp ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{:>+.2}", state.trace.chirp_sub_sample)),
        ]),
        Line::from(vec![
            Span::styled("Best Δφ  ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{:+} samples", state.trace.best_phase_offset)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Attempts ", Style::default().fg(Color::DarkGray)),
            Span::raw(format!("{}", state.attempts)),
        ]),
        Line::from(vec![
            Span::styled("Decoded  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", state.successes),
                Style::default().fg(if state.successes > 0 {
                    Color::Green
                } else {
                    Color::Reset
                }),
            ),
        ]),
        Line::from(vec![
            Span::styled("Last     ", Style::default().fg(Color::DarkGray)),
            Span::raw(payload_line),
        ]),
        Line::from(vec![
            Span::styled("Seen     ", Style::default().fg(Color::DarkGray)),
            Span::raw(last_seen),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "q / Esc / Ctrl-C  →  quit",
            Style::default().fg(Color::DarkGray),
        )),
    ]
}
