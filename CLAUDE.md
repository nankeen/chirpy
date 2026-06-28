# CLAUDE.md

Acoustic packet modem in Rust. Transmits data through the air (speaker → mic) using IQ modulation. Two CLI binaries (`chirpy-tx`, `chirpy-rx`) share a DSP library (`chirpy-core`).

## Layout

Cargo workspace.

- `crates/chirpy-core` — DSP library. All the interesting code.
- `crates/chirpy-tx` — transmitter binary. Modulate bytes → WAV or play through speaker.
- `crates/chirpy-rx` — receiver binary. Decode WAV or live mic input. Has a debug TUI.

## Pipeline at a glance

**TX**: payload → frame bytes → bits → coherent BPSK symbols (±1) → RRC pulse shape → upconvert to carrier → prepend linear-FM chirp → real audio samples.

**RX**: audio samples → FFT-correlate against chirp template (with parabolic peak refinement for sub-sample timing) → downconvert + RRC matched-filter → brute-force ±sps/2 phase search + linear interpolation → one-tap channel estimate from training preamble → DFE (LMS-adapted) → BPSK slicer → bits → frame decode (verifies training, sync, CRC).

## Signal parameters (`Config::default`)

| Param | Value |
|---|---|
| Sample rate | 48 000 Hz |
| Carrier | 6 000 Hz |
| Symbol rate | 1 000 baud (sps = 48) |
| Pulse | Root-raised cosine, β=0.35, span=8 symbols |
| Chirp preamble | 4→8 kHz linear sweep, 50 ms, Hann-windowed |
| Modulation | Coherent BPSK (not differential) |

All configurable via CLI flags (`--carrier`, `--baud`, etc).

## Frame format (on-air bytes, after the chirp)

```
[TRAINING: 16 B = 128 symbols] [SYNC: 4 B = 0x1A 0xCF 0xFC 0x1D]
[LEN: u16 LE] [MOD: u8] [PAYLOAD: LEN B] [CRC32: u32 LE]
```

CRC covers LEN, MOD, and PAYLOAD only. Training and sync are fixed markers.
`TRAINING_SEQUENCE` lives in `frame.rs`; both sides must agree byte-for-byte.

## DFE design

`equalizer.rs`. Symbol-spaced LMS-adapted decision-feedback equalizer:

- FFF: 9 complex taps, centered window
- FBF: 7 complex taps, causal feedback from past decisions
- μ = 0.02
- Coarse channel estimate (joint phase + gain, one complex coefficient) from the first 16 training symbols; received stream divided by this before the DFE runs
- Training mode (k < 128): known targets, LMS updates against them
- Decision-directed mode (k ≥ 128): slicer output becomes both decision and target

## Commands

```bash
cargo build --release
cargo test --release                 # 33 tests, all green

# WAV roundtrip
target/release/chirpy-tx --wav /tmp/x.wav msg.txt
target/release/chirpy-rx --wav /tmp/x.wav --output decoded.bin

# Live acoustic
target/release/chirpy-rx --output got.bin           # listens
target/release/chirpy-tx msg.txt                    # plays

# Live with debug TUI (spectrogram, constellation, status sidebar)
target/release/chirpy-rx --debug-tui --output got.bin
# Quit with q / Esc / Ctrl-C
```

## Where things live

- DSP primitives: `chirpy-core/src/{pulse,carrier,chirp,bpsk}.rs`
- Framing / CRC / training: `chirpy-core/src/frame.rs`
- Equalizer: `chirpy-core/src/equalizer.rs`
- TX end-to-end: `chirpy-core/src/tx.rs::encode_samples`
- RX end-to-end: `chirpy-core/src/rx.rs::decode_samples` (and `_traced` variant)
- STFT for spectrogram: `chirpy-core/src/spectrogram.rs`
- WAV I/O: `chirpy-core/src/wav.rs`
- Live TUI: `chirpy-rx/src/debug_tui.rs`
- Integration tests: `chirpy-core/tests/loopback.rs`

## Key invariants

- The chirp matched filter is FFT-accelerated (O(L log L)). Don't reintroduce sliding correlation outside of test reference impls.
- Symbol timing recovery is open-loop, done once per frame: chirp peak (sub-sample via parabolic interp) → ±sps/2 brute-force phase search → linear-interp sampling. There is no Gardner / Mueller-Müller continuous tracking.
- BPSK is **coherent**, not differential. The DFE depends on absolute phase via the training preamble. If you reintroduce DBPSK, the DFE will not work.
- Frame format change is breaking. Old TX and new RX (or vice versa) will not interoperate.
- All public functions in `chirpy-core` operate on owned `Vec` / `&[]` slices, no streaming abstractions. Decode-once-per-buffer model.

## Out of scope

- Costas loop / PLL (replaced by training-based phase estimate)
- Gardner / Mueller-Müller continuous timing tracking
- Forward error correction (CRC-detect only, no Reed-Solomon / convolutional)
- QPSK, 16-QAM (Modulation enum exists but only Bpsk is implemented)
- Fractionally-spaced equalizer (symbol-spaced only)
- Streaming / continuous-frame receiver (single-frame decode each attempt)
- GUI debug viewer (terminal TUI only)

## Notable acoustic gotchas

- TX and RX device sample rates must match `--sample-rate` exactly; cpal will error otherwise.
- TX device must be in f32 format (we hardcode it).
- Bluetooth speakers/headphones destroy the signal — codec compression mangles 4–8 kHz. Use wired.
- Laptop speakers distort badly at >70% volume; back off if you see clipping in the TUI input meter.
- Lower `--baud` (e.g. 250) widens the symbol period; useful when reverb tail is long.
