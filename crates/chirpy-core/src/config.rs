use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modulation {
    Bpsk,
}

impl Modulation {
    pub fn bits_per_symbol(self) -> usize {
        match self {
            Self::Bpsk => 1,
        }
    }

    pub fn to_byte(self) -> u8 {
        match self {
            Self::Bpsk => 0,
        }
    }

    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::Bpsk),
            _ => None,
        }
    }
}

impl FromStr for Modulation {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "bpsk" => Ok(Self::Bpsk),
            other => Err(format!("unknown modulation: {other}")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub sample_rate: u32,
    pub carrier_hz: f32,
    pub baud: u32,
    pub rrc_beta: f32,
    pub rrc_span_symbols: usize,
    pub chirp_dur_ms: f32,
    pub chirp_f0_hz: f32,
    pub chirp_f1_hz: f32,
    pub modulation: Modulation,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            carrier_hz: 6_000.0,
            baud: 1_000,
            rrc_beta: 0.35,
            rrc_span_symbols: 8,
            chirp_dur_ms: 50.0,
            chirp_f0_hz: 4_000.0,
            chirp_f1_hz: 8_000.0,
            modulation: Modulation::Bpsk,
        }
    }
}

impl Config {
    pub fn sps(&self) -> usize {
        (self.sample_rate / self.baud) as usize
    }

    pub fn chirp_len_samples(&self) -> usize {
        (self.chirp_dur_ms / 1000.0 * self.sample_rate as f32).round() as usize
    }
}
