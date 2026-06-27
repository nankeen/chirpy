use num_complex::Complex;

/// Differential BPSK modulation.
///
/// We emit `bits.len() + 1` symbols: one reference, then one per data bit.
/// Bit 0 = no phase change. Bit 1 = π phase flip.
///
/// Using *differential* encoding means the receiver doesn't need absolute
/// carrier phase — it can decode by comparing each symbol to the previous one.
pub fn modulate(bits: &[u8]) -> Vec<Complex<f32>> {
    let mut symbols = Vec::with_capacity(bits.len() + 1);
    let mut current = 1.0_f32;
    symbols.push(Complex::new(current, 0.0));
    for &b in bits {
        debug_assert!(b <= 1);
        if b == 1 {
            current = -current;
        }
        symbols.push(Complex::new(current, 0.0));
    }
    symbols
}

/// Inverse of [`modulate`]. Given complex symbols (after carrier recovery; an
/// unknown global phase rotation is acceptable), decide each bit from the sign
/// of Re{ s_n · conj(s_{n-1}) }.
pub fn demodulate(symbols: &[Complex<f32>]) -> Vec<u8> {
    if symbols.len() < 2 {
        return Vec::new();
    }
    let mut bits = Vec::with_capacity(symbols.len() - 1);
    for w in symbols.windows(2) {
        let prod = w[1] * w[0].conj();
        bits.push(if prod.re > 0.0 { 0 } else { 1 });
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_recovers_bits() {
        let bits = vec![0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0];
        let syms = modulate(&bits);
        let out = demodulate(&syms);
        assert_eq!(bits, out);
    }

    #[test]
    fn roundtrip_robust_to_global_phase() {
        let bits = vec![1, 0, 1, 1, 0, 0, 1];
        let syms = modulate(&bits);
        let rot = Complex::from_polar(1.0, 0.7);
        let rotated: Vec<_> = syms.iter().map(|s| s * rot).collect();
        let out = demodulate(&rotated);
        assert_eq!(bits, out);
    }
}
