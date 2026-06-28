use num_complex::Complex;

/// Coherent BPSK modulation. Bit 0 → +1, bit 1 → −1. Symbols are real.
///
/// We use coherent (not differential) BPSK because the decision-feedback
/// equalizer needs an absolute phase reference. The receiver recovers that
/// from the training preamble.
pub fn modulate(bits: &[u8]) -> Vec<Complex<f32>> {
    bits.iter()
        .map(|&b| {
            debug_assert!(b <= 1);
            Complex::new(if b == 0 { 1.0 } else { -1.0 }, 0.0)
        })
        .collect()
}

/// Coherent BPSK slicer. Returns 0 for `Re(s) > 0`, else 1.
pub fn demodulate(symbols: &[Complex<f32>]) -> Vec<u8> {
    symbols
        .iter()
        .map(|s| if s.re > 0.0 { 0 } else { 1 })
        .collect()
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
    fn maps_to_expected_symbols() {
        let syms = modulate(&[0, 1, 0]);
        assert_eq!(syms[0].re, 1.0);
        assert_eq!(syms[1].re, -1.0);
        assert_eq!(syms[2].re, 1.0);
    }
}
