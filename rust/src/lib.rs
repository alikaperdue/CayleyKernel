// SPDX-License-Identifier: Apache-2.0
//! Exact CPU Cayley-Dickson basis products. Indices are never truncated.
#![cfg_attr(all(feature = "no_std", not(feature = "python")), no_std)]

extern crate alloc;
use alloc::{vec, vec::Vec};
use num_bigint::{BigInt, BigUint, Sign};

/// Sign of e_r * e_c, for zero-based unsigned basis indices.
pub fn twist(r: &BigUint, c: &BigUint) -> i8 {
    if r.bits() <= 64 && c.bits() <= 64 {
        return twist_u64(
            r.iter_u64_digits().next().unwrap_or(0),
            c.iter_u64_digits().next().unwrap_or(0),
        );
    }
    let bits = r.bits().max(c.bits());
    if bits <= 128 {
        return twist_words::<2>(r, c);
    }
    if bits <= 256 {
        return twist_words::<4>(r, c);
    }
    twist_big(r, c)
}

fn twist_big(r: &BigUint, c: &BigUint) -> i8 {
    if r == &BigUint::ZERO || c == &BigUint::ZERO {
        return 1;
    }
    if r == c {
        return -1;
    }
    let h = r.trailing_zeros().unwrap();
    let v = c.trailing_zeros().unwrap();
    let d = (r ^ c).trailing_zeros().unwrap();
    let shift = h.max(v).max(d);
    // The masked complement in the Python formula contributes at most one
    // bit. Inspect that bit directly, without constructing signed BigInts.
    let selected = if h > v {
        !c.bit(h)
    } else if v < d {
        !r.bit(d)
    } else {
        r.bit(v)
    };
    let parity = (((r | c) >> shift).count_ones() & 1 != 0) ^ selected;
    if parity {
        1
    } else {
        -1
    }
}

fn twist_u64(r: u64, c: u64) -> i8 {
    if r == 0 || c == 0 {
        return 1;
    }
    if r == c {
        return -1;
    }
    let h = r.trailing_zeros();
    let v = c.trailing_zeros();
    let d = (r ^ c).trailing_zeros();
    let selected = if h > v {
        c & (1 << h) == 0
    } else if v < d {
        r & (1 << d) == 0
    } else {
        r & (1 << v) != 0
    };
    let parity = (((r | c) >> h.max(v).max(d)).count_ones() & 1 != 0) ^ selected;
    if parity {
        1
    } else {
        -1
    }
}

// Stack-only limbs avoid BigUint XOR/OR/shift allocations through 256 bits.
fn twist_words<const N: usize>(r: &BigUint, c: &BigUint) -> i8 {
    let mut row = [0u64; N];
    let mut col = [0u64; N];
    for (slot, digit) in row.iter_mut().zip(r.iter_u64_digits()) {
        *slot = digit;
    }
    for (slot, digit) in col.iter_mut().zip(c.iter_u64_digits()) {
        *slot = digit;
    }
    let trailing = |words: &[u64; N]| {
        words
            .iter()
            .enumerate()
            .find(|(_, word)| **word != 0)
            .map(|(i, word)| i * 64 + word.trailing_zeros() as usize)
    };
    let (Some(h), Some(v)) = (trailing(&row), trailing(&col)) else {
        return 1;
    };
    let xor = core::array::from_fn(|i| row[i] ^ col[i]);
    let Some(d) = trailing(&xor) else {
        return -1;
    };
    let bit = |words: &[u64; N], i: usize| words[i / 64] & (1u64 << (i % 64)) != 0;
    let selected = if h > v {
        !bit(&col, h)
    } else if v < d {
        !bit(&row, d)
    } else {
        bit(&row, v)
    };
    let shift = h.max(v).max(d);
    let mut parity = selected;
    for i in shift / 64..N {
        let mut word = row[i] | col[i];
        if i == shift / 64 {
            word >>= shift % 64;
        }
        parity ^= word.count_ones() & 1 != 0;
    }
    if parity {
        1
    } else {
        -1
    }
}

/// Dense real product in ascending row order, using O(n²) work and O(n) memory.
///
/// # Panics
/// Panics unless inputs have the same nonzero power-of-two length.
pub fn multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert!(
        a.len() == b.len() && a.len().is_power_of_two(),
        "Inputs must have the same power-of-two length"
    );
    let dim = a.len();
    let mut out = vec![0.0; dim];
    if dim < 4 {
        for (i, output) in out.iter_mut().enumerate() {
            for (r, &value) in a.iter().enumerate() {
                let c = i ^ r;
                *output += f64::from(twist_u64(r as u64, c as u64)) * value * b[c];
            }
        }
        return out;
    }
    // Four independent output tallies expose instruction-level parallelism
    // without reordering additions within any coefficient or requiring threads.
    for (block, outputs) in out.chunks_exact_mut(4).enumerate() {
        let base = block * 4;
        let mut tally = [0.0; 4];
        for (r, &value) in a.iter().enumerate() {
            let term = |i: usize| {
                let c = i ^ r;
                f64::from(twist_u64(r as u64, c as u64)) * value * b[c]
            };
            tally[0] += term(base);
            tally[1] += term(base + 1);
            tally[2] += term(base + 2);
            tally[3] += term(base + 3);
        }
        outputs.copy_from_slice(&tally);
    }
    out
}

/// Signed, one-based basis labels; zero is not a basis label.
pub fn basis(r: &BigInt, c: &BigInt) -> Option<BigInt> {
    if r.sign() == Sign::NoSign || c.sign() == Sign::NoSign {
        return None;
    }
    let row = r.magnitude() - 1u8;
    let col = c.magnitude() - 1u8;
    let negative = (twist(&row, &col) < 0) ^ (r.sign() != c.sign());
    let magnitude = (row ^ col) + 1u8;
    Some(BigInt::from_biguint(
        if negative { Sign::Minus } else { Sign::Plus },
        magnitude,
    ))
}

#[cfg(feature = "python")]
mod bindings {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    #[pyfunction(name = "twist")]
    fn py_twist(r: BigUint, c: BigUint) -> i8 {
        twist(&r, &c)
    }

    #[pyfunction(name = "basis")]
    fn py_basis(r: BigInt, c: BigInt) -> PyResult<BigInt> {
        basis(&r, &c).ok_or_else(|| PyValueError::new_err("basis labels must be nonzero"))
    }

    #[pyfunction]
    fn twist_many(pairs: Vec<(BigUint, BigUint)>) -> Vec<i8> {
        pairs.iter().map(|(r, c)| twist(r, c)).collect()
    }

    #[pyfunction]
    fn basis_many(pairs: Vec<(BigInt, BigInt)>) -> PyResult<Vec<BigInt>> {
        pairs
            .iter()
            .map(|(r, c)| {
                basis(r, c).ok_or_else(|| PyValueError::new_err("basis labels must be nonzero"))
            })
            .collect()
    }

    #[pyfunction(name = "multiply")]
    fn py_multiply(py: Python<'_>, a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
        if a.len() != b.len() || !a.len().is_power_of_two() {
            return Err(PyValueError::new_err(
                "Inputs must have the same power-of-two length",
            ));
        }
        Ok(py.detach(move || multiply(&a, &b)))
    }

    #[pymodule]
    fn cayley_kernel_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(py_multiply, m)?)?;
        m.add_function(wrap_pyfunction!(py_twist, m)?)?;
        m.add_function(wrap_pyfunction!(py_basis, m)?)?;
        m.add_function(wrap_pyfunction!(twist_many, m)?)?;
        m.add_function(wrap_pyfunction!(basis_many, m)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent doubling definition, rather than the closed-form sign law.
    fn reference(mut a: u64, mut b: u64) -> i8 {
        let mut sign = 1;
        for bit in (0..64).rev() {
            let half = 1u64 << bit;
            let upper_a = a & half != 0;
            let upper_b = b & half != 0;
            a &= half - 1;
            b &= half - 1;
            if upper_a {
                if b != 0 {
                    sign = -sign;
                }
                if upper_b {
                    sign = -sign;
                }
            }
            if upper_b {
                core::mem::swap(&mut a, &mut b);
            }
        }
        sign
    }

    #[test]
    fn exhaustive_and_word_boundaries() {
        let mut values: Vec<u64> = (0..64).collect();
        values.extend([1 << 31, 1 << 32, 1 << 63, u64::MAX, u64::MAX - 1]);
        for &a in &values {
            for &b in &values {
                assert_eq!(twist(&a.into(), &b.into()), reference(a, b), "{a}, {b}");
            }
        }
    }

    #[test]
    fn fixed_width_matches_bigint() {
        for bits in [65usize, 127, 128, 129, 191, 192, 255, 256, 257] {
            let top = BigUint::from(1u8) << (bits - 1);
            let values = [
                BigUint::ZERO,
                top.clone(),
                &top - 1u8,
                &top + 1u8,
                &top + 2u8,
                (&top << 1) - 1u8,
            ];
            for a in &values {
                for b in &values {
                    assert_eq!(twist(a, b), twist_big(a, b), "{a}, {b}");
                }
            }
        }
        // Exercise every possible limb boundary for trailing-zero selection.
        for h in 0..256usize {
            for v in 0..256usize {
                let a = (BigUint::from(1u8) << h) | (BigUint::from(1u8) << 255usize);
                let b = (BigUint::from(1u8) << v) | (BigUint::from(3u8) << 254usize);
                assert_eq!(twist(&a, &b), twist_big(&a, &b));
            }
        }
    }

    #[test]
    fn dense_product_against_doubling() {
        for dim in [1, 2, 4, 8, 16, 32] {
            let a: Vec<_> = (0..dim).map(|i| (i % 7) as f64 - 3.0).collect();
            let b: Vec<_> = (0..dim).map(|i| (i % 5) as f64 - 2.0).collect();
            let mut expected = vec![0.0; dim];
            for r in 0..dim {
                for c in 0..dim {
                    expected[r ^ c] += f64::from(reference(r as u64, c as u64)) * a[r] * b[c];
                }
            }
            assert_eq!(multiply(&a, &b), expected);
        }
    }

    #[test]
    #[should_panic(expected = "same power-of-two length")]
    fn dense_rejects_empty() {
        multiply(&[], &[]);
    }

    // Run with cargo test --release benchmark_index_paths -- --ignored --nocapture.
    #[cfg(not(feature = "no_std"))]
    #[test]
    #[ignore = "manual timing benchmark; no timing assertions"]
    fn benchmark_index_paths() {
        use std::{hint::black_box, time::Instant};
        println!("bits,dispatch_ns,bigint_ns");
        for bits in [32usize, 64, 65, 128, 129, 256, 257, 1024, 4096] {
            let pairs: Vec<_> = (1..=128u64)
                .map(|i| {
                    let top = BigUint::from(1u8) << (bits - 1);
                    (&top + i, &top + i * 3 + 1u64)
                })
                .collect();
            for (a, b) in &pairs {
                assert_eq!(twist(a, b), twist_big(a, b));
            }
            let mut times = [0.0; 2];
            for (slot, function) in times
                .iter_mut()
                .zip([twist as fn(&BigUint, &BigUint) -> i8, twist_big])
            {
                let mut samples = Vec::new();
                for _ in 0..5 {
                    let start = Instant::now();
                    for _ in 0..1000 {
                        for (a, b) in &pairs {
                            black_box(function(black_box(a), black_box(b)));
                        }
                    }
                    samples.push(start.elapsed().as_nanos() as f64 / 128_000.0);
                }
                samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                *slot = samples[2];
            }
            println!("{bits},{:.2},{:.2}", times[0], times[1]);
        }
    }

    #[test]
    fn large_indices_and_signed_labels() {
        let a = BigUint::from(3u8) << 100_000usize;
        let b = BigUint::from(6u8) << 100_000usize;
        assert_eq!(twist(&a, &b), reference(3, 6));
        assert_eq!(twist(&a, &a), -1);
        assert_eq!(twist(&BigUint::ZERO, &a), 1);
        let r = BigInt::from(a + 1u8);
        let c = BigInt::from(b + 1u8);
        assert_eq!(basis(&(-&r), &c), basis(&r, &c).map(|v| -v));
        assert!(basis(&BigInt::ZERO, &c).is_none());
    }
}
