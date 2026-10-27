// SPDX-License-Identifier: Apache-2.0
//! Exact CPU Cayley-Dickson basis products. Indices are never truncated.
use num_bigint::{BigInt, BigUint, Sign};

/// Sign of e_r * e_c, for zero-based unsigned basis indices.
pub fn twist(r: &BigUint, c: &BigUint) -> i8 {
    if r.bits() <= 64 && c.bits() <= 64 {
        return twist_u64(
            r.iter_u64_digits().next().unwrap_or(0),
            c.iter_u64_digits().next().unwrap_or(0),
        );
    }
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

    #[pymodule]
    fn cayley_kernel_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
                std::mem::swap(&mut a, &mut b);
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
