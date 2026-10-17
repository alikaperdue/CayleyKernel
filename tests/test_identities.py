# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Algebraic identity tests for the Cayley-Dickson product.

These tests exercise the *laws* the algebra must obey rather than raw
value equality against a reference:

  * norm multiplicativity:     |x*y|^2 == |x|^2 |y|^2   (levels 0..3)
  * conjugation laws:          conj(x*y) == conj(y)*conj(x),
                               conj(conj(x)) == x
  * real-composition identity: x * conj(x) == |x|^2 e_0  (levels 0..3)
  * unity:                     x * e_0 == e_0 * x == x
  * zero-divisor witness:      (e_3 + e_10)(e_6 - e_15) == 0  (sedenions)

Levels 4+ are non-division algebras, so the norm/composition identities are
only asserted where mathematically valid (2^0..2^3), while conjugation and
unity hold at every level.

Run with:
    pytest -q tests/test_identities.py
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from CayleyKernel import conjugate, multiply  # noqa: E402


def _rand(dtype, dim):
    return cp.random.randn(dim).astype(dtype)


# ---------------------------------------------------------------------------
# Normed-division levels: reals (1), complex (2), quaternions (4),
# octonions (8).  |x*y| = |x||y| and x*x* = |x|^2 hold here.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [1, 2, 4, 8])
def test_norm_multiplicativity(dim):
    for _ in range(5):
        x = _rand(cp.float64, dim)
        y = _rand(cp.float64, dim)
        lhs = float(cp.sum(multiply(x, y) ** 2))
        rhs = float(cp.sum(x**2)) * float(cp.sum(y**2))
        assert abs(lhs - rhs) <= 1e-8 * max(1.0, rhs), (
            f"norm multiplicativity failed at dim={dim}: {lhs} vs {rhs}"
        )


@pytest.mark.parametrize("dim", [1, 2, 4, 8])
def test_real_composition_xx_star_is_norm_squared(dim):
    for _ in range(5):
        x = _rand(cp.float64, dim)
        got = multiply(x, conjugate(x))
        norm_sq = float(cp.sum(x**2))
        assert float(got[0]) == pytest.approx(norm_sq, rel=1e-10, abs=1e-12)
        cp.testing.assert_allclose(
            got[1:], cp.zeros(dim - 1, dtype=cp.float64), atol=1e-12
        )


@pytest.mark.parametrize("dim", [2, 4, 8, 16, 32])
def test_conjugation_anti_automorphism(dim):
    """conj(x*y) == conj(y) * conj(x)."""
    for _ in range(5):
        x = _rand(cp.float64, dim)
        y = _rand(cp.float64, dim)
        lhs = conjugate(multiply(x, y))
        rhs = multiply(conjugate(y), conjugate(x))
        cp.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("dim", [2, 4, 8, 16, 32, 64, 2**16])
def test_conjugation_involution(dim):
    """conj(conj(x)) == x."""
    x = _rand(cp.float32, dim)
    cp.testing.assert_array_equal(conjugate(conjugate(x)), x)


# ---------------------------------------------------------------------------
# Unity: e_0 is the multiplicative identity at every level.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 4, 8, 16, 32, 2**10])
def test_multiplicative_identity(dim):
    x = _rand(cp.float64, dim)
    e0 = cp.zeros(dim, dtype=cp.float64)
    e0[0] = 1.0
    cp.testing.assert_allclose(multiply(x, e0), x, rtol=1e-12, atol=1e-14)
    cp.testing.assert_allclose(multiply(e0, x), x, rtol=1e-12, atol=1e-14)


# ---------------------------------------------------------------------------
# Zero divisors appear at the sedenions (dim = 16).
# ---------------------------------------------------------------------------
def test_sedenion_zero_divisors():
    """(e_3 + e_10)(e_6 - e_15) == 0 -- a classic sedenion zero divisor."""
    dim = 16
    x = cp.zeros(dim, dtype=cp.float64)
    y = cp.zeros(dim, dtype=cp.float64)
    x[3] = 1.0
    x[10] = 1.0
    y[6] = 1.0
    y[15] = -1.0

    got = multiply(x, y)
    cp.testing.assert_allclose(
        got, cp.zeros(dim, dtype=cp.float64), atol=1e-14,
        err_msg="expected a zero product from the sedenion zero-divisor pair",
    )

    # Neither factor is zero.
    assert float(cp.sum(x**2)) == pytest.approx(2.0)
    assert float(cp.sum(y**2)) == pytest.approx(2.0)


@pytest.mark.parametrize("dim", [32, 64])
def test_higher_levels_have_zero_divisors(dim):
    """Embed the sedenion zero divisor into every higher algebra."""
    x = cp.zeros(dim, dtype=cp.float64)
    y = cp.zeros(dim, dtype=cp.float64)
    x[3] = 1.0
    x[10] = 1.0
    y[6] = 1.0
    y[15] = -1.0
    got = multiply(x, y)
    cp.testing.assert_allclose(got, cp.zeros(dim, dtype=cp.float64), atol=1e-14)


# ---------------------------------------------------------------------------
# A spot of high-dimensional identity checking (cheap via unity + involution)
# ---------------------------------------------------------------------------
def test_high_dim_unity_and_involution_2pow20():
    dim = 2**20
    x = cp.zeros(dim, dtype=cp.float32)
    idx = cp.random.randint(0, dim, size=64)
    x[idx] = cp.random.randn(64, dtype=cp.float32)

    e0 = cp.zeros(dim, dtype=cp.float32)
    e0[0] = 1.0

    cp.testing.assert_allclose(multiply(x, e0), x, rtol=1e-5, atol=1e-6)
    cp.testing.assert_array_equal(conjugate(conjugate(x)), x)
