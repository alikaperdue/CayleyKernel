# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Slow, opt-in tests: deep identities and batched products at 2^20..2^22.

These are excluded from the default run (they are marked ``slow`` in
pyproject.toml).  Run them explicitly with:

    pytest -q -m slow tests/test_slow.py

Budget note (tune per machine): 2^20 ~ a few seconds per identity check,
2^21 ~ tens of seconds, 2^22 ~ minutes.  The default suite already covers
all of the same laws at smaller dimensions.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from CayleyKernel import conjugate, multiply  # noqa: E402


def _unity_check(dim, n_nonzero, dtype):
    """x * e_0 == e_0 * x == x with random support of size n_nonzero."""
    x = cp.zeros(dim, dtype=dtype)
    idx = cp.random.randint(0, dim, size=n_nonzero)
    vals = cp.random.randn(n_nonzero).astype(dtype)
    x[idx] = vals

    e0 = cp.zeros(dim, dtype=dtype)
    e0[0] = 1

    cp.testing.assert_allclose(multiply(x, e0), x, rtol=1e-5, atol=1e-6)
    cp.testing.assert_allclose(multiply(e0, x), x, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Unity / identity law at extreme dimensions
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("dim", [2**20, 2**21])
def test_deep_unity(dim):
    _unity_check(dim, n_nonzero=64, dtype=cp.float32)


# ---------------------------------------------------------------------------
# Conjugation involution + anti-automorphism at deep dimensions.
# (Anti-automorphism needs a full product: use sparse operands.)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("dim", [2**20, 2**21])
def test_deep_conjugation_involution(dim):
    x = cp.zeros(dim, dtype=cp.float32)
    idx = cp.random.randint(0, dim, size=64)
    x[idx] = cp.random.randn(64, dtype=cp.float32)
    cp.testing.assert_array_equal(conjugate(conjugate(x)), x)


@pytest.mark.slow
@pytest.mark.parametrize("dim", [2**20])
def test_deep_conjugation_anti_automorphism_sparse(dim):
    """conj(x*y) == conj(y)*conj(x) on sparse operands at 2^20."""
    rng = np.random.default_rng(99)
    x = cp.zeros(dim, dtype=cp.float64)
    y = cp.zeros(dim, dtype=cp.float64)
    ix = rng.choice(dim, size=4, replace=False)
    iy = rng.choice(dim, size=4, replace=False)
    x[ix] = rng.normal(size=4)
    y[iy] = rng.normal(size=4)

    lhs = conjugate(multiply(x, y))
    rhs = multiply(conjugate(y), conjugate(x))
    cp.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Batched deep products: many 2^16-dimensional products in one launch.
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_batched_deep_products():
    dim = 2**16
    batch = 32
    A = cp.random.randn(batch, dim).astype(cp.float32) * 0.01
    B = cp.random.randn(batch, dim).astype(cp.float32) * 0.01

    got = multiply(A, B)
    assert got.shape == (batch, dim)

    e0 = cp.zeros(dim, dtype=cp.float32)
    e0[0] = 1.0
    # Sanity: multiply the batch by the identity broadcast over the stack.
    ident = multiply(A, cp.broadcast_to(e0, (batch, dim)))
    cp.testing.assert_allclose(ident, A, rtol=1e-5, atol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("dim", [2**22])
def test_very_deep_unity_smoke(dim):
    """2^22 smoke test: single sparse unity product (minutes-scale)."""
    _unity_check(dim, n_nonzero=16, dtype=cp.float32)
