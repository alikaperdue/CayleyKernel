# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Canonical recursive Cayley-Dickson reference used by the test suite.

This module is the single source of truth for "what the doubling product
mathematically is" inside this repository.  It mirrors the classical
recursive definition:

    conj(x) = (x0, -x1, ..., -x_{n-1})
    (a, b) * (c, d) = (a*c - conj(d)*b, d*a + b*conj(c))

and is used to validate the O(1) GPU kernel in ``CayleyKernel``.
It is intentionally simple and readable rather than fast.

Everything here operates on 1-D ``cupy`` arrays.
"""

from __future__ import annotations

try:
    import cupy as cp
except ImportError:  # pragma: no cover - CPU-only machines
    cp = None


def ref_conj(x):
    """Conjugation: negate every component except the first."""
    xstar = -x
    xstar[0] = x[0]
    return xstar


def ref_mult(x, y):
    """Recursive Cayley-Dickson doubling product for 1-D cupy arrays."""
    n = len(x)
    if n == 1:
        return x * y
    m = n // 2
    a, b = x[:m], x[m:]
    c, d = y[:m], y[m:]
    z = cp.zeros(n, dtype=x.dtype)
    z[:m] = ref_mult(a, c) - ref_mult(ref_conj(d), b)
    z[m:] = ref_mult(d, a) + ref_mult(b, ref_conj(c))
    return z


def ref_sign(row, col):
    """Basis product sign from the recursive doubling rule."""
    if row == 0 or col == 0:
        return 1
    half = 1 << ((row | col).bit_length() - 1)
    if row < half:
        # d*a
        return ref_sign(col - half, row)
    if col < half:
        # b*conj(c), with a nonzero basis index for c
        return -ref_sign(row - half, col)
    # -conj(d)*b, with d scalar exactly when col == half
    sign = -1 if col == half else 1
    return sign * ref_sign(col - half, row - half)
