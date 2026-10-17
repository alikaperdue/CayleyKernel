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
    """Basis product sign  e_row * e_col = sign * e_(row ^ col).

    Closed-form branch-free expression mirroring the CUDA kernel's
    ``cayley_sign`` (see cuda/cuda_kernel.cu).  Kept here so the host-side sign
    table can be cross-checked against the kernel's behaviour without
    needing to run device code.
    """
    r = row
    c = row ^ col
    if r == 0 or c == 0:
        return 1
    if r == c:
        return -1

    ur, uc = r, c
    tr = _ffs(ur) - 1
    tc = _ffs(uc) - 1
    tx = _ffs(ur ^ uc) - 1

    if tc < tr:
        s = (~uc >> tr)
    elif tc < tx:
        s = (~ur >> tx)
    else:
        s = (ur >> tc)

    m = max(tc, tx, tr)
    parity = s ^ _popcount((ur | uc) >> m)
    return 1 if (parity & 1) else -1


def _ffs(x):
    """Find-first-set, 1-based (CUDA __ffs semantics); 0 for x == 0."""
    if x == 0:
        return 0
    return (x & -x).bit_length()


def _popcount(x):
    return bin(x & 0xFFFFFFFF).count("1")
