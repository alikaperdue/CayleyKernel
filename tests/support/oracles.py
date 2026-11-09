# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yotta Space Corp
"""Independent Cayley–Dickson doubling oracles."""

import numpy as np


def doubling_conjugate(x):
    """Negate the non-scalar coordinates, including for complex coefficients."""
    out = -x
    out[..., 0] = x[..., 0]
    return out


def recursive_product(a, b, *, xp, calc_dtype):
    """Evaluate the standard doubling rule on the last axis.

    Inputs have equal, power-of-two vector lengths and broadcastable batch
    axes. Use NumPy or CuPy explicitly. Pass calc_dtype=None to retain the
    input dtypes and the backend's arithmetic rules (use matching input dtypes
    for fixed-width arithmetic). Pass a dtype to convert both inputs before
    any arithmetic. Any final storage rounding is the caller's responsibility.

    The four-product recurrence is bilinear over complex coefficients. Its
    conjugation acts on basis coordinates and leaves scalar coefficients alone.
    For boolean kernels, pass an integer calculation dtype and compare the
    resulting nonzero pattern.
    """
    a = xp.asarray(a, dtype=calc_dtype)
    b = xp.asarray(b, dtype=calc_dtype)

    def product(x, y):
        if x.shape[-1] == 1:
            return x * y
        half = x.shape[-1] // 2
        a, b = x[..., :half], x[..., half:]
        c, d = y[..., :half], y[..., half:]
        low = product(a, c) - product(doubling_conjugate(d), b)
        high = product(d, a) + product(b, doubling_conjugate(c))
        return xp.concatenate((low, high), axis=-1)

    return product(a, b)


def doubling_sign(row, col):
    """Basis product sign from the recursive doubling rule."""
    if row == 0 or col == 0:
        return 1
    half = 1 << ((row | col).bit_length() - 1)
    if row < half:
        # d*a
        return doubling_sign(col - half, row)
    if col < half:
        # b*conj(c), with a nonzero basis index for c
        return -doubling_sign(row - half, col)
    # -conj(d)*b, with d scalar exactly when col == half
    sign = -1 if col == half else 1
    return sign * doubling_sign(col - half, row - half)


def sparse_product(dim, ix, vx, iy, vy):
    """Expand bounded sparse coefficients using the independent basis signs."""
    out = np.zeros(dim, dtype=np.float64)
    for i, vi in zip(ix, vx):
        for j, vj in zip(iy, vy):
            out[i ^ j] += doubling_sign(int(i), int(j)) * float(vi) * float(vj)
    return out


def iterative_doubling_sign(a, b):
    """Iterative block doubling; no closed sign formula or module calls."""
    s = 1
    while a or b:
        h = 1 << (max(a.bit_length(), b.bit_length()) - 1)
        u, v = a >= h, b >= h
        a, b = a % h, b % h
        if u and v:
            s *= 1 if b else -1
            a, b = b, a
        elif u:
            s *= -1 if b else 1
        elif v:
            a, b = b, a
    return s


def doubling_sign_at_level(a, b, level):
    """Independent reference from (ac-conj(d)b, da+b conj(c)).

    Process the most significant doubling bit first. The four branches
    correspond to lower/lower, lower/upper, upper/lower, upper/upper.
    This is the recursive definition evaluated iteratively, so levels above
    Python's recursion limit are supported. It does not call twist or use
    its trailing-zero/popcount formula.
    """
    sign = 1
    for bit in range(level - 1, -1, -1):
        half = 1 << bit
        upper_a, upper_b = bool(a & half), bool(b & half)
        a, b = a & (half - 1), b & (half - 1)
        if upper_a:
            if b:  # conjugation of the second operand's lower component
                sign = -sign
            if upper_b:  # minus conj(d)*b
                sign = -sign
        if upper_b:  # da or conj(d)b reverses the remaining factors
            a, b = b, a
    return sign
