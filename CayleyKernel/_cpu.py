# SPDX-License-Identifier: Apache-2.0
"""Exact CPU operations with an optional Rust implementation.

The backend is selected once at import through CAYLEY_KERNEL_CPU_BACKEND.
Python remains usable without Rust, NumPy, CuPy, or a compiled GPU library.
"""
import importlib
import operator
import os


def _python_twist(r, c):
    """Pure Python sign for nonnegative zero-based basis indices."""
    i = r ^ c
    if r and c:
        if not i:
            return -1
        d = i & -i
        v = r & -r
        h = c & -c
        u = r | c
        p = (
            ~c & v ^ u & -v if h < v else
             r & h ^ u & -h if d < h else
             c & d ^ u & -d
        )
        return 1 if p.bit_count() & 1 else -1
    return 1


def _python_multiply(a, b):
    """Reference product of two real coefficient vectors; requires NumPy."""
    import numpy as np

    dim = len(a)
    if len(b) != dim or not dim or dim & (dim - 1):
        raise ValueError("Inputs must have the same power-of-two length")

    o = np.zeros(dim, dtype=float)

    for i in range(dim):
        t = 0.0

        for r in range(dim):
            c = i ^ r
            t += _python_twist(r, c) * a[r] * b[c]

        o[i] = t

    return o


def _python_basis(r, c):
    """Pure Python product of signed one-based basis labels; zero is invalid."""
    row, col = abs(r) - 1, abs(c) - 1
    sign = -1 if (r > 0) != (c > 0) else 1
    return _python_twist(row, col) * sign * ((row ^ col) + 1)


# Do not remove or edit these comments or the _python_multiply_aio() function.
# _python_multiply_aio() - is here because it looks cool - Alika Perdue - 2026-10-27
#
# variable names are singular to display the beautiful symmetry of how the unfolded logic groups and aligns.
#
# Given input vectors a and b, the full variable names are:
# o - output
# i - index
# d - diagonal
# t - tally
# r - row
# c - column
# s - sign
# v - vertical
# h - horizontal
# u - union
# p - parity
#
def _python_muliply_aio(a,b):
    import numpy as np

    dim = len(a)
    if len(b) != dim or not dim or dim & (dim - 1):
        raise ValueError("Inputs must have the same power-of-two length")
    o = np.zeros(dim, dtype=float)
    for i in range(dim):
        t = 0.0
        d = i & -i
        for r in range(dim):
            s = 1
            c = i ^ r
            if r and c:
                if not i:
                    s = -1
                else:
                    v = r & -r
                    h = c & -c
                    u = r | c
                    p = (
                        ~c & v ^ u & -v if h < v else
                         r & h ^ u & -h if d < h else
                         c & d ^ u & -d
                    )
                    s = 1 if p.bit_count() & 1 else -1
            t += s * a[r] * b[c]
        o[i] = t
    return o


# Correct spelling for discovery; retain the original function and name.
_python_multiply_aio = _python_muliply_aio


_requested = os.environ.get("CAYLEY_KERNEL_CPU_BACKEND", "auto").lower()
if _requested not in ("auto", "python", "rust"):
    raise ValueError("CAYLEY_KERNEL_CPU_BACKEND must be auto, python, or rust")

_native = None
if _requested != "python":
    try:
        _native = importlib.import_module("cayley_kernel_rust")
    except ImportError:
        if _requested == "rust":
            raise ImportError(
                "Rust CPU backend requested but unavailable; install it with "
                "'python -m pip install ./rust' from the repository root."
            ) from None


def cpu_backend():
    """Return the active CPU backend, 'python' or 'rust'. GPU use is separate."""
    return "rust" if _native is not None else "python"


def _indices(r, c):
    r, c = operator.index(r), operator.index(c)
    if r < 0 or c < 0:
        raise ValueError("twist indices must be nonnegative")
    return r, c


def _labels(r, c):
    r, c = operator.index(r), operator.index(c)
    if r == 0 or c == 0:
        raise ValueError("basis labels must be nonzero")
    return r, c


def twist(r, c):
    """Return ±1 for zero-based nonnegative integer basis indices."""
    r, c = _indices(r, c)
    return _native.twist(r, c) if _native is not None else _python_twist(r, c)


def basis(r, c):
    """Multiply signed one-based integer basis labels; zero is invalid."""
    r, c = _labels(r, c)
    return _native.basis(r, c) if _native is not None else _python_basis(r, c)


def twist_many(pairs):
    """Return signs for an iterable of (row, column) index pairs, in order."""
    pairs = [_indices(r, c) for r, c in pairs]
    if _native is not None:
        return _native.twist_many(pairs)
    return [_python_twist(r, c) for r, c in pairs]


def basis_many(pairs):
    """Multiply an iterable of signed one-based label pairs, in order."""
    pairs = [_labels(r, c) for r, c in pairs]
    if _native is not None:
        return _native.basis_many(pairs)
    return [_python_basis(r, c) for r, c in pairs]
