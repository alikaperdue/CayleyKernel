# SPDX-License-Identifier: Apache-2.0
"""Exact CPU operations with an optional Rust implementation.

The backend is selected once at import through CAYLEY_KERNEL_CPU_BACKEND.
Python remains usable without Rust, NumPy, CuPy, or a compiled GPU library.
"""
import importlib
import operator
import os


def _python_twist(r, c):
    if r and c:
        h, v, d = (-value & value for value in (r, c, r ^ c))
        masked = (~c & h) if h > v else ((~r & d) if v < d else (r & v))
        if r == c or not (((r | c) // max(h, v, d)) ^ masked).bit_count() & 1:
            return -1
    return 1


def _python_basis(r, c):
    row, col = abs(r) - 1, abs(c) - 1
    sign = -1 if (r > 0) != (c > 0) else 1
    return _python_twist(row, col) * sign * ((row ^ col) + 1)


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
