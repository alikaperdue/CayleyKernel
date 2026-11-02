# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Sparse (CSR-style) Cayley-Dickson product — Python API.

Experimental / research.  See ``docs/sparse-csr-design.md``.

The sparse path stores only the nonzero basis coefficients of each vector
and multiplies them using the O(1) sign rule from :mod:`_cpu`.  The current
implementation is a correct CPU-side reference; a dedicated CUDA kernel is
Phase C work.

Supported dtypes: real floating point (float16, bfloat16, float32, float64).
Complex and integer dtypes are not supported by the sparse path; use the
dense :func:`multiply` for those.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None

from ._cpu import twist as _twist

__all__ = [
    "SparseVector",
    "to_sparse",
    "from_sparse",
    "sparse_multiply",
    "is_sparse_beneficial",
]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass
class SparseVector:
    """CSR-style sparse representation of a Cayley-Dickson vector or batch.

    Parameters
    ----------
    support : cp.ndarray, dtype int64, shape (batch, K)
        Zero-based basis indices of the nonzero coefficients.
        Padded (empty) slots use ``-1``.
    signs : cp.ndarray, dtype int8, shape (batch, K)
        Sign of each nonzero coefficient (``+1`` or ``-1``).
        Padded slots are ``0``.
    values : cp.ndarray, shape (batch, K)
        Magnitude (``abs()``) of each nonzero coefficient.
        Padded slots are ``0``.
    K : int
        Maximum number of nonzero coefficients across the batch.
        ``K <= dim``.  ``K == 0`` means the vector is identically zero.
    dim : int
        The algebra dimension (``2^n``).
    """
    support: "cp.ndarray"
    signs: "cp.ndarray"
    values: "cp.ndarray"
    K: int
    dim: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_cupy():
    if cp is None:
        raise ImportError(
            "CayleyKernel sparse API requires CuPy.  "
            "Install it with, e.g.:  pip install cupy-cuda12x"
        )


def _check_real_float(dtype):
    """Validate that *dtype* is a supported real floating type."""
    name = str(dtype).lower()
    allowed = ("float16", "float32", "float64", "bfloat16")
    if any(a in name for a in allowed):
        return
    raise TypeError(
        f"sparse path supports real floating dtypes (float16, bfloat16, "
        f"float32, float64); got {dtype}.  Use dense multiply for complex "
        f"or integer dtypes."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def to_sparse(x) -> SparseVector:
    """Extract the nonzero coefficients of a dense CuPy array.

    Parameters
    ----------
    x : cp.ndarray
        Shape ``(..., dim)`` where ``dim`` is a power of two.
        Real floating dtype (float16, bfloat16, float32, float64).

    Returns
    -------
    SparseVector
        With ``support``, ``signs``, ``values`` as CuPy arrays of shape
        ``(batch, K)`` and ``K = max nonzero count`` across the batch.
    """
    _require_cupy()
    if not isinstance(x, cp.ndarray):
        x = cp.asarray(x)
    if x.ndim < 1:
        raise ValueError("x must have at least one axis")
    dim = int(x.shape[-1])
    if dim & (dim - 1):
        raise ValueError(f"algebra dimension must be a power of two, got {dim}")
    _check_real_float(x.dtype)

    batch = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
    xn = cp.asnumpy(x).reshape(batch, dim)

    if batch == 0:
        return SparseVector(
            support=cp.zeros((0, 0), dtype=cp.int64),
            signs=cp.zeros((0, 0), dtype=cp.int8),
            values=cp.zeros((0, 0), dtype=x.dtype),
            K=0, dim=dim,
        )

    nonzero_counts = np.count_nonzero(xn, axis=1)
    K = int(nonzero_counts.max())
    if K == 0:
        return SparseVector(
            support=cp.zeros((batch, 0), dtype=cp.int64),
            signs=cp.zeros((batch, 0), dtype=cp.int8),
            values=cp.zeros((batch, 0), dtype=x.dtype),
            K=0, dim=dim,
        )

    support = np.full((batch, K), -1, dtype=np.int64)
    signs = np.zeros((batch, K), dtype=np.int8)
    values = np.zeros((batch, K), dtype=xn.dtype)

    for r in range(batch):
        nz = np.nonzero(xn[r])[0]
        for c, idx in enumerate(nz):
            val = xn[r, idx]
            support[r, c] = idx
            signs[r, c] = 1 if val > 0 else -1
            values[r, c] = abs(val)

    return SparseVector(
        support=cp.asarray(support),
        signs=cp.asarray(signs),
        values=cp.asarray(values, dtype=x.dtype),
        K=K, dim=dim,
    )


def from_sparse(sv: SparseVector):
    """Materialise a :class:`SparseVector` as a dense CuPy array.

    Returns an array of shape ``(batch, dim)``.
    """
    _require_cupy()
    support = cp.asnumpy(sv.support)
    signs = cp.asnumpy(sv.signs)
    values = cp.asnumpy(sv.values)
    batch, K = support.shape
    dim = sv.dim

    out = np.zeros((batch, dim), dtype=values.dtype)
    for r in range(batch):
        for c in range(K):
            idx = int(support[r, c])
            if idx < 0:
                continue
            out[r, idx] = int(signs[r, c]) * values[r, c]
    return cp.asarray(out, dtype=sv.values.dtype)


def sparse_multiply(
    x: SparseVector,
    y: SparseVector,
    out: Optional[SparseVector] = None,
    stream=None,
) -> SparseVector:
    """Cayley-Dickson product of two :class:`SparseVector` operands.

    Uses the O(1) sign rule from :mod:`_cpu` (``twist``) and the
    support-XOR model: ``e_a · e_b = ±e_{a⊕b}``.  The result is a
    :class:`SparseVector` whose support is the set of distinct
    ``a⊕b`` indices with nonzero accumulated coefficients.

    This is a **CPU-side reference implementation** (correct but not
    GPU-accelerated).  A dedicated CUDA kernel is Phase C work.

    Parameters
    ----------
    x, y : SparseVector
        Must have the same ``dim`` and the same batch size.
    out : SparseVector, optional
        Reserved for a future in-place / pre-allocated output; currently
        ignored.
    stream : optional
        Reserved for a future CUDA stream; currently ignored.

    Returns
    -------
    SparseVector
        Shape ``(batch, K_out)`` where ``K_out ≤ K_a · K_b`` (and
        ``K_out = 0`` for a zero product).  Values are float64
        (accumulation precision); cast as needed.
    """
    _require_cupy()
    if x.dim != y.dim:
        raise ValueError(f"dimension mismatch: x.dim={x.dim}, y.dim={y.dim}")
    if x.support.shape[0] != y.support.shape[0]:
        raise ValueError(
            f"batch mismatch: x batch={x.support.shape[0]}, "
            f"y batch={y.support.shape[0]}"
        )
    dim = x.dim

    x_sup = cp.asnumpy(x.support)
    x_sig = cp.asnumpy(x.signs)
    x_val = cp.asnumpy(x.values)
    y_sup = cp.asnumpy(y.support)
    y_sig = cp.asnumpy(y.signs)
    y_val = cp.asnumpy(y.values)

    batch = x_sup.shape[0]
    Ka = x_sup.shape[1]
    Kb = y_sup.shape[1]

    # Accumulate in float64 for exactness (the reference path).
    acc = np.zeros((batch, dim), dtype=np.float64)

    for r in range(batch):
        for i in range(Ka):
            a = int(x_sup[r, i])
            if a < 0:
                continue
            sa = int(x_sig[r, i])
            va = float(x_val[r, i])
            for j in range(Kb):
                b = int(y_sup[r, j])
                if b < 0:
                    continue
                sb = int(y_sig[r, j])
                vb = float(y_val[r, j])
                idx = a ^ b
                sign = _twist(a, b)
                acc[r, idx] += sa * sb * sign * va * vb

    # Extract the nonzero entries of the product.
    nonzero_counts = np.count_nonzero(acc, axis=1)
    K_out = int(nonzero_counts.max()) if batch > 0 else 0

    if K_out == 0:
        return SparseVector(
            support=cp.zeros((batch, 0), dtype=cp.int64),
            signs=cp.zeros((batch, 0), dtype=cp.int8),
            values=cp.zeros((batch, 0), dtype=np.float64),
            K=0, dim=dim,
        )

    support = np.full((batch, K_out), -1, dtype=np.int64)
    signs = np.zeros((batch, K_out), dtype=np.int8)
    values = np.zeros((batch, K_out), dtype=np.float64)

    for r in range(batch):
        nz = np.nonzero(acc[r])[0]
        for c, idx in enumerate(nz):
            val = acc[r, idx]
            support[r, c] = idx
            signs[r, c] = 1 if val > 0 else -1
            values[r, c] = abs(val)

    return SparseVector(
        support=cp.asarray(support),
        signs=cp.asarray(signs),
        values=cp.asarray(values),
        K=K_out, dim=dim,
    )


def is_sparse_beneficial(dim: int, k_a: int, k_b: int) -> bool:
    """Heuristic: is the sparse path expected to beat the dense kernel?

    Implements the crossover rule from the design doc:
    ``K* ≈ sqrt(dim / α)`` with ``α = 3.0`` (geometric mean of the
    α = 2 and α = 4 estimates in §4 of ``docs/sparse-csr-design.md``).
    Sparse wins when ``k_a · k_b < dim / α``.

    Parameters
    ----------
    dim : int
        Algebra dimension (``2^n``).
    k_a, k_b : int
        Support sizes of the two operands.

    Returns
    -------
    bool
        ``True`` if the sparse path is expected to be faster than the
        dense kernel for the given dimensions and support sizes.
    """
    alpha = 3.0
    return k_a * k_b < dim / alpha
