# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Sparse (CSR-style) Cayley-Dickson product — Python API.

Experimental / research.  See ``docs/sparse-csr-design.md``.

The sparse path stores only the nonzero basis coefficients of each vector
and multiplies them using the O(1) sign rule from :mod:`_cpu`.  The
implementation dispatches to a dedicated CUDA kernel (``sparse_mult_<type>``
exported by ``cuda/cuda_kernel.cu``, ABI ≥ 5) when the kernel library is
loaded and the dtype is one of the six the sparse path supports; otherwise
it falls back to a CPU-side NumPy reference that uses the same sign rule.

Supported dtypes (per ``docs/sparse-csr-design.md`` §6 and the Phase C
kernel exports): float16, bfloat16, float32, float64, complex64,
complex128.  Integer and boolean dtypes are intentionally not supported by
the sparse path (see §7); use the dense :func:`multiply` for those.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

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
    "SPARSE_SUPPORTED_DTYPES",
]

# dtypes the Phase C kernel exports sparse_mult_<suffix> for; this set is
# the single source of truth for which dtypes take the GPU path.
SPARSE_SUPPORTED_DTYPES = (
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)

# kernel suffix per supported dtype name (must match cuda_kernel.cu)
_SPARSE_SUFFIX = {
    "float16": "half",
    "bfloat16": "bfloat16",
    "float32": "float",
    "float64": "double",
    "complex64": "complex64",
    "complex128": "complex128",
}


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
        Padded (empty) slots use ``-1`` (the sentinel the kernel reads).
    signs : cp.ndarray, dtype int8, shape (batch, K)
        Sign of each nonzero coefficient (``+1`` or ``-1``).  For complex
        dtypes this is the sign of the real part (the coefficient is
        ``sign * value`` where ``value`` carries the full magnitude and
        imaginary part); padded slots are ``0``.
    values : cp.ndarray, shape (batch, K), dtype = the input dtype
        Magnitude of each nonzero coefficient (for real dtypes); the full
        coefficient for complex dtypes.  Padded slots are ``0``.
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


def _require_numpy():
    if np is None:
        raise ImportError(
            "CayleyKernel sparse API requires NumPy.  "
            "Install it with, e.g.:  pip install numpy"
        )


def _dtype_name(dtype):
    """Return the canonical dtype name (``str(dtype).lower()``)."""
    return str(dtype).lower()


def _check_supported(dtype):
    """Validate that *dtype* is one of the six supported sparse dtypes."""
    name = _dtype_name(dtype)
    if name in SPARSE_SUPPORTED_DTYPES:
        return name
    raise TypeError(
        "sparse path supports float16, bfloat16, float32, float64, "
        f"complex64, complex128; got {dtype}.  Use dense multiply for "
        "integer or boolean dtypes (sparse integer products are a separate "
        "research question, see docs/sparse-csr-design.md §7)."
    )


def _sparse_kernel_handle(dtype_name):
    """Return the ctypes handle for ``sparse_mult_<suffix>`` or ``None``.

    ``None`` means the loaded library (if any) does not export the symbol,
    e.g. it was built before ABI 5, and the CPU fallback should be used.
    The handle is configured with ``argtypes``/``restype`` so repeated
    calls are safe.
    """
    from . import _lib, _KERNEL_LOADED

    if not _KERNEL_LOADED or _lib is None:
        return None
    suffix = _SPARSE_SUFFIX.get(dtype_name)
    if suffix is None:
        return None
    fn = getattr(_lib, "sparse_mult_%s" % suffix, None)
    if fn is None:
        return None
    fn.argtypes = [
        ctypes.c_void_p,  # x_support
        ctypes.c_void_p,  # x_signs
        ctypes.c_void_p,  # x_values
        ctypes.c_void_p,  # y_support
        ctypes.c_void_p,  # y_signs
        ctypes.c_void_p,  # y_values
        ctypes.c_void_p,  # out
        ctypes.c_int,     # K_a
        ctypes.c_int,     # K_b
        ctypes.c_int,     # dim
        ctypes.c_longlong,  # batch
    ]
    fn.restype = None
    return fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def to_sparse(x) -> SparseVector:
    """Extract the nonzero coefficients of a dense CuPy array.

    Parameters
    ----------
    x : cp.ndarray
        Shape ``(..., dim)`` where ``dim`` is a power of two.
        Supported dtype: float16, bfloat16, float32, float64, complex64,
        complex128.

    Returns
    -------
    SparseVector
        With ``support``, ``signs``, ``values`` as CuPy arrays of shape
        ``(batch, K)`` and ``K = max nonzero count`` across the batch.
    """
    _require_cupy()
    _require_numpy()
    if not isinstance(x, cp.ndarray):
        x = cp.asarray(x)
    if x.ndim < 1:
        raise ValueError("x must have at least one axis")
    dim = int(x.shape[-1])
    if dim & (dim - 1):
        raise ValueError(f"algebra dimension must be a power of two, got {dim}")
    dtype_name = _check_supported(x.dtype)

    batch = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
    xn = cp.asnumpy(x).reshape(batch, dim)
    is_complex = np.issubdtype(xn.dtype, np.complexfloating)

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
            if is_complex:
                # Store the full complex coefficient; sign is the sign of
                # the real part (kernel multiplies sign_a*sign_b*value_a*value_b
                # which equals the coefficient product for the bilinear
                # complex case we support).
                signs[r, c] = 1 if np.real(val) >= 0 else -1
                values[r, c] = val
            else:
                v = float(val)
                signs[r, c] = 1 if v >= 0 else -1
                values[r, c] = abs(v)

    return SparseVector(
        support=cp.asarray(support),
        signs=cp.asarray(signs),
        values=cp.asarray(values, dtype=x.dtype),
        K=K, dim=dim,
    )


def from_sparse(sv: SparseVector):
    """Materialise a :class:`SparseVector` as a dense CuPy array.

    Returns an array of shape ``(batch, dim)`` in the dtype of
    ``sv.values``.
    """
    _require_cupy()
    _require_numpy()
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


def _sparse_multiply_cpu(
    x_sup, x_sig, x_val, y_sup, y_sig, y_val, dim, batch, out_dtype
):
    """CPU reference implementation; used when the kernel is unavailable.

    Operates on NumPy arrays (already device-free); returns a NumPy array
    of shape ``(batch, dim)`` in ``out_dtype``.
    """
    Ka = x_sup.shape[1]
    Kb = y_sup.shape[1]
    acc = np.zeros((batch, dim), dtype=np.float64)
    is_complex = np.issubdtype(x_val.dtype, np.complexfloating)

    for r in range(batch):
        for i in range(Ka):
            a = int(x_sup[r, i])
            if a < 0:
                continue
            sa = int(x_sig[r, i])
            va = x_val[r, i]
            for j in range(Kb):
                b = int(y_sup[r, j])
                if b < 0:
                    continue
                sb = int(y_sig[r, j])
                vb = y_val[r, j]
                idx = a ^ b
                sign = _twist(a, b)
                if is_complex:
                    acc[r, idx] += sa * sb * sign * va * vb
                else:
                    acc[r, idx] += sa * sb * sign * float(va) * float(vb)

    # drop exact zeros (cancellations)
    mask = acc != 0
    if np.issubdtype(acc.dtype, np.complexfloating):
        mask = (acc.real != 0) | (acc.imag != 0)
    return acc, mask


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
    ``a⊕b`` indices with nonzero accumulated coefficients (cancellations
    yield an empty support, ``K == 0``).

    When the GPU kernel is loaded and exposes the Phase C
    ``sparse_mult_<type>`` symbol for the operand dtype (float16,
    bfloat16, float32, float64, complex64, complex128), the product is
    computed on the GPU in a single kernel launch; the returned
    :class:`SparseVector` is then materialised by reading the dense
    ``out`` buffer back to the host and re-extracting the nonzero
    entries.  Otherwise a CPU-side reference implementation is used.

    Parameters
    ----------
    x, y : SparseVector
        Must have the same ``dim`` and the same batch size.  ``x.values``
        and ``y.values`` must share a dtype that is one of the six
        supported sparse dtypes.
    out : SparseVector, optional
        Reserved for a future in-place / pre-allocated output; currently
        ignored.
    stream : optional
        Reserved for a future CUDA stream; currently ignored.

    Returns
    -------
    SparseVector
        Shape ``(batch, K_out)`` where ``K_out ≤ K_a · K_b`` (and
        ``K_out = 0`` for a zero product).  ``values`` has the same dtype
        as the inputs (the dense output is cast to the input dtype before
        re-extraction).
    """
    _require_cupy()
    _require_numpy()
    if x.dim != y.dim:
        raise ValueError(f"dimension mismatch: x.dim={x.dim}, y.dim={y.dim}")
    if x.support.shape[0] != y.support.shape[0]:
        raise ValueError(
            f"batch mismatch: x batch={x.support.shape[0]}, "
            f"y batch={y.support.shape[0]}"
        )
    dim = x.dim
    batch = x.support.shape[0]
    Ka = x.support.shape[1]
    Kb = y.support.shape[1]

    x_sup = cp.asnumpy(x.support)
    x_sig = cp.asnumpy(x.signs)
    x_val = cp.asnumpy(x.values)
    y_sup = cp.asnumpy(y.support)
    y_sig = cp.asnumpy(y.signs)
    y_val = cp.asnumpy(y.values)

    out_dtype = x_val.dtype  # same as y_val for a homogeneous call
    out_dtype_name = _dtype_name(out_dtype)
    if out_dtype_name not in SPARSE_SUPPORTED_DTYPES:
        raise TypeError(
            f"sparse_multiply requires a supported dtype; got {out_dtype}"
        )

    kernel_fn = _sparse_kernel_handle(out_dtype_name)

    if kernel_fn is not None and batch > 0 and Ka > 0 and Kb > 0:
        # ---- GPU path --------------------------------------------------
        from . import _require_gpu_runtime

        _require_gpu_runtime()
        x_sup_c = cp.ascontiguousarray(x.support, dtype=cp.int64)
        x_sig_c = cp.ascontiguousarray(x.signs, dtype=cp.int8)
        x_val_c = cp.ascontiguousarray(x.values, dtype=out_dtype)
        y_sup_c = cp.ascontiguousarray(y.support, dtype=cp.int64)
        y_sig_c = cp.ascontiguousarray(y.signs, dtype=cp.int8)
        y_val_c = cp.ascontiguousarray(y.values, dtype=out_dtype)
        dense_out = cp.zeros((batch, dim), dtype=out_dtype)

        # Same stream-ordering guard as the dense _launch: the kernel runs
        # on the legacy default stream, so pending H2D copies on a
        # non-blocking stream must be flushed first.
        if cp is not None:
            cp.cuda.get_current_stream().synchronize()

        kernel_fn(
            x_sup_c.data.ptr,
            x_sig_c.data.ptr,
            x_val_c.data.ptr,
            y_sup_c.data.ptr,
            y_sig_c.data.ptr,
            y_val_c.data.ptr,
            dense_out.data.ptr,
            ctypes.c_int(Ka),
            ctypes.c_int(Kb),
            ctypes.c_int(dim),
            ctypes.c_longlong(batch),
        )

        dense_np = cp.asnumpy(dense_out)
    else:
        # ---- CPU fallback ---------------------------------------------
        dense_np, _ = _sparse_multiply_cpu(
            x_sup, x_sig, x_val, y_sup, y_sig, y_val, dim, batch, out_dtype
        )
        dense_np = dense_np.astype(out_dtype, copy=False)

    return _dense_to_sparse_vector(dense_np, dim, out_dtype)


def _dense_to_sparse_vector(dense_np, dim, dtype) -> SparseVector:
    """Shared helper: extract the nonzero entries of a dense (batch, dim)
    NumPy array into a :class:`SparseVector` of the given dtype."""
    batch = dense_np.shape[0]
    is_complex = np.issubdtype(dense_np.dtype, np.complexfloating)

    if is_complex:
        nonzero_mask = (dense_np.real != 0) | (dense_np.imag != 0)
    else:
        nonzero_mask = dense_np != 0

    nonzero_counts = nonzero_mask.sum(axis=1)
    K_out = int(nonzero_counts.max()) if batch > 0 else 0

    if K_out == 0:
        return SparseVector(
            support=cp.zeros((batch, 0), dtype=cp.int64),
            signs=cp.zeros((batch, 0), dtype=cp.int8),
            values=cp.zeros((batch, 0), dtype=dtype),
            K=0, dim=dim,
        )

    support = np.full((batch, K_out), -1, dtype=np.int64)
    signs = np.zeros((batch, K_out), dtype=np.int8)
    values = np.zeros((batch, K_out), dtype=dense_np.dtype)

    for r in range(batch):
        nz = np.nonzero(nonzero_mask[r])[0]
        for c, idx in enumerate(nz):
            val = dense_np[r, idx]
            support[r, c] = idx
            if is_complex:
                signs[r, c] = 1 if np.real(val) >= 0 else -1
                values[r, c] = val
            else:
                v = float(val)
                signs[r, c] = 1 if v >= 0 else -1
                values[r, c] = abs(v)

    return SparseVector(
        support=cp.asarray(support),
        signs=cp.asarray(signs),
        values=cp.asarray(values, dtype=dtype),
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
