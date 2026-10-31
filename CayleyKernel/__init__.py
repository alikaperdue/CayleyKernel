# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""CayleyKernel -- O(1)-order GPU Cayley-Dickson doubling products.

The Python package for the CayleyKernel project.  ``multiply`` computes
the Cayley-Dickson doubling product of any 2^n-dimensional algebra (complex,
quaternions, octonions, sedenions, ...) on NVIDIA GPUs through a compiled
CUDA kernel.

The heavy lifting is performed by ``cuda_kernel.so`` (built from
``cuda/cuda_kernel.cu`` via ``make kernel``), loaded with ctypes.  The wrapper:

  * validates that the algebra dimension is a power of two,
  * supports arbitrary batch shapes ``(B, ..., dim)`` and broadcasting,
  * handles float16/bfloat16/float32/float64/complex64/complex128/integer/boolean dtypes,
  * falls back gracefully when the shared object is missing or stale
    (no ``_nd`` symbols / no diagnostics), with an informative error.

Public API:
    multiply(a, b)       -- Cayley-Dickson product, O(1) per element on GPU
    conjugate(a)         -- standard Cayley-Dickson conjugation
    norm_sq(a), inverse(a), right_div(a, b), left_div(a, b)
    power(a, n), exp(a), inner(a, b), real(a), imag(a)
    library_path()       -- resolved path of the loaded shared object
    loaded()             -- True once the kernel library has been found
    abi_version()        -- kernel ABI version (1 for legacy builds), or 0
"""

import ctypes
import os

# GPU / array runtime.  Imports are lazy so that ``help(CayleyKernel)``
# works even on machines without CUDA, and so the error message below is the
# one users see for the actual failure.
try:
    import cupy as cp
except ImportError:  # pragma: no cover - exercised only without CUDA
    cp = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

__version__ = "0.3.0"
__all__ = [
    "multiply",
    "conjugate",
    "norm_sq",
    "inverse",
    "right_div",
    "left_div",
    "power",
    "exp",
    "inner",
    "real",
    "imag",
    "library_path",
    "loaded",
    "abi_version",
    "__version__",
    "twist",
    "basis",
    "twist_many",
    "basis_many",
    "cpu_backend",
]

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where the compiled kernel may live, in priority order:
#   1. next to this package (wheel / installed layouts)
#   2. next to the source tree (``make kernel`` in the repo root)
_LIBRARY_CANDIDATES = (
    os.path.join(_BASE_DIR, "cuda_kernel.so"),
    os.path.join(os.path.dirname(_BASE_DIR), "cuda_kernel.so"),
)

_lib = None
_library_path = None

for _candidate in _LIBRARY_CANDIDATES:
    if os.path.isfile(_candidate):
        try:
            _lib = ctypes.CDLL(_candidate)
            _library_path = _candidate
        except OSError:
            _lib = None
        break

_KERNEL_LOADED = _lib is not None

if _KERNEL_LOADED:
    try:
        _abi_fn = getattr(_lib, "cayley_abi_version")
        _abi_fn.argtypes = []
        _abi_fn.restype = ctypes.c_int
        _ABI_VERSION = int(_abi_fn())
    except AttributeError:
        _ABI_VERSION = 1
else:
    _ABI_VERSION = 0


def library_path():
    """Absolute path of the loaded kernel shared object, or ``None``."""
    return _library_path


def loaded():
    """``True`` when the CUDA kernel library has been located and loaded."""
    return _KERNEL_LOADED


def abi_version():
    """ABI/feature version reported by the loaded kernel.

    Returns ``0`` when no library is loaded and ``1`` for legacy builds that
    do not expose the ``cayley_abi_version`` symbol.  Current builds
    report ``2`` (batched ``_nd`` entry points + diagnostics).
    """
    return _ABI_VERSION


def _require_kernel():
    if not _KERNEL_LOADED:
        raise RuntimeError(
            "CayleyKernel: kernel shared library 'cuda_kernel.so' "
            "not found.  Build it with:\n"
            "    make kernel            # or: make kernel ARCHS=\"sm_80\"\n"
            "then restart Python.  (Check for prebuilt wheels on "
            "GitHub Releases if you prefer not to compile.)"
        )


# =============================================================================
# C ENTRY POINT BINDINGS
# =============================================================================
# Every type exposes two symbols:
#   mult_<type>     (a, b, out, dim)          -> single vector (legacy)
#   mult_<type>_nd  (a, b, out, dim, batch)   -> batched / multi-dimensional
#
# "dim" is always the length of the LAST axis (the algebra dimension, 2^n)
# and "batch" is the product of all leading (batch) axes.
#
# NOTE: the symbol names below are part of the stable ABI; keep them in sync
# with cuda/cuda_kernel.cu and cuda/amd_kernel.cu.

_TYPE_SUFFIXES = (
    "float",     # float32
    "double",    # float64
    "int32",
    "int64",
    "uint32",
    "uint64",
    "int16",
    "uint16",
    "int8",
    "uint8",
    "bool",
    "half",       # 16-bit float (FP16 / __half)
    "bfloat16",   # 16-bit brain float (CUDA / __nv_bfloat16)
    "complex64",  # float2  / complex float
    "complex128", # double2 / complex double
)

_LEGACY_ARGTYPES = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_ND_ARGTYPES = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_longlong]

_DTYPE_TO_SUFFIX = {
    "float32": "float",
    "float64": "double",
    "int32":   "int32",
    "int64":   "int64",
    "uint32":  "uint32",
    "uint64":  "uint64",
    "int16":   "int16",
    "uint16":  "uint16",
    "int8":    "int8",
    "uint8":   "uint8",
    "bool":    "bool",
    "bool_":   "bool",
    "float16":    "half",
    "bfloat16":   "bfloat16",
    "complex64":  "complex64",
    "complex128": "complex128",
}

# dtype name -> (legacy_fn, nd_fn_or_None).  Populated below once the
# library is loaded.
_KERNELS = {}

if _KERNEL_LOADED:
    for _suffix in _TYPE_SUFFIXES:
        _legacy = getattr(_lib, "mult_%s" % _suffix, None)
        if _legacy is None:
            continue
        _legacy.argtypes = _LEGACY_ARGTYPES
        _legacy.restype = None

        # The batched symbol only exists in kernels built from the current
        # sources.  Older shared objects fall back to a per-vector loop.
        try:
            _nd = getattr(_lib, "mult_%s_nd" % _suffix)
            _nd.argtypes = _ND_ARGTYPES
            _nd.restype = None
        except AttributeError:
            _nd = None

        for _name, _mapped in _DTYPE_TO_SUFFIX.items():
            if _mapped == _suffix:
                _KERNELS[_name] = (_legacy, _nd)

    del _suffix, _legacy, _nd, _name, _mapped


# =============================================================================
# HELPERS
# =============================================================================
def _require_gpu_runtime():
    if cp is None:
        raise ImportError(
            "CayleyKernel requires CuPy with CUDA support.  "
            "Install it with, e.g.:  pip install cupy-cuda12x"
        )
    if np is None:
        raise ImportError("CayleyKernel requires NumPy.")


def is_power_of_two(n):
    """Verify that the algebra dimension is exactly 2^n."""
    return n > 0 and (n & (n - 1)) == 0


def _select_kernel(dtype):
    key = dtype.name
    if key not in _KERNELS:
        raise TypeError(
            "Unsupported precision: %s! Use float16, bfloat16 (CUDA), float32, float64, integer, "
            "complex64, complex128, or boolean variants. Rebuild the kernel "
            "if the requested type is missing." % dtype
        )
    return _KERNELS[key]


def _result_dtype(a, b):
    """Preserve BF16 pairs; widen mixed BF16 inputs before normal promotion.

    ml_dtypes does not define a common dtype for every BF16/NumPy pair
    (notably FP16 and integers). FP32 can represent every BF16 value.
    """
    if a == b:
        return a
    if a.name == "bfloat16":
        a = cp.dtype(cp.float32)
    if b.name == "bfloat16":
        b = cp.dtype(cp.float32)
    return cp.result_type(a, b)


def _as_cupy(x, ref_dtype=None):
    """Normalise list / NumPy / CuPy input to a CuPy array."""
    if isinstance(x, cp.ndarray):
        return x
    if isinstance(x, np.ndarray):
        return cp.asarray(x)
    # lists, tuples, scalars
    inferred = cp.asarray(x)
    # Real lists follow floating/complex reference precision (including BF16).
    # Integer references and complex lists need promotion to avoid discarding
    # fractional or imaginary coefficients.
    if ref_dtype is not None:
        ref_dtype = cp.dtype(ref_dtype)
        dtype = (_result_dtype(ref_dtype, inferred.dtype)
                 if ref_dtype.kind in "biu" or inferred.dtype.kind == "c"
                 else ref_dtype)
    else:
        dtype = inferred.dtype if inferred.dtype.kind == "c" else cp.float32
    return inferred.astype(dtype, copy=False)


def _broadcast_shapes(shapes):
    """Right-aligned broadcast of several shapes (NumPy/CuPy semantics).

    Implemented by hand so we never depend on ``cp.broadcast_shapes`` (which
    is not present in every CuPy build).  Only the leading "batch" axes are
    passed in here; the algebra dimension is handled separately.
    """
    ndim = max((len(s) for s in shapes), default=0)
    out = [1] * ndim
    for s in shapes:
        i = ndim - len(s)
        for j in range(len(s)):
            d = s[j]
            cur = out[i + j]
            if cur == 1:
                out[i + j] = d
            elif d != 1 and d != cur:
                raise ValueError(
                    "operands could not be broadcast together "
                    "(batch shapes: %s)" % (shapes,)
                )
    return tuple(out)


def _launch(kernel_pair, a, b, o, dim, batch):
    """Run the kernel over ``batch`` contiguous vectors of length ``dim``."""
    legacy_fn, nd_fn = kernel_pair

    if nd_fn is not None:
        nd_fn(a.data.ptr, b.data.ptr, o.data.ptr,
              ctypes.c_int(dim), ctypes.c_longlong(batch))
        return

    # Fallback for an older kernel without the batched symbol:
    # walk the batch axis host-side, one launch per vector.
    itemsize = a.dtype.itemsize
    stride = dim * itemsize
    for i in range(batch):
        legacy_fn(a.data.ptr + i * stride,
                  b.data.ptr + i * stride,
                  o.data.ptr + i * stride,
                  ctypes.c_int(dim))


# =============================================================================
# PUBLIC API
# =============================================================================
def multiply(a, b):
    """Cayley-Dickson doubling product, executed on the GPU.

    The LAST axis is the algebra dimension and must be a power of two (2^n).
    All leading axes are treated as a batch, so *any* number of dimensions is
    supported:

        (dim,)              -> single product            (legacy behaviour)
        (N, dim)            -> N independent products
        (B, N, dim)         -> B*N independent products
        (..., dim)          -> prod(...) products

    ``a`` and ``b`` are broadcast against each other over the batch axes
    (standard NumPy/CuPy right-aligned rules), so a single vector can be
    multiplied against a whole stack::

        multiply(cp.random.rand(1024, 8), cp.random.rand(8))

    The output has the broadcast shape and the promoted dtype, matching the
    input container type (list -> list, NumPy -> NumPy, CuPy -> CuPy).
    Complex dtypes represent complex coefficients in the same basis; the
    product is complex-bilinear and does not conjugate scalar coefficients.
    Float16 and bfloat16 inputs use float32 products and accumulation, then
    round once on output. BF16 requires CuPy 14+, ml_dtypes, and a CUDA build.
    Mixed BF16 inputs widen to float32 before CuPy result_type promotion;
    all other mixed inputs follow CuPy result_type promotion directly.
    A real list as the first operand defaults to float32. As the second
    operand, it follows a floating/complex first operand's precision, or
    promotes with an integral first operand to preserve fractional values.
    Complex lists always promote to preserve imaginary coefficients.
    """
    _require_gpu_runtime()
    _require_kernel()

    # ---- 1. Track the incoming container type and standardise to CuPy ----
    if isinstance(a, cp.ndarray) or isinstance(b, cp.ndarray):
        return_type = "cupy"
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return_type = "numpy"
    else:
        return_type = "list"

    a = _as_cupy(a)
    b = _as_cupy(b, ref_dtype=a.dtype)

    if a.ndim == 0 or b.ndim == 0:
        raise ValueError(
            "Inputs must have at least one axis; the last axis is the "
            "Cayley-Dickson algebra dimension (2^n)."
        )

    # ---- 2. Unify dtypes (kernel is homogeneous) ----
    out_dtype = _result_dtype(a.dtype, b.dtype)
    if a.dtype != out_dtype:
        a = a.astype(out_dtype)
    if b.dtype != out_dtype:
        b = b.astype(out_dtype)

    kernel_pair = _select_kernel(out_dtype)

    # ---- 3. Validate / align the algebra dimension (last axis) ----
    dim = a.shape[-1]
    if b.shape[-1] != dim:
        raise ValueError(
            "Algebra dimension mismatch: a.shape[-1] (%d) != b.shape[-1] (%d). "
            "The last axis must be identical for both operands."
            % (dim, b.shape[-1])
        )
    if not is_power_of_two(dim):
        raise ValueError(
            "Algebra dimension (last axis = %d) must be a power of two (2^n)." % dim
        )

    # ---- 4. Broadcast the leading (batch) axes ----
    batch_shape = _broadcast_shapes([a.shape[:-1], b.shape[:-1]])
    out_shape = tuple(batch_shape) + (dim,)

    if a.shape != out_shape:
        a = cp.broadcast_to(a, out_shape)
    if b.shape != out_shape:
        b = cp.broadcast_to(b, out_shape)

    # ---- 5. The kernel indexes raw memory: force C-contiguity ----
    a = cp.ascontiguousarray(a)
    b = cp.ascontiguousarray(b)

    batch = 1
    for s in batch_shape:
        batch *= int(s)

    o = cp.zeros(out_shape, dtype=out_dtype)

    if batch > 0:
        _launch(kernel_pair, a, b, o, int(dim), int(batch))

    # ---- 6. Return in the caller's container type ----
    if return_type == "list":
        return cp.asnumpy(o).tolist()
    if return_type == "numpy":
        return cp.asnumpy(o)
    return o


# CPU operations have an independent optional Rust backend. GPU entry points
# and their CuPy/ctypes ABI remain unchanged.
# basis/basis_many multiply signed one-based basis labels (zero is invalid),
# using _python_basis from _cpu.py when the Rust backend is unavailable.
from ._cpu import basis, basis_many, cpu_backend, twist, twist_many

# Preserve the existing Python reference entry points without duplicating logic.
from ._cpu import _python_twist as py_sign, _python_multiply as py_multiply

from ._algebra import (
    conjugate, norm_sq, inverse, right_div, left_div, power, exp, inner, real, imag,
)
