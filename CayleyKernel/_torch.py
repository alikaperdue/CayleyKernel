# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Optional PyTorch interop adapter for CayleyKernel.

This module provides a thin, zero-copy adapter that lets CUDA
``torch.Tensor`` inputs call into the existing CuPy CUDA kernel without any
D2H round-trip.  The adapter wraps the caller's PyTorch storage in
``cupy.cuda.UnownedMemory`` so the kernel reads and writes the same physical
memory, and the original ``torch.Tensor`` is returned (identity preserved
when ``out=`` is supplied).

Not supported in v1:

* **autograd** -- calling with ``requires_grad=True`` raises
  ``NotImplementedError`` with a documented message.
* ``torch.jit`` scripting/compilation.
* CPU tensors -- the adapter is GPU-only by design.
* dtype promotion -- both operands must share a dtype (homogeneous kernel).

JAX interop is intentionally a separate branch (``feat/jax-interop``).
"""

from __future__ import annotations

import torch  # noqa: F401  (module-level import; guarded by the parent __init__)

from . import is_power_of_two

# Map torch dtype -> cupy dtype name (string, looked up on the cp module).
_TORCH_TO_CUPY = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
}

_SUPPORTED_TORCH_DTYPES = "float16, bfloat16, float32, float64, complex64, complex128"

_GRAD_MSG = (
    "CayleyKernel.torch.multiply: PyTorch autograd is not supported in v1 "
    "(see the 'PyTorch interop' section of the README). "
    "Disable gradients with torch.no_grad() or pass requires_grad=False tensors."
)


def _cupy():
    """Return the CuPy module or raise a clear ImportError."""
    from . import cp

    if cp is None:
        raise ImportError(
            "CayleyKernel requires CuPy with CUDA support. "
            "Install with: pip install 'cayley-kernel[cuda12]'"
        )
    return cp


def _supported_cupy_dtype(t_dtype):
    """Map a torch dtype to its CuPy name, or raise TypeError."""
    if t_dtype not in _TORCH_TO_CUPY:
        raise TypeError(
            "CayleyKernel.torch supports only these dtypes: %s (got %s)."
            % (_SUPPORTED_TORCH_DTYPES, t_dtype)
        )
    return _TORCH_TO_CUPY[t_dtype]


def _torch_to_cupy(t):
    """Wrap a CUDA ``torch.Tensor`` in a CuPy ndarray over the same storage.

    Uses ``cupy.cuda.UnownedMemory`` so no copy happens for CUDA tensors.
    For non-contiguous tensors, ``.contiguous()`` is called first, which
    returns a new tensor with a new storage; the UnownedMemory wraps that
    new storage.
    """
    cp = _cupy()
    if not t.is_cuda:
        raise TypeError(
            "CayleyKernel.torch.multiply requires CUDA tensors (got a CPU tensor on %s). "
            "Move inputs to GPU with .to('cuda') first." % t.device
        )
    if not t.is_contiguous():
        t = t.contiguous()
    dtype_name = _supported_cupy_dtype(t.dtype)
    cupy_dtype = getattr(cp, dtype_name)
    if dtype_name == "bfloat16" and cupy_dtype is None:
        raise ImportError(
            "CayleyKernel.torch bfloat16 requires CuPy 14+ and ml_dtypes. "
            "Install with: pip install 'cayley-kernel[bfloat16]'"
        )
    ptr = t.untyped_storage().data_ptr()
    size = t.nelement() * t.element_size()
    mem = cp.cuda.UnownedMemory(ptr, size, owner=t)
    mp = cp.cuda.MemoryPointer(mem, 0)
    arr = cp.ndarray(tuple(t.shape), dtype=cupy_dtype, memptr=mp)
    return arr


def multiply(a, b, out=None, stream=None):
    """Cayley-Dickson doubling product for CUDA ``torch.Tensor`` inputs.

    The LAST axis is the algebra dimension and must be a power of two (2^n).
    All leading axes are treated as a batch.  ``a`` and ``b`` must share
    shape, dtype, and device.  Supported dtypes are ``float16``, ``bfloat16``,
    ``float32``, ``float64``, ``complex64``, and ``complex128``.

    When ``out`` is supplied the result is written in-place and the same
    ``torch.Tensor`` is returned (identity preserved).  When ``out`` is
    ``None`` a new tensor is allocated and returned.

    ``stream`` is passed through to the underlying CuPy kernel launch.

    Raises ``NotImplementedError`` when ``torch.is_grad_enabled()`` and either
    operand has ``requires_grad=True`` (autograd is out of scope in v1).
    """
    _cupy()

    # Autograd guard
    if torch.is_grad_enabled():
        if a.requires_grad or b.requires_grad or (out is not None and out.requires_grad):
            raise NotImplementedError(_GRAD_MSG)

    # Device / shape / dtype validation
    if not a.is_cuda:
        raise TypeError("CayleyKernel.torch.multiply: a must be a CUDA tensor (got %s)." % a.device)
    if not b.is_cuda:
        raise TypeError("CayleyKernel.torch.multiply: b must be a CUDA tensor (got %s)." % b.device)
    if a.device != b.device:
        raise ValueError(
            "CayleyKernel.torch.multiply: a.device (%s) != b.device (%s)." % (a.device, b.device)
        )
    if a.ndim < 1 or b.ndim < 1:
        raise ValueError(
            "CayleyKernel.torch.multiply: inputs must have at least one axis; "
            "the last axis is the algebra dimension (2^n)."
        )
    if a.shape != b.shape:
        raise ValueError(
            "CayleyKernel.torch.multiply: a.shape %s != b.shape %s. "
            "Both operands must have identical shapes." % (tuple(a.shape), tuple(b.shape))
        )
    dim = a.shape[-1]
    if not is_power_of_two(dim):
        raise ValueError("Algebra dimension (last axis = %d) must be a power of two (2^n)." % dim)
    if a.dtype != b.dtype:
        raise TypeError(
            "CayleyKernel.torch.multiply requires homogeneous dtypes (got %s and %s). "
            "Cast one operand with .to(<dtype>) first." % (a.dtype, b.dtype)
        )
    _supported_cupy_dtype(a.dtype)

    # Validate / prepare out
    if out is not None:
        if out.shape != a.shape:
            raise ValueError(
                "out.shape %s must equal a.shape %s." % (tuple(out.shape), tuple(a.shape))
            )
        if out.dtype != a.dtype:
            raise TypeError("out.dtype (%s) must equal a.dtype (%s)." % (out.dtype, a.dtype))
        if not out.is_cuda:
            raise TypeError("out must be a CUDA tensor.")
        if out.device != a.device:
            raise ValueError("out.device (%s) must equal a.device (%s)." % (out.device, a.device))
        if not out.is_contiguous():
            raise TypeError("out must be a contiguous tensor.")
        out_cp = _torch_to_cupy(out)
        result_t = out
    else:
        result_t = torch.empty_like(a)
        out_cp = _torch_to_cupy(result_t)

    # Wrap inputs in CuPy views over the same storage
    a_cp = _torch_to_cupy(a)
    b_cp = _torch_to_cupy(b)

    # Call the existing kernel
    from . import multiply as _ck_multiply

    _ck_multiply(a_cp, b_cp, out=out_cp, stream=stream)

    return result_t


def multiply_into(a, b, out):
    """Write ``a * b`` into ``out`` and return ``out``.

    This is the explicit "multiply into a caller-allocated buffer" form of
    :func:`multiply`.  It is a thin alias for ``multiply(a, b, out=out)``.
    """
    if out is None:
        raise TypeError("multiply_into requires an 'out' buffer")
    return multiply(a, b, out=out)
