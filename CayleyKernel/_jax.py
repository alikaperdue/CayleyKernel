# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Optional JAX interop adapter for CayleyKernel.

This module exposes ``CayleyKernel.jax.multiply`` — a thin adapter that lets
``jax.Array`` inputs call the existing CuPy CUDA kernel.

Forward pass and VJP
--------------------
Both are implemented via :func:`jax.pure_callback`, which JAX traces as a
single primitive node.  The callback executes on the host; the underlying
CuPy kernel runs on the GPU via ``CayleyKernel.multiply``.

**VJP derivation (closed form).**

The Cayley-Dickson product ``f(a, b) = a * b`` is *bilinear*: for fixed
``b`` the map ``a ↦ a*b`` is linear in ``a``, and for fixed ``a`` the map
``b ↦ a*b`` is linear in ``b``.  By linearity in each factor the Jacobian
with respect to each argument is a single Cayley-Dickson product with the
other operand held fixed::

    ∂(a*b)/∂a = right multiplication by b   (δa ↦ δa * b)
    ∂(a*b)/∂b = left  multiplication by a   (δb ↦ a * δb)

Because the product is bilinear but *not* symmetric, the transpose of the
Jacobian (which is what the VJP needs) is obtained by conjugating the other
operand (Cayley-Dickson conjugation negates every component except the
leading real one).  In VJP form, for an output cotangent ``g`` of the same
shape and dtype::

    vjp_a(g) = g * conj(b)    # one Cayley-Dickson product
    vjp_b(g) = conj(a) * g    # one Cayley-Dickson product

No associativity is assumed.  At dim ≥ 8 the algebra is non-associative and
the cyclic law fails (see ``scripts/ASSOCIATIVITY.md``), but each VJP term
is a *single* product — never a triple product — so the bilinear derivation
remains exact at every dimension.

**JIT traceability.**

``jax.pure_callback`` is designed to be traceable: there is no host-side
``.item()``, no Python control flow on traced values, and the dtype is
fixed per call.  ``jax.jit(lambda a, b: ck.jax.multiply(a, b))`` works.

**dim ≤ 256 (v1).**

The dense kernel is O(dim²) per product.  JAX's tracing cost does not
amortise over such small workloads at large dim, so dimensions above 256
raise ``NotImplementedError`` in the JAX path.  Use the native CuPy API
(``CayleyKernel.multiply``) for large dimensions.

**dtypes (v1).**

``float32``, ``float64``, ``complex64``.  Both operands must share the same
dtype (the kernel is homogeneous).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import is_power_of_two
from . import multiply as _ck_multiply

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_DIM = 256

_SUPPORTED_JAX_DTYPES = (jnp.float32, jnp.float64, jnp.complex64)

_JAX_TO_NUMPY = {
    jnp.float32: np.float32,
    jnp.float64: np.float64,
    jnp.complex64: np.complex64,
}

_UNSUPPORTED_DTYPE_MSG = "CayleyKernel.jax.multiply supports float32, float64, complex64 (got %s)."

# ---------------------------------------------------------------------------
# Forward pass (raw, used by both the public multiply and the VJP)
# ---------------------------------------------------------------------------


def _multiply_pure(a, b):
    """Host callback: call ``CayleyKernel.multiply`` on concrete arrays.

    ``a`` and ``b`` are concrete Python objects (numpy arrays for the CPU
    JAX backend, device arrays for GPU backends).  We convert to NumPy
    (D2H copy if on GPU) and call the CuPy-backed kernel, which re-uploads
    to the GPU.  For v1 this is the accepted trade-off; a future version
    may use ``cupy.cuda.UnownedMemory`` over the JAX buffer for zero-copy.
    """
    a_np = np.asarray(a)
    b_np = np.asarray(b)
    result = _ck_multiply(a_np, b_np)
    return np.asarray(result)


def _validate(a, b):
    """Static shape / dtype validation.  Safe to call during tracing."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)

    if a.ndim < 1 or b.ndim < 1:
        raise ValueError(
            "CayleyKernel.jax.multiply: inputs must have at least one axis; "
            "the last axis is the algebra dimension (2^n)."
        )
    if a.shape != b.shape:
        raise ValueError(
            "CayleyKernel.jax.multiply: a.shape %s != b.shape %s. "
            "Both operands must have identical shapes in v1." % (tuple(a.shape), tuple(b.shape))
        )
    if a.dtype != b.dtype:
        raise TypeError(
            "CayleyKernel.jax.multiply requires homogeneous dtypes "
            "(got %s and %s).  Cast one operand with jnp.asarray(<dtype>) first."
            % (a.dtype, b.dtype)
        )
    dim = int(a.shape[-1])
    if not is_power_of_two(dim):
        raise ValueError("Algebra dimension (last axis = %d) must be a power of two (2^n)." % dim)
    if dim > _MAX_DIM:
        raise NotImplementedError(
            "CayleyKernel.jax.multiply: dim=%d exceeds the v1 limit of %d. "
            "The dense kernel is O(dim^2) and JAX's tracing cost does not "
            "amortise at this size.  Use the native CuPy API "
            "(CayleyKernel.multiply) for large dimensions." % (dim, _MAX_DIM)
        )
    if a.dtype not in _SUPPORTED_JAX_DTYPES:
        raise TypeError(_UNSUPPORTED_DTYPE_MSG % a.dtype)
    return a, b


def _multiply_impl(a, b):
    """Raw forward: dispatch through :func:`jax.pure_callback`.

    This is the function wrapped by :func:`jax.custom_vjp`.  It is also
    called directly inside the VJP to compute the two bilinear terms.
    """
    a, b = _validate(a, b)
    return jax.pure_callback(
        _multiply_pure,
        jax.ShapeDtypeStruct(tuple(a.shape), a.dtype),
        a,
        b,
    )


# ---------------------------------------------------------------------------
# VJP
# ---------------------------------------------------------------------------


def _conj(x):
    """Cayley-Dickson conjugation (negate all but the first component)."""
    return jnp.concatenate([x[..., :1], -x[..., 1:]], axis=-1)


def _multiply_fwd(a, b):
    """Forward rule for :func:`jax.custom_vjp`.

    Runs the Cayley-Dickson product (the actual forward computation) and
    returns the output together with the residuals needed by the backward
    rule.  The residuals are simply the two input factors ``a`` and ``b``;
    the VJP terms are re-derived from them in :func:`_multiply_bwd`.
    """
    out = _multiply_impl(a, b)
    return out, (a, b)


def _multiply_bwd(residuals, g):
    """Backward rule for :func:`jax.custom_vjp`.

    Closed form (derived from bilinearity; see module docstring)::

        vjp_a(g) = g * conj(b)      (single Cayley-Dickson product)
        vjp_b(g) = conj(a) * g      (single Cayley-Dickson product)

    No associativity is assumed.  Each term is a single product, so the
    formula is exact at every dimension including non-associative dim ≥ 8
    (see ``scripts/ASSOCIATIVITY.md`` for the cyclic-law failures that a
    triple-product derivation would run into).
    """
    a, b = residuals
    # g has the same shape and dtype as the forward output.
    # Both terms are single Cayley-Dickson products — no associativity needed.
    vjp_a = _multiply_impl(g, _conj(b))  # g * conj(b)  (right-multiply cotangent by conj(b))
    vjp_b = _multiply_impl(_conj(a), g)  # conj(a) * g  (left-multiply conj(a) by g)
    return (vjp_a, vjp_b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Build the custom_vjp-wrapped function once at module level.
# This is the object that ``CayleyKernel.jax.multiply`` points to.
_multiply_cvp = jax.custom_vjp(_multiply_impl)
_multiply_cvp.defvjp(_multiply_fwd, _multiply_bwd)


def multiply(a, b, out=None):
    """Cayley-Dickson doubling product for ``jax.Array`` inputs.

    Parameters
    ----------
    a, b : jax.Array
        Both must have identical shape ``(batch..., dim)`` where ``dim`` is
        a power of two and ≤ 256.  Supported dtypes: ``float32``,
        ``float64``, ``complex64``.  Both operands must share the same dtype.
    out : None
        Accepted for API symmetry with the CuPy interface.  Must be ``None``;
        JAX does not support in-place operations.

    Returns
    -------
    jax.Array
        The Cayley-Dickson product, same shape and dtype as the inputs.

    Notes
    -----
    **VJP** (bilinear, no associativity assumed)::

        vjp_a(g) = g * conj(b)
        vjp_b(g) = conj(a) * g

    where ``conj`` is Cayley-Dickson conjugation (negate every component
    except the leading real one).  See the module docstring for the full
    derivation.

    **JIT**: ``jax.jit(lambda a, b: ck.jax.multiply(a, b))`` works.  The
    forward and VJP are both traced as ``jax.pure_callback`` primitives.

    **dim ≤ 256** in v1.  See module docstring for the rationale.
    """
    if out is not None:
        raise NotImplementedError(
            "CayleyKernel.jax.multiply: out= is not supported in v1. "
            "JAX does not support in-place operations."
        )
    return _multiply_cvp(a, b)
