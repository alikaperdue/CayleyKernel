# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Batched Cayley-Dickson algebra helpers for NumPy, CuPy, and lists.

Coefficient operations run on the input's device without a compiled kernel.
Products use the public GPU ``multiply`` implementation.
"""

import operator
import warnings


def _array(a, xp=None, floating=False, signed=False):
    from . import cp, np

    if xp is None:
        xp = cp if cp is not None and isinstance(a, cp.ndarray) else np
    if xp is None:
        raise ImportError("CayleyKernel algebra helpers require NumPy.")
    a = xp.asarray(a)
    if a.ndim == 0 or not a.shape[-1] or a.shape[-1] & (a.shape[-1] - 1):
        raise ValueError("Inputs must have a power-of-two algebra dimension on the last axis.")
    is_bfloat16 = a.dtype.name == "bfloat16"
    if a.dtype.kind not in "biufc" and not is_bfloat16:
        raise TypeError("Algebra coefficients must be numeric.")
    # Avoid unsigned negation and integer overflow in norms and products.
    if (floating and a.dtype.kind in "biu") or (signed and a.dtype.kind in "bu"):
        a = a.astype(xp.float64)
    elif floating and (a.dtype == xp.float16 or is_bfloat16):
        a = a.astype(xp.float32)
    return a, xp


def _return(value, *inputs):
    from . import cp, np

    if cp is not None and any(isinstance(a, cp.ndarray) for a in inputs):
        return value
    if np is not None and any(isinstance(a, np.ndarray) for a in inputs):
        return value
    return value.tolist()


def _pair(a, b):
    from . import cp

    xp = cp if cp is not None and any(isinstance(v, cp.ndarray) for v in (a, b)) else None
    x, xp = _array(a, xp, floating=True)
    y, _ = _array(b, xp, floating=True)
    if x.shape[-1] != y.shape[-1]:
        raise ValueError("Algebra dimension mismatch: last axes must match.")
    return x, y, xp


def conjugate(x):
    """Negate imaginary basis coefficients, preserving the real coefficient.

    Complex scalar coefficients are not complex-conjugated. The last axis is
    the algebra dimension; leading axes are batches. Returns a new container.
    Unsigned and boolean coefficients promote to float64 to allow negation.
    """
    a, _ = _array(x, signed=True)
    result = -a
    result[..., 0] = a[..., 0]
    return _return(result, x)


def norm_sq(a):
    """Return sum(a_k**2) along the last axis, removing that axis.

    This is the squared Euclidean norm for real coefficients and the scalar
    coefficient of a * conjugate(a). For complex coefficients it is a
    bilinear quadratic form, not a Hermitian norm, and can vanish for nonzero a.
    Integers promote to float64; float16/bfloat16 promote to float32.
    """
    x, xp = _array(a, floating=True)
    return _return(xp.sum(x * x, axis=-1), a)


def inverse(a):
    """Return conjugate(a) / norm_sq(a), raising ZeroDivisionError at zero norm.

    In dimensions >= 16, emit RuntimeWarning: zero divisors can have positive
    norm, so this two-sided element inverse does not guarantee cancellation
    or unique solutions of multiplication equations. No zero-divisor test is
    performed. If any batch member has zero norm, the entire call fails.
    """
    x, xp = _array(a, floating=True)
    squared = xp.sum(x * x, axis=-1, keepdims=True)
    if bool(xp.any(squared == 0)):
        raise ZeroDivisionError("Cannot invert an element with zero norm_sq.")
    if x.shape[-1] >= 16:
        warnings.warn(
            "Dimensions >= 16 admit zero divisors even with positive norm_sq; "
            "the conjugate/norm inverse does not guarantee cancellation.",
            RuntimeWarning, stacklevel=2,
        )
    result = -x
    result[..., 0] = x[..., 0]
    return _return(result / squared, a)


def right_div(a, b):
    """Return a * inverse(b) using GPU multiply and batch broadcasting.

    This need not solve x*b=a in dimensions >= 16; see inverse.
    """
    from . import multiply

    x, y, _ = _pair(a, b)
    return _return(multiply(x, inverse(y)), a, b)


def left_div(a, b):
    """Return inverse(a) * b using GPU multiply and batch broadcasting.

    This need not solve a*x=b in dimensions >= 16; see inverse.
    """
    from . import multiply

    x, y, _ = _pair(a, b)
    return _return(multiply(inverse(x), y), a, b)


def power(a, n):
    """Integer powers by binary exponentiation using GPU multiply.

    Zero yields the identity, including for zero input. Negative powers use
    inverse and inherit its zero-norm errors and high-dimensional warning.
    A single integer exponent applies to every member of a batch.
    """
    from . import multiply

    n = operator.index(n)
    x, xp = _array(a, floating=True)
    if n < 0:
        x = inverse(x)
        n = -n
    result = xp.zeros_like(x)
    result[..., 0] = 1
    while n:
        if n & 1:
            result = multiply(result, x)
        n >>= 1
        if n:
            x = multiply(x, x)
    return _return(result, a)


def exp(a):
    """Return exp(a0) * (cos(theta) + sinc(theta) * imag(a)).

    Here theta**2 = sum(a_k**2, k>0), and sinc(theta)=sin(theta)/theta
    with value 1 at zero. Works at every doubling level. Complex coefficients
    use the analytic quadratic form, consistent with norm_sq.
    """
    x, xp = _array(a, floating=True)
    theta = xp.sqrt(xp.sum(x[..., 1:] * x[..., 1:], axis=-1, keepdims=True))
    # Use a safe denominator rather than a ufunc ``where`` keyword, which
    # is not supported by every CuPy version.
    denominator = xp.where(theta == 0, 1, theta)
    scale = xp.where(theta == 0, 1, xp.sin(theta) / denominator)
    result = x.copy()
    result[..., 0] = 0
    result *= scale
    result[..., :1] = xp.cos(theta)
    result *= xp.exp(x[..., :1])
    return _return(result, a)


def inner(a, b):
    """Scalar coefficient of a * conjugate(b), with batch broadcasting.

    For real coefficients this is the Euclidean dot product. Complex
    coefficients use the bilinear dot product (no scalar conjugation), which
    can itself be complex. The algebra dimension must match.
    """
    x, y, xp = _pair(a, b)
    return _return(xp.sum(x * y, axis=-1), a, b)


def real(a):
    """Extract the e0 coefficient, removing the last axis (also for complex a)."""
    x, _ = _array(a)
    return _return(x[..., 0].copy(), a)


def imag(a):
    """Return a copy with e0 set to zero, preserving the algebra dimension."""
    x, _ = _array(a)
    result = x.copy()
    result[..., 0] = 0
    return _return(result, a)
