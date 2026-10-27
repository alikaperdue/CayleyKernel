"""Exact public-basis arithmetic and independent tools for the radius paper.

Elements use zero-based coordinate lists. Signed one-based labels are confined
here. No CPU function is presented as execution of the GPU multiply() routine.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np
import CayleyKernel as CK


def e(i, dim):
    return [int(j == i) for j in range(dim)]


def add(a, b):
    return [x+y for x, y in zip(a, b)]


def scale(a, s):
    return [x*s for x in a]


def conj(a):
    return a[:1]+[-x for x in a[1:]]


def norm2(a):
    return sum(x*x for x in a)


def mul(a, b):
    """Bilinear extension of CayleyKernel.basis, exact on rationals."""
    if len(a) != len(b) or not len(a) or len(a) & (len(a)-1):
        raise ValueError('equal power-of-two dimensions required')
    out = [0]*len(a)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y:
                    label = CK.basis(i+1, j+1)
                    out[abs(label)-1] += x*y*(1 if label > 0 else -1)
    return out


@lru_cache(None)
def recursive_sign(i, j, dim):
    """Independent four-case expansion of the paper's doubling definition."""
    if dim == 1:
        return 1
    half = dim//2
    a, b = i % half, j % half
    if i < half and j < half:
        return recursive_sign(a, b, half)
    if i < half <= j:
        return recursive_sign(b, a, half)
    if j < half <= i:
        return (-1 if b else 1)*recursive_sign(a, b, half)
    return -(-1 if b else 1)*recursive_sign(b, a, half)


def reference(a, b):
    """Oracle that never calls twist or basis."""
    out = [0]*len(a)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x and y:
                out[i ^ j] += x*y*recursive_sign(i, j, len(a))
    return out


def power(a, k, product_fn=mul):
    if k < 0:
        raise ValueError('nonnegative exponent required')
    result = e(0, len(a))
    for _ in range(k):
        result = product_fn(result, a)
    return result


def quadratic_power(a, k):
    """Independent recurrence in R[t]/(t²-2*a0*t+N(a))."""
    c, d = 1, 0  # c*1+d*a
    for _ in range(k):
        c, d = -norm2(a)*d, c+2*a[0]*d
    return add(scale(e(0, len(a)), c), scale(a, d))


def bracketed_powers(a, k):
    """Enumerate every full binary parenthesization, retaining duplicates."""
    @lru_cache(None)
    def values(n):
        if n == 1:
            return (tuple(a),)
        return tuple(tuple(mul(list(x), list(y)))
                     for i in range(1, n)
                     for x in values(i) for y in values(n-i))
    return values(k)


def polynomial(a, coefficients):
    """Ascending real coefficients, evaluated by powers, including constant."""
    out = [0]*len(a)
    for k, c in enumerate(coefficients):
        out = add(out, scale(power(a, k), c))
    return out


def polymul(a, b):
    out = [0]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


def remainder(a, b):
    out = [Fraction(x) for x in a]
    while len(out) >= len(b):
        offset = len(out)-len(b)
        factor = out[-1]/b[-1]
        for j, x in enumerate(b):
            out[offset+j] -= factor*x
        while out and not out[-1]:
            out.pop()
    return out


def minimal_polynomial(a):
    # If a has a nonzero imaginary component, 1 and a are independent.
    return [-a[0], 1] if not any(a[1:]) else [norm2(a), -2*a[0], 1]


def radius_from_roots(a):
    return float(max(abs(z) for z in np.roots(list(reversed(minimal_polynomial(a))))))


def lp(a, p):
    if p == math.inf:
        return max(map(abs, a))
    # Scaled evaluation avoids overflowing large powers unnecessarily.
    largest = max(map(abs, a))
    if not largest:
        return 0.0
    return float(largest)*sum((float(abs(x)/largest))**p for x in a)**(1/p)


def unit_power_samples(a, exponents):
    """Multiply exact integer/rational powers, then normalize each saved value.

    Scaling after exact arithmetic prevents transverse roundoff amplification
    in higher algebras from contaminating the Gelfand experiment.
    """
    n = norm2(a)
    radius = math.sqrt(float(n))
    current = e(0, len(a))
    out = {}
    for k in range(1, max(exponents)+1):
        current = mul(current, a)
        if k in exponents:
            denominator = n**(k//2)
            odd_scale = radius if k % 2 else 1
            out[k] = [float(Fraction(x, denominator))/odd_scale for x in current]
    return radius, out


def bumped(a, kappa=3):
    """Eq (3.4), f=Euclidean norm and V=real scalar line."""
    return math.sqrt(float(norm2(a)))*(kappa if not any(a[1:]) else 1)


def paper_sqrt(a):
    """Literal Lemma 5.1 formula, with real and zero branches included."""
    out = [0.0]*len(a)
    if not any(a[1:]):
        if a[0] >= 0:
            out[0] = math.sqrt(a[0])
        elif len(a) > 1:
            out[1] = math.sqrt(-a[0])
        else:
            raise ValueError('negative real has no root in A0')
        return out
    radius = math.sqrt(float(norm2(a)))
    denominator = math.sqrt(2*a[0]+2*radius)
    return scale(add(a, scale(e(0, len(a)), radius)), 1/denominator)


def stable_sqrt(a):
    """Same root, reformulated to avoid cancellation near negative reals."""
    if not any(a[1:]):
        return paper_sqrt(a)
    imaginary = math.hypot(*a[1:])
    radius = math.hypot(a[0], imaginary)
    if a[0] >= 0:
        real = math.sqrt((radius+a[0])/2)
        return [real]+[x/(2*real) for x in a[1:]]
    imag_magnitude = math.sqrt((radius-a[0])/2)
    real = imaginary/(2*imag_magnitude)
    return [real]+[x/imaginary*imag_magnitude for x in a[1:]]


@dataclass(frozen=True)
class Angle:
    """Exact member p*pi+r of Q*pi+Q. phi(p*pi+r)=r.

    This is a RESTRICTION of a possible Hamel-basis extension, not an
    executable everywhere-defined discontinuous additive real function.
    No phi value is ever inferred from a floating-point angle.
    """
    pi: Fraction = Fraction(0)
    one: Fraction = Fraction(0)

    def __add__(self, other):
        return Angle(self.pi+other.pi, self.one+other.one)

    def __neg__(self):
        return Angle(-self.pi, -self.one)

    def times(self, scalar):
        scalar = Fraction(scalar)
        return Angle(self.pi*scalar, self.one*scalar)

    def value(self):
        return float(self.pi)*math.pi+float(self.one)

    def phi(self):
        return self.one


def complex_argument(angle):
    """Return exact angle mod 2*pi; float only chooses the separated branch."""
    turns = math.floor(angle.value()/(2*math.pi))
    result = angle+Angle(Fraction(-2*turns), Fraction(0))
    if not -1e-12 <= result.value() < 2*math.pi+1e-12:
        raise ArithmeticError('angle reduction failed')
    return result


def quaternion_argument(angle):
    """The paper's folded angle in [0,pi]; exact phi tracking retained."""
    result = complex_argument(angle)
    return -result+Angle(Fraction(2), Fraction(0)) if result.value() > math.pi else result


def angle_vector(angle, dim=2, log_radius=0):
    radius = math.exp(log_radius)
    return [radius*math.cos(angle.value()), radius*math.sin(angle.value())]+[0]*(dim-2)
