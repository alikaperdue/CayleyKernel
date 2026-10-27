"""Exact CPU products through CayleyKernel; independent doubling and analysis tools.

CPU products are NOT a fallback implementation of the GPU multiply API.
Paper octonion indices map to binary indices via PAPER_TO_KERNEL.
"""
from fractions import Fraction
from itertools import product

import numpy as np
import CayleyKernel as CK

PAPER_TO_KERNEL = (0, 1, 2, 4, 3, 5, 6, 7)
# Signed one-based PAPER labels, transcribed from printed page 4.
PAPER_TABLE = (
    (-1, 5, 6, -3, -4, -8, 7),
    (-5, -1, 7, 2, 8, -4, -6),
    (-6, -7, -1, -8, 2, 3, 5),
    (3, -2, 8, -1, -7, 6, -4),
    (4, -8, -2, 7, -1, -5, 3),
    (8, 4, -3, -6, 5, -1, -2),
    (-7, 6, -5, 4, -3, 2, -1),
)


def e(i, d=8):
    return [int(j == i) for j in range(d)]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def scale(a, s):
    return [x * s for x in a]


def conj(a):
    return [a[0]] + [-x for x in a[1:]]


def norm(a):
    return sum(x*x for x in a)


def mul(a, b):
    """Exact bilinear extension of signed one-based basis()."""
    assert len(a) == len(b)
    out = [0] * len(a)
    for i, x in enumerate(a):
        if not x:
            continue
        for j, y in enumerate(b):
            if y:
                label = CK.basis(i+1, j+1)
                out[abs(label)-1] += x*y*(1 if label > 0 else -1)
    return out


def recursive(a, b):
    """Independent recursive doubling, no module signs or product calls."""
    if len(a) == 1:
        return [a[0]*b[0]]
    h = len(a)//2
    x, y, u, v = a[:h], a[h:], b[:h], b[h:]
    return (add(recursive(x, u), scale(recursive(conj(v), y), -1))
            + add(recursive(v, x), recursive(y, conj(u))))


def inverse(a):
    return scale(conj(a), Fraction(1, norm(a)))


def q(a):
    """q0 = conjugate(z)/|z|^8, exact for rational octonions."""
    return scale(conj(a), Fraction(1, norm(a)**4))


def dq(a, j):
    """Analytic partial derivative, including x0 when j=0."""
    return add(scale(conj(e(j)), Fraction(1, norm(a)**4)),
               scale(conj(a), Fraction(-8*a[j], norm(a)**5)))


def transformed(z, lam, mu, product_fn=mul):
    return product_fn(product_fn(mu, q(product_fn(lam, product_fn(z, mu)))), lam)


def tensor():
    t = np.zeros((8, 8, 8))
    for i, j in product(range(8), repeat=2):
        label = CK.basis(i+1, j+1)
        t[i, j, abs(label)-1] = 1 if label > 0 else -1
    return t


TENSOR = tensor()


def nmul(a, b):
    """Vectorized CPU numerical product using the module's basis table."""
    return np.einsum('...i,...j,ijk->...k', a, b, TENSOR)


def nq(a):
    a = np.asarray(a, dtype=float)
    c = a.copy()
    c[..., 1:] *= -1
    return c / np.sum(a*a, axis=-1)[..., None]**4


def ndq(a, j):
    a = np.asarray(a, dtype=float)
    r2 = np.sum(a*a, axis=-1)
    return np.asarray(conj(e(j)))/r2[..., None]**4 - 8*a[..., j, None]*nq(a)/r2[..., None]


def lattice(cutoff, scales):
    grid = np.array(list(product(range(-cutoff, cutoff+1), repeat=8)), dtype=float)
    return grid[np.any(grid, axis=1)] * scales


def zeta(z, points, correction=-1):
    """Finite series; -1 is explicitly proposed correction, +1 is printed."""
    z = np.asarray(z, dtype=float)
    out = nq(z) + np.sum(nq(z+points)-nq(points), axis=0)
    for j in range(1, 8):
        v = z[j]*np.eye(8)[0] - z[0]*np.eye(8)[j]
        out += correction*nmul(v, np.sum(ndq(points, j), axis=0))
    return out


def wp(z, points, j):
    return ndq(z, j) + np.sum(ndq(z+points, j)-ndq(points, j), axis=0)


def jacobian(f, z, h=1e-5):
    eye = np.eye(8)*h
    return np.array([(f(z+v)-f(z-v))/(2*h) for v in eye])


def dirac(jac, right=False):
    return sum((nmul(jac[j], np.eye(8)[j]) if right else
                nmul(np.eye(8)[j], jac[j])) for j in range(8))
