# SPDX-License-Identifier: Apache-2.0
"""Section 4 over the ordinary real CD sequence.

Exact rational determinant polynomials; numerical representatives of real roots
and singular spheres. This is a research verifier, not a certified general solver.
"""
import numpy as np
import sympy as sp
from tests.papers.desktop_batch.common import mul, add, scale, e, conj


def dense(x, dim):
    return np.array([float(x.get(i, 0)) for i in range(dim)])


def sparse(x):
    return {i: v for i, v in enumerate(x) if v != 0}


def linear_matrix(linear, dim):
    return sp.Matrix(dim, dim, lambda i, j: linear(e(j)).get(i, 0))


def evaluate(q, linear, r):
    return add(mul(q, q), linear(q), r)


def elimination(G, r):
    """Uncancelled D, Di and denominator-cleared E0 (ordinary signature)."""
    t = sp.Symbol('t')
    m = G.rows - 1
    M = 2*t*sp.eye(m) + G[1:, 1:]
    rhs = -sp.Matrix([r.get(i, 0) for i in range(1, m+1)]) - t*G[1:, 0]
    D = sp.Poly(M.det(method='berkowitz'), t)
    Ds = []
    for i in range(m):
        Mi = M.copy()
        Mi[:, i] = rhs
        Ds.append(sp.Poly(Mi.det(method='berkowitz'), t))
    P = (sp.Poly(t*t + G[0, 0]*t + r.get(0, 0), t)*D*D
         - sum((p*p for p in Ds), sp.Poly(0, t))
         + sum((G[0, i+1]*p*D for i, p in enumerate(Ds)), sp.Poly(0, t)))
    return t, D, Ds, P


def singular_slice(G, r, t, tol=1e-9):
    """Intersect affine imaginary solutions with the sphere from E0R."""
    G = np.array(G).astype(float)
    dim = len(G)
    M = 2*t*np.eye(dim-1) + G[1:, 1:]
    rhs = -dense(r, dim)[1:] - t*G[1:, 0]
    p = np.linalg.lstsq(M, rhs, rcond=None)[0]
    if np.linalg.norm(M@p-rhs) > tol*(1+np.linalg.norm(rhs)):
        return {'kind': 'inconsistent', 'representatives': []}
    _, s, vh = np.linalg.svd(M)
    rank = np.count_nonzero(s > tol*max(1, s[0] if len(s) else 0))
    N = vh[rank:].T
    c = G[0, 1:]/2
    center = p + N@(N.T@(c-p))
    rho = t*t + G[0, 0]*t + r.get(0, 0) + c@c
    radius2 = float(rho - (c-center)@(c-center))
    if radius2 < -tol:
        kind, reps = 'empty', []
    elif abs(radius2) <= tol:
        kind, reps = 'point', [np.r_[t, center]]
    elif N.shape[1]:
        kind = 'sphere'
        reps = [np.r_[t, center + sign*np.sqrt(radius2)*N[:, j]]
                for j in range(N.shape[1]) for sign in (-1, 1)]
    else:
        kind, reps = 'empty', []
    return dict(kind=kind, radius2=radius2, nullity=N.shape[1],
                sphere_dimension=N.shape[1]-1, representatives=reps)


def real_candidates(G, r):
    """Find isolated candidates plus singular-slice representatives.

    Exact real-root isolation precedes float64 linear solves. Repeated roots
    remain distinct only by value. Always check the original residual afterward.
    """
    t, D, Ds, P = elimination(G, r)
    # Remove every denominator factor: its trace is handled separately.
    regular = P
    while True:
        common = sp.gcd(regular, D)
        if common.degree() == 0:
            break
        regular = regular.exquo(common)
    Gf = np.array(G).astype(float)
    roots = []
    for root in sp.polys.polytools.intervals(regular, eps=sp.Rational(1, 10**16)):
        (lo, hi), _ = root
        value = float((lo+hi)/2)
        v = np.linalg.solve(2*value*np.eye(G.rows-1)+Gf[1:, 1:],
                            -dense(r, G.rows)[1:]-value*Gf[1:, 0])
        roots.append(np.r_[value, v])
    slices = []
    for (lo, hi), _ in sp.polys.polytools.intervals(D, eps=sp.Rational(1, 10**16)):
        item = singular_slice(G, r, float((lo+hi)/2))
        slices.append(item)
        roots.extend(item['representatives'])
    return roots, slices, (D, Ds, P)
