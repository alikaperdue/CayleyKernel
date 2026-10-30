# SPDX-License-Identifier: Apache-2.0
"""Exact operator arithmetic; transport the paper's Fano convention explicitly."""
from functools import lru_cache
import sympy as s
from tests.papers.desktop_batch.common import basis_product, mul, add, neg, scale, conj, e

# Paper labels e1,...,e7 are NOT CayleyKernel's binary labels.
PAPER = (e(0), e(1), e(2), e(4), e(3), e(6), {7: -1}, e(5))
TRIPLES = ((1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3))


def vector(x, d):
    return s.Matrix([x.get(i, 0) for i in range(d)])


def operator(x, d, right=False):
    return s.Matrix.hstack(*(vector(mul(e(j), x) if right else mul(x, e(j)), d)
                             for j in range(d)))


@lru_cache(None)
def operators(d):
    units = PAPER if d == 8 else tuple(e(i) for i in range(d))
    return ([operator(x, d) for x in units], [operator(x, d, True) for x in units])


def words(gammas):
    out = [s.eye(gammas[0].rows)]
    for g in gammas:
        out += [x*g for x in out]
    return out


def flat(x):
    return x.reshape(x.rows*x.cols, 1)


def span_rank(xs):
    return s.Matrix.hstack(*(flat(x) for x in xs)).rank() if xs else 0


def comm(a,b):
    return a*b-b*a


def matrix_units(d, skew=False):
    if skew:
        return [s.eye(d)[:, i]*s.eye(d)[j, :]-s.eye(d)[:, j]*s.eye(d)[i, :]
                for i in range(d) for j in range(i+1, d)]
    return [s.eye(d)[:, i]*s.eye(d)[j, :] for i in range(d) for j in range(d)]


def restrict(basis, constraint):
    """Solve all linear constraints over Q, retaining an exact spanning basis."""
    cols = [s.Matrix(constraint(x)) for x in basis]
    kernel = s.Matrix.hstack(*cols).nullspace()
    return [sum((c*x for c,x in zip(v,basis)), s.zeros(basis[0].rows)) for v in kernel]


@lru_cache(None)
def centralizer(d, both=False, skew=False):
    left,right = operators(d)
    js = [left[-1]] + ([right[-1]] if both else [])
    return restrict(matrix_units(d,skew), lambda x: s.Matrix.vstack(*(flat(comm(x,j)) for j in js)))


@lru_cache(None)
def phases():
    lo,ro=operators(8); lh,rh=operators(4)
    po=(s.eye(8)-lo[7]*ro[7])/2
    ph=(s.eye(4)-lh[3]*rh[3])/2
    return po,s.eye(8)-po,ph,s.eye(4)-ph


@lru_cache(None)
def lie_model(low=False):
    """Full skew centralizers on O+H+C+R, before the two conditions."""
    bo=centralizer(8,True,True)
    bh=centralizer(4,low,True)
    bc=[operators(2)[0][1]]
    return ([s.diag(x,s.zeros(4),s.zeros(2),s.zeros(1)) for x in bo]
            +[s.diag(s.zeros(8),x,s.zeros(2),s.zeros(1)) for x in bh]
            +[s.diag(s.zeros(8),s.zeros(4),x,s.zeros(1)) for x in bc])


@lru_cache(None)
def physical(low=False):
    jo=operators(8)[0][7]; jh=operators(4)[0][3]; jc=operators(2)[0][1]
    vo=vector({next(iter(PAPER[7])):1},15)
    vh=vector({8+3:1},15)
    def constraints(x):
        a=(jo*x[:8,:8]).trace(); b=(jh*x[8:12,8:12]).trace(); c=(jc*x[12:14,12:14]).trace()
        return s.Matrix.vstack(x*vo, x*vh if low else s.zeros(0,1), s.Matrix([a-b,b-c]))
    return restrict(lie_model(low),constraints)


def embedding():
    """Eq.49, S case: columns O(8), H267(4), C7(2), R(1), extra real(1)."""
    columns = [mul(e(8),e(i)) for i in range(8)]
    columns += [mul(PAPER[5],PAPER[j]) for j in (0,2,6,7)]
    columns += [mul(PAPER[6],PAPER[j]) for j in (0,7)]
    columns += [PAPER[7],e(0)]
    return s.Matrix.hstack(*(vector(x,16) for x in columns))


def hopf(a,b):
    na=sum(c*c for c in a.values()); nb=sum(c*c for c in b.values())
    return na-nb,scale(mul(a,conj(b)),2)


def oracle(i,j,n):
    """Independent iterative implementation of the four recursive doubling cases."""
    sign=1
    for bit in range(n-1,-1,-1):
        h=1<<bit; u,v=bool(i&h),bool(j&h)
        i,j=i&(h-1),j&(h-1)
        if u and j: sign=-sign
        if u and v: sign=-sign
        if v: i,j=j,i
    return sign
