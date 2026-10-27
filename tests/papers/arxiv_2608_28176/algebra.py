"""Exact sparse arithmetic for Zhilina 2608.28176v1; no array dependencies.

Indices in dictionaries are mathematical, zero based. Only this adapter
translates to CayleyKernel's signed, one based basis labels.
"""
from fractions import Fraction
from functools import lru_cache
from itertools import product

from CayleyKernel import basis, twist


def e(i):
    return {i: 1}


def scale(x, c):
    return {i: a * c for i, a in x.items() if a * c}


def add(*xs):
    out = {}
    for x in xs:
        for i, a in x.items():
            out[i] = out.get(i, 0) + a
    return {i: a for i, a in out.items() if a}


def neg(x):
    return scale(x, -1)


def conj(x):
    return {i: a if i == 0 else -a for i, a in x.items()}


def pair(a, b, half):
    return add(a, {i + half: c for i, c in b.items()})


def parts(x, half):
    return ({i: a for i, a in x.items() if i < half},
            {i - half: a for i, a in x.items() if i >= half})


def line(x):
    """Canonical projective line; zero has no projective representative."""
    if not x:
        raise ValueError('zero is not a graph vertex')
    pivot = x[min(x)]
    return tuple(sorted(scale(x, Fraction(1, pivot)).items()))


@lru_cache(None)
def recursive_basis(i, j, gammas):
    """Independent Definition 2.4 oracle, including nonunit parameters.

    Four branches implement (a,b)(c,d)=(ac+gamma*conj(d)*b, da+b*conj(c)).
    No twist, basis, or closed sign law is used here.
    """
    if not gammas:
        return 1
    half = 1 << (len(gammas) - 1)
    u, v = i >= half, j >= half
    a, b = i % half, j % half
    if not u and not v:
        return recursive_basis(a, b, gammas[:-1])
    if not u and v:
        return recursive_basis(b, a, gammas[:-1])
    if u and not v:
        return (-1 if b else 1) * recursive_basis(a, b, gammas[:-1])
    return gammas[-1] * (-1 if b else 1) * recursive_basis(b, a, gammas[:-1])


class Algebra:
    def __init__(self, n, gammas=None, reference=False):
        self.n = n
        self.dim = 1 << n
        self.gammas = tuple(gammas if gammas is not None else [-1] * n)
        assert len(self.gammas) == n
        self.reference = reference
        if not reference and any(g != -1 for g in self.gammas):
            raise ValueError('CayleyKernel supports the main sequence only')

    def mul(self, x, y):
        out = {}
        for i, a in x.items():
            for j, b in y.items():
                assert 0 <= i < self.dim and 0 <= j < self.dim
                if self.reference:
                    k, s = i ^ j, recursive_basis(i, j, self.gammas)
                else:
                    label = basis(i + 1, j + 1)
                    k, s = abs(label) - 1, (1 if label > 0 else -1)
                out[k] = out.get(k, 0) + a * b * s
        return {i: c for i, c in out.items() if c}

    def assoc(self, x, y, z):
        return add(self.mul(self.mul(x, y), z), neg(self.mul(x, self.mul(y, z))))

    def antiassoc(self, x, y, z):
        return add(self.mul(self.mul(x, y), z), self.mul(x, self.mul(y, z)))

    def strong(self, x, y):
        return not self.assoc(x, x, y) and not self.assoc(y, y, x)

    def norm(self, x):
        p = self.mul(x, conj(x))
        assert not any(i for i in p), p
        return p.get(0, 0)

    def dot(self, x, y):
        return Fraction(self.norm(add(x, y)) - self.norm(x) - self.norm(y), 2)

    def tilde(self, x):
        return self.mul(x, e(self.dim // 2))

    def orth(self, x, y):
        """Paper's graph orthogonality: both products zero, NOT dot=0."""
        return not self.mul(x, y) and not self.mul(y, x)


def rank(columns):
    """Exact sparse column elimination over Q (also gives rank over R)."""
    pivots = {}
    for column in columns:
        v = {i: Fraction(c) for i, c in column.items() if c}
        while v:
            i = min(v)
            if i not in pivots:
                pivots[i] = scale(v, 1 / v[i])
                break
            v = add(v, scale(pivots[i], -v[i]))
    return len(pivots)


def nullity(A, x, left=False):
    # Kernel of y -> xy (or y -> yx), not just a search over sparse partners.
    return A.dim - rank([A.mul(e(i), x) if left else A.mul(x, e(i))
                         for i in range(A.dim)])


def vertices(n):
    m = 1 << n
    return [(i, j, s) for i in range(1, m) for j in range(m) for s in (-1, 1)]


def vector(v, half):
    i, j, s = v
    return pair(e(i), scale(e(j), s), half)


def graph(A):
    """All pure basis-pair candidates, retaining isolated nonvertices/loops.

    A four-term product can vanish only if i XOR j = k XOR l: this follows
    directly from the output supports, without assuming Lemma 6.2 or 6.16.
    Both sign choices and both orders are evaluated. Graph edges exclude
    loops, but orthogonality sets retain them for self-orthogonality claims.
    """
    half = A.dim // 2
    vs = vertices(A.n - 1)
    adjacency = {v: set() for v in vs}
    for v in vs:
        i, j, s = v
        x = vector(v, half)
        for k, t in product(range(1, half), (-1, 1)):
            w = (k, i ^ j ^ k, t)
            if A.orth(x, vector(w, half)):
                adjacency[v].add(w)
    return adjacency


def witness(A):
    """First deterministic basis-pair zero product with four distinct components."""
    half = A.dim // 2
    for i, j, k in product(range(1, half), repeat=3):
        l = i ^ j ^ k
        if l == 0 or len({i, j, k, l}) < 4:
            continue
        for s in (-1, 1):
            x, y = pair(e(i), e(j), half), pair(e(k), scale(e(l), s), half)
            if not A.mul(x, y):
                return e(i), e(j), e(k), scale(e(l), s)
    raise ValueError('no witness at this dimension')


def double_hexagon(B, a, b, c, d):
    """Theorem 4.11's f_0..f_9 and their tildes, in the doubled algebra."""
    m = B.dim
    ac, ad = B.mul(a, c), B.mul(a, d)
    f = [e(0), pair(a, b, m), pair(c, neg(d), m), pair(ac, ad, m),
         pair(ac, neg(ad), m), pair(c, d, m), pair(a, neg(b), m),
         pair({}, B.mul(a, b), m), pair({}, B.mul(d, c), m),
         pair({}, B.mul(ac, ad), m)]
    A = Algebra(B.n + 1)
    return f, [A.tilde(x) for x in f]
