"""Lupercio--Uribe: finite cochain models and U(1) phase identities."""
from fractions import Fraction as Q
from itertools import combinations, product
import math
import unittest
from .common import add, e, mul, phase, close, gpu_replay

try:
    import pytest
    pytestmark = pytest.mark.cpu
except ImportError:
    pass


def delta(cochain, vertices, degree):
    return {s: sum((-1)**i * cochain.get(s[:i] + s[i+1:], 0)
                   for i in range(len(s)))
            for s in combinations(vertices, degree + 2)}


class Sections(unittest.TestCase):
    def test_01_intro_character_additivity(self):
        # R/Z character represented by unit complex numbers; compare to
        # independent Python complex arithmetic, not just twist vs basis.
        for a, b in product([Q(-2, 7), Q(0), Q(3, 5)], repeat=2):
            x, y = phase(2*math.pi*float(a)), phase(2*math.pi*float(b))
            close(self, mul(x, y), phase(2*math.pi*float(a+b)))
            z = complex(x[0], x[1]) * complex(y[0], y[1])
            close(self, mul(x, y), {0: z.real, 1: z.imag})

    def test_02_1_differential_squared_and_character_boundary(self):
        vs = tuple(range(6))
        for n in (1, 2, 3):
            c = {s: sum(s)+1 for s in combinations(vs, n+1)}
            h = {s: Q(sum(s)+2, 7) for s in combinations(vs, n)}
            w = {s: Q(sum(s)+3, 5) for s in combinations(vs, n+1)}
            dc, dh, dw = delta(c, vs, n), delta(h, vs, n-1), delta(w, vs, n)
            middle = {s: w[s]-c[s]-dh[s] for s in c}
            dm = delta(middle, vs, n)
            self.assertTrue(all(v == 0 for v in delta(dc, vs, n+1).values()))
            self.assertTrue(all(dw[s]-dc[s]-dm[s] == 0 for s in dc))
            # A cocycle w=c+delta(h) gives h(boundary S)=w(S) mod Z.
            for s in c:
                self.assertEqual((dh[s] - (c[s]+dh[s])) % 1, 0)
        # The q-1 truncation has middle -c-delta(h), and zero form.
        self.assertTrue(all(-dc[s] - delta({t: -c[t]-dh[t] for t in c}, vs, n)[s] == 0 for s in dc))

    def test_02_2_finite_group_cocycle(self):
        # A nontrivial normalized coboundary on cyclic groups. This tests the
        # Cech/group cocycle algebra, not the BD-CS isomorphism.
        for order in (2, 3, 5, 11):
            f = [Q(g*g, order*order) for g in range(order)]
            rho = lambda g,h: (f[g]+f[h]-f[(g+h)%order]) % 1
            for g,h,k in product(range(order), repeat=3):
                self.assertEqual((rho(h,k)-rho((g+h)%order,k)+rho(g,(h+k)%order)-rho(g,h)) % 1, 0)

    def test_03_string_connection_gluing_and_group_action(self):
        # Exact root-of-unity groupoid characters, then surface integral
        # additivity in radians (the normalization on PDF p.13).
        roots = [e(0), e(1), {0:-1}, {1:-1}]
        for g,h in product(range(4), repeat=2):
            self.assertEqual(mul(roots[g], roots[h]), roots[(g+h)%4])
        for a,b in product([-.8, .3, 1.2], repeat=2):
            close(self, mul(phase(a), phase(b)), phase(a+b))
            close(self, mul(phase(a), phase(-a)), e(0))
            # Stokes compatibility model: boundary integral is b-a.
            close(self, mul(phase(a), phase(b-a)), phase(b))

    def test_04_extension_complex_planes_at_high_levels(self):
        from CayleyKernel import twist
        for n in (4, 8, 16, 32, 128, 1024, 4096):
            u = (1 << (n-1)) + 1
            self.assertEqual(twist(u, u), -1)
            self.assertEqual(mul(e(u), e(u)), {0:-1})
            close(self, mul(phase(.3,u), phase(-.7,u)), phase(-.4,u))

    def test_05_gpu_phase_products(self):
        gpu_replay(self, [(e(1), e(1)), ({0:3,1:4}, {0:5,1:-2})], 2)

    def test_06_gpu_native_complex_holonomy(self):
        import numpy as np
        import CayleyKernel as CK
        from tests.papers.type_replay import require_gpu, host
        require_gpu(self)
        angles=np.array([-.8,0.,.3,1.2])
        for dtype,tol in ((np.complex64,2e-6),(np.complex128,1e-14)):
            a=np.exp(1j*angles).astype(dtype)[:,None,None]
            b=np.exp(1j*angles).astype(dtype)[None,:,None]
            result=host(CK.multiply(a,b))
            self.assertEqual(result.dtype,np.dtype(dtype))
            np.testing.assert_allclose(result[...,0],a[...,0]*b[...,0],rtol=tol,atol=tol)
            inverse=host(CK.multiply(a,np.conjugate(a)))
            np.testing.assert_allclose(inverse,1,rtol=tol,atol=tol)
