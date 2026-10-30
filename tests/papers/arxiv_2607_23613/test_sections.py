# SPDX-License-Identifier: Apache-2.0
"""Tests ordered by the paper's sections, then explicitly labeled extensions."""
import random
import math
import os
import itertools
import unittest
from fractions import Fraction

MIN_LEVEL = 4
MAX_LEVEL = 22

try:
    import numpy as np
    import sympy as sp
except ModuleNotFoundError as exc:
    raise unittest.SkipTest('paper companion requires NumPy and SymPy: ' + str(exc))
import CayleyKernel as CK
from tests.papers.desktop_batch.common import (
    basis_product, mul, add, neg, scale, conj, e, recursive_basis, gpu_replay,
)
from .method import dense, sparse, linear_matrix, evaluate, elimination, singular_slice, real_candidates

try:
    import pytest
    pytestmark = pytest.mark.cpu
except ImportError:
    pass


EVIDENCE = {}


def inverse(x):
    return scale(conj(x), Fraction(1, sum(v*v for v in x.values())))


def random_vector(rng, dim, count=5):
    return {i: rng.choice([-2, -1, 1, 2]) for i in rng.sample(range(dim), min(count, dim))}


class Sections(unittest.TestCase):
    def assertResidual(self, q, linear, r, tol=1e-7):
        q = sparse(q) if not isinstance(q, dict) else q
        residual = evaluate(q, linear, r)
        self.assertLess(max(map(abs, residual.values()), default=0), tol)

    def test_01_section1_nonmonic_rootless_commutator(self):
        # Scalar coordinate ix-xi+1 is identically 1, on every basis column.
        for dim in (4, 8, 16, 32):
            for j in range(dim):
                self.assertEqual(add(mul(e(1), e(j)), neg(mul(e(j), e(1)))).get(0, 0), 0)

    def test_02_section2_basis_convention(self):
        for n in range(1, 7):
            for i in range(1 << n):
                for j in range(1 << n):
                    s = recursive_basis(i, j, (-1,)*n)
                    self.assertEqual(CK.twist(i, j), s)
                    self.assertEqual(basis_product(i+1, j+1), s*((i^j)+1))
                    self.assertEqual(basis_product(-i-1, j+1), -s*((i^j)+1))

    def test_03_section2_quadratic_flexibility_powers(self):
        rng = random.Random(23613)
        for n in range(1, 9):
            for _ in range(12):
                x, y = [random_vector(rng, 1 << n) for _ in range(2)]
                norm = sum(c*c for c in x.values())
                self.assertEqual(mul(x, conj(x)), {0: norm})
                self.assertEqual(add(mul(x, x), scale(x, -2*x.get(0, 0)), {0: norm}), {})
                self.assertEqual(mul(mul(x, y), x), mul(x, mul(y, x)))
                self.assertEqual(mul(mul(x, x), mul(x, x)), mul(x, mul(x, mul(x, x))))

    def test_04_section2_structure_losses(self):
        self.assertNotEqual(mul(e(1), e(2)), mul(e(2), e(1)))
        self.assertNotEqual(mul(mul(e(1), e(2)), e(4)), mul(e(1), mul(e(2), e(4))))
        rng = random.Random(4)
        witnesses = []
        for dim in (8, 16):
            failures = 0
            for _ in range(100):
                x, y = [random_vector(rng, dim) for _ in range(2)]
                failures += mul(mul(x, x), y) != mul(x, mul(x, y))
            witnesses.append(failures)
        self.assertEqual(witnesses[0], 0)
        self.assertGreater(witnesses[1], 0)

    def test_05_section3_left_evaluation_not_homomorphism(self):
        # f=x, g=i, q=j: (fg)(q)=ij whereas f(q)g(q)=ji.
        self.assertNotEqual(mul(e(1), e(2)), mul(e(2), e(1)))
        # (a*x)(x-b), evaluated at b: a*(b*b)-(a*b)*b.
        rng = random.Random(31)
        found = False
        for dim in (8, 16):
            for _ in range(50):
                a, b = [random_vector(rng, dim) for _ in range(2)]
                residual = add(mul(a, mul(b, b)), neg(mul(mul(a, b), b)))
                if dim == 8:
                    self.assertEqual(residual, {})
                else:
                    found |= bool(residual)
        self.assertTrue(found)

    def test_06_section3_brackets_and_linear_map(self):
        i, j, x = e(1), e(2), e(4)
        values = [mul(mul(i, j), x), mul(i, mul(x, j)), mul(mul(i, x), j)]
        self.assertGreater(len({tuple(sorted(v.items())) for v in values}), 1)
        linear = lambda q: add(mul(mul(e(1), mul(mul(q, e(2)), e(4))), e(1)),
                               scale(mul(add(e(1), e(2)), q), 34))
        G = linear_matrix(linear, 16)
        q = {0: 2, 5: 3, 13: -1}
        self.assertEqual(G*sp.Matrix([q.get(i, 0) for i in range(16)]),
                         sp.Matrix([linear(q).get(i, 0) for i in range(16)]))

    def test_07_section4_coordinate_reduction(self):
        rng = random.Random(40)
        for dim in (2, 4, 8, 16, 32):
            a, b, c, r, q = [random_vector(rng, dim) for _ in range(5)]
            linear = lambda x: add(mul(a, mul(x, b)), mul(c, x))
            G = linear_matrix(linear, dim)
            v = sp.Matrix([q.get(i, 0) for i in range(dim)])
            expected = G*v + sp.Matrix([r.get(i, 0) for i in range(dim)])
            expected[0] += v[0]**2-sum(z*z for z in v[1:])
            for i in range(1, dim): expected[i] += 2*v[0]*v[i]
            self.assertEqual(list(expected), [evaluate(q, linear, r).get(i, 0) for i in range(dim)])

    def test_08_section4_claim41_and_lemma42(self):
        # x²+2x+1: consistent singular trace -1, all Cramer minors vanish.
        G = 2*sp.eye(4)
        t, D, Ds, P = elimination(G, {0: 1})
        self.assertEqual(D.degree(), 3)
        self.assertEqual(D.LC(), 8)
        self.assertEqual(D.eval(-1), 0)
        for p in Ds: self.assertEqual(p.eval(-1), 0)
        reduced_D = D.exquo(sp.Poly(t+1, t))
        self.assertEqual(reduced_D.degree(), 2)
        self.assertLessEqual(P.exquo(sp.Poly((t+1)**2, t)).degree(), 2+2*(3-1))
        roots, slices, _ = real_candidates(G, {0: 1})
        self.assertEqual(len(roots), 1)
        self.assertEqual(slices[0]['kind'], 'point')
        self.assertResidual(roots[0], lambda x: scale(x, 2), {0: 1})
        # Nonzero Cramer numerator as well: 2*i*x + 2*j at t=0.
        linear = lambda x: scale(mul(e(1), x), 2)
        t, D, Ds, _ = elimination(linear_matrix(linear, 4), {2: 2})
        self.assertEqual(D.eval(0), 0)
        self.assertTrue(any(not p.is_zero for p in Ds))
        for p in Ds:
            self.assertEqual(p.eval(0), 0)
            self.assertTrue(p.rem(sp.Poly(t, t)).is_zero)

    def test_09_section4_inconsistent_singular_trace(self):
        G = sp.zeros(4)
        item = singular_slice(G, {1: 1}, 0)
        self.assertEqual(item['kind'], 'inconsistent')
        roots, _, (_, _, P) = real_candidates(G, {1: 1})
        self.assertEqual(len(roots), 2)
        self.assertLessEqual(P.degree(), 8)
        for q in roots: self.assertResidual(q, lambda x: {}, {1: 1})

    def test_10_section41_spheres_empty_point_and_pair(self):
        for dim in (2, 4, 8, 16, 32, 64):
            G = sp.zeros(dim)
            for r0, kind in ((-1, 'empty'), (0, 'point'), (1, 'sphere')):
                item = singular_slice(G, {0: r0}, 0)
                self.assertEqual(item['kind'], kind)
                self.assertEqual(item['nullity'], dim-1)
                for q in item['representatives']:
                    self.assertResidual(q, lambda x: {}, {0: r0})
            self.assertEqual(singular_slice(G, {0: 1}, 0)['sphere_dimension'], dim-2)
        # An actual linear term 2ix gives singular affine lines in H.
        linear = lambda x: scale(mul(e(1), x), 2)
        G = linear_matrix(linear, 4)
        for r0, kind in ((-2, 'empty'), (-1, 'point'), (0, 'sphere')):
            item = singular_slice(G, {0: r0}, 0)
            self.assertEqual(item['kind'], kind)
            self.assertEqual(item['nullity'], 1)
            for q in item['representatives']: self.assertResidual(q, linear, {0: r0})

    def test_11_section41_unplanted_quadratics(self):
        # No root is planted: constants and coefficients sampled independently.
        rng = random.Random(46)
        for dim in (2, 4, 8):
            for _ in range(2):
                a, b, c, r = [random_vector(rng, dim, 3) for _ in range(4)]
                linear = lambda x: add(mul(a, mul(x, b)), mul(c, x))
                G = linear_matrix(linear, dim)
                roots, slices, (_, _, P) = real_candidates(G, r)
                self.assertTrue(roots)
                self.assertLessEqual(P.degree(), 2*dim)
                self.assertLessEqual(len(slices), dim-1)
                for q in roots: self.assertResidual(q, linear, r, 1e-5)

    def test_12_section41_remark47_real_exception(self):
        # Exact minimum of x²+1 over R; complex unit supplies a root at n=1.
        x = sp.Symbol('x', real=True)
        self.assertEqual(sp.solveset(x*x+1, x, domain=sp.S.Reals), sp.S.EmptySet)
        self.assertEqual(add(mul(e(1), e(1)), e(0)), {})

    def test_13_section5_example53(self):
        one, i, j, ell = e(0), e(1), e(2), e(4)
        a, b, c, d = one, i, add(one, scale(i, 2), j), add(one, ell)
        y = mul(add(i, mul(i, j)), ell)
        x = mul(neg(i), y)
        lam = add(a, mul(b, x))
        self.assertEqual(x, mul(add(one, neg(j)), ell))
        linear = lambda z: add(mul(b, mul(mul(inverse(b), z), a)), neg(mul(b, mul(d, mul(inverse(b), z)))))
        self.assertEqual(evaluate(y, linear, neg(mul(b, c))), {})
        self.assertEqual(add(mul(y, y), mul(ell, y), scale(one, 2), neg(i), neg(mul(i, j))), {})
        self.assertEqual(add(c, mul(d, x)), mul(x, lam))
        EVIDENCE['example_5_3'] = {'y': y, 'x': x, 'lambda': lam}
        roots, _, _ = real_candidates(linear_matrix(linear, 8), neg(mul(b, c)))
        self.assertTrue(roots)
        for root in roots:
            xx = mul(inverse(b), sparse(root))
            ll = add(a, mul(b, xx))
            self.assertLess(np.linalg.norm(dense(add(c, mul(d, xx), neg(mul(xx, ll))), 8)), 1e-7)

    def test_14_section5_general_matrices_and_b_zero(self):
        rng = random.Random(51)
        for _ in range(2):
            a, b, c, d = [random_vector(rng, 8, 3) for _ in range(4)]
            bi = inverse(b)
            linear = lambda y: add(mul(b, mul(mul(bi, y), a)), neg(mul(b, mul(d, mul(bi, y)))))
            r = neg(mul(b, c))
            roots, _, _ = real_candidates(linear_matrix(linear, 8), r)
            self.assertTrue(roots)
            for y in roots:
                x = mul(bi, sparse(y))
                lam = add(a, mul(b, x))
                self.assertLess(np.linalg.norm(dense(add(c, mul(d, x), neg(mul(x, lam))), 8)), 1e-5)
            # b=0: eigenvector (0,1), lambda=d, regardless of a,c.
            self.assertEqual(mul(d, e(0)), d)

    def test_15_section6_sylvester_and_back_substitution(self):
        rng = random.Random(61)
        for lower in (False, True):
            for repeated in (False, True):
                diag = [{0: i*3, i+1: 1} for i in range(3)]
                if repeated: diag[2] = diag[0]
                B = [[diag[i] if i == j else random_vector(rng, 8, 3) if i < j else {}
                      for j in range(3)] for i in range(3)]
                for left in (False, True):
                    for target in range(3):
                        q = e(0) if left else random_vector(rng, 8, 3)
                        lam = diag[target] if left else mul(mul(inverse(q), diag[target]), q)
                        k = 0 if repeated and target == 2 else target
                        v = [{} for _ in range(3)]
                        v[k] = q
                        for i in range(k-1, -1, -1):
                            op = (lambda z: mul(add(diag[i], neg(lam)), z)) if left else (lambda z: add(mul(diag[i], z), neg(mul(z, lam))))
                            M = np.array(linear_matrix(op, 8)).astype(float)
                            rhs = neg(add(*(mul(B[i][j], v[j]) for j in range(i+1, k+1))))
                            self.assertEqual(np.linalg.matrix_rank(M), 8)
                            v[i] = sparse(np.linalg.solve(M, dense(rhs, 8)))
                        BB, vv = (list(reversed([list(reversed(row)) for row in B])), list(reversed(v))) if lower else (B, v)
                        for i in range(3):
                            lhs = add(*(mul(BB[i][j], vv[j]) for j in range(3)))
                            rhs = mul(lam, vv[i]) if left else mul(vv[i], lam)
                            self.assertLess(np.linalg.norm(dense(add(lhs, neg(rhs)), 8)), 1e-8)
                        if not left:
                            M = linear_matrix(lambda z: add(mul(diag[target], z), neg(mul(z, lam))), 8)
                            self.assertLess(np.linalg.matrix_rank(np.array(M).astype(float), tol=1e-9), 8)

    def test_16_section6_standard_vector_is_not_enough(self):
        # B=[[0,1],[0,i]]: (0,1) is not a left eigenvector for i.
        self.assertNotEqual(mul(e(0), e(0)), {})
        x = neg(e(1))
        self.assertEqual(e(0), mul(e(1), x))
        self.assertEqual(e(1), mul(e(1), e(0)))

    def test_17_extension_high_level_quadratic_families(self):
        rng = random.Random(2607)
        for n in (4, 8, 16, 32, 64, 128, 1024):
            dim = 1 << n
            for _ in range(16):
                indices = {rng.getrandbits(n) or 1 for _ in range(6)}
                u = {i: rng.choice([-2, -1, 1, 2]) for i in indices}
                s = sum(c*c for c in u.values())
                # x²+s=0, and translated (x-a)²+s=0, use full-width indices.
                self.assertEqual(add(mul(u, u), {0: s}), {})
                a = rng.randint(-7, 7)
                q = add({0: a}, u)
                self.assertEqual(evaluate(q, lambda x: scale(x, -2*a), {0: a*a+s}), {})
                # Every planted general bracketed quadratic keeps its exact root.
                b = {rng.getrandbits(n): 2, rng.getrandbits(n): -1}
                linear = lambda x: mul(b, mul(x, b))
                r = neg(add(mul(q, q), linear(q)))
                self.assertEqual(evaluate(q, linear, r), {})

    def test_18_extension_octonion_proof_fails_in_sedenions(self):
        rng = random.Random(52)
        failures = 0
        for _ in range(100):
            b, x = [random_vector(rng, 16) for _ in range(2)]
            lhs = mul(b, mul(x, mul(b, x)))
            rhs = mul(mul(b, x), mul(b, x))
            if lhs != rhs and failures == 0:
                EVIDENCE['sedenion_artin_witness'] = {'b': b, 'x': x, 'lhs_minus_rhs': add(lhs, neg(rhs))}
            failures += lhs != rhs
        self.assertGreater(failures, 0)
        # Zero divisors invalidate triangular spectra over n>=4.
        pairs = [add(e(i), scale(e(j), s)) for i in range(1, 8) for j in range(8, 16) for s in (-1, 1)]
        witness = next((a, x) for a in pairs for x in pairs if not mul(a, x))
        a, x = witness
        EVIDENCE["sedenion_zero_divisor"] = {"a": a, "x": x}
        EVIDENCE["sedenion_artin_identity_failures_of_100"] = failures
        self.assertTrue(a); self.assertTrue(x)
        self.assertEqual(mul(a, x), {})  # 1x1 [a] has left/right eigenvalue 0 != a.

    def test_19_extension_unplanted_deep_roots(self):
        # Solve x²+2x+r=0 within the complex subalgebra generated by r.
        # r is sampled independently, not defined from an already known root.
        rng = random.Random(1946)
        rows = []
        for n in (4, 8, 16, 32, 64, 128, 1024):
            worst = 0.0
            for _ in range(32):
                r = {0: rng.randint(-10, 10)}
                r.update({rng.getrandbits(n) or 1: rng.choice([-3, -2, 1, 2]) for _ in range(8)})
                imaginary_norm2 = sum(v*v for i, v in r.items() if i)
                w0 = 1-r[0]
                length = math.sqrt(w0*w0+imaginary_norm2)
                a = math.sqrt((length+w0)/2)
                root = {i: -v/(2*a) for i, v in r.items() if i}
                root[0] = a-1
                residual = evaluate(root, lambda x: scale(x, 2), r)
                worst = max(worst, max(map(abs, residual.values()), default=0))
                self.assertResidual(root, lambda x: scale(x, 2), r, 1e-10)
            rows.append({'level': n, 'cases': 32, 'max_absolute_residual': worst})
        EVIDENCE['unplanted_deep_scalar_linear_quadratics'] = rows

    def test_20_section4_finite_field_signature_supplement(self):
        # Independent recursive arithmetic: not native split/finite-field CK.
        from tests.papers.arxiv_2608_28176.algebra import Algebra
        rows = []
        for p in (3, 5):
            for gammas in itertools.product((-1, 1), repeat=2):
                A = Algebra(2, gammas=gammas, reference=True)
                linear = lambda x: add(A.mul(e(1), A.mul(x, e(2))), A.mul(x, e(3)))
                G = [[linear(e(j)).get(i, 0) % p for j in range(4)] for i in range(4)]
                alpha = [A.mul(e(i), e(i))[0] for i in range(1, 4)]
                roots = 0
                traces = set()
                for coords in itertools.product(range(p), repeat=4):
                    q = sparse(coords)
                    actual = add(A.mul(q, q), linear(q), e(0))
                    expected = [sum(G[i][j]*coords[j] for j in range(4)) for i in range(4)]
                    expected[0] += coords[0]**2 + sum(alpha[i-1]*coords[i]**2 for i in range(1, 4)) + 1
                    for i in range(1, 4): expected[i] += 2*coords[0]*coords[i]
                    self.assertEqual([actual.get(i, 0) % p for i in range(4)], [v % p for v in expected])
                    if all(v % p == 0 for v in expected):
                        roots += 1
                        traces.add((2*coords[0]) % p)
                self.assertLessEqual(len(traces), 8)
                rows.append({'p': p, 'gammas': gammas, 'vectors': p**4, 'roots': roots, 'distinct_traces': len(traces)})
        EVIDENCE['finite_field_reference'] = rows

    def test_21_gpu_multiply_replay(self):
        rng = random.Random(19)
        for dim in (2, 4, 8, 16, 64, 256):
            vectors = [(random_vector(rng, dim), random_vector(rng, dim)) for _ in range(12)]
            gpu_replay(self, vectors, dim)

    def test_22_gpu_paper_equations_and_deep_quadratic(self):
        gpu_replay(self, [(e(1), e(2))], 8)  # prerequisite checks, then fail on errors
        def product(a, b):
            return np.asarray(CK.multiply(a, b))
        one, i, j, ell = [dense(e(k), 8) for k in (0, 1, 2, 4)]
        y = product(i+product(i, j), ell)
        x = product(-i, y)
        lam = one+product(i, x)
        np.testing.assert_allclose(product(y, y)+product(ell, y)+2*one-i-product(i, j), 0, atol=1e-12)
        np.testing.assert_allclose(one+2*i+j+product(one+ell, x), product(x, lam), atol=1e-12)
        level = int(os.environ.get('CK_PAPER_GPU_LEVEL', '12'))
        if not MIN_LEVEL <= level <= MAX_LEVEL:
            raise ValueError('CK_PAPER_GPU_LEVEL must be between {} and {}'.format(MIN_LEVEL,MAX_LEVEL))
        rng = random.Random(22)
        for n in sorted({4, 8, level}):
            q = random_vector(rng, 1 << n, 8)
            x = dense(q, 1 << n)
            expected = dense(add(scale(q, 2*q.get(0, 0)), {0: -sum(v*v for v in q.values())}), 1 << n)
            np.testing.assert_allclose(product(x, x), expected, rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
