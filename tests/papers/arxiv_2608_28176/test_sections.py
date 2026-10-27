"""Finite, exact checks in paper order. Run with unittest or pytest.

python3 -m unittest -v tests.papers.arxiv_2608_28176.test_sections

All CPU arithmetic uses exact integers/rationals. Tests for mixed signatures
are explicitly reference-only. CUDA tests skip only for missing prerequisites.
"""
import itertools as it
import random
import unittest
from functools import lru_cache
from fractions import Fraction

import CayleyKernel as CK
from tests.papers.type_replay import replay_product
from .algebra import (Algebra, e, add, neg, scale, conj, pair, parts, line,
                      rank, nullity, graph, vector, witness, double_hexagon,
                      recursive_basis)
from .tables import table1, quaternion_table, octonion_table

try:
    import pytest
except ImportError:
    pass
else:
    # CPU marker bypasses the repository's blanket GPU skip, even for the
    # CUDA tests below: those perform their own prerequisite check.
    pytestmark = pytest.mark.cpu


@lru_cache(None)
def ordinary_graph(n):
    return graph(Algebra(n + 1))


def examples(n, limit=12):
    """Deterministic adjacent pairs; no random search or vacuous implications."""
    G = ordinary_graph(n)
    result = []
    for v in sorted(G):
        for w in sorted(G[v]):
            if v < w:
                B = Algebra(n)
                a,b = parts(vector(v, B.dim), B.dim)
                c,d = parts(vector(w, B.dim), B.dim)
                result.append((a,b,c,d))
                if len(result) == limit:
                    return result
    assert result
    return result


def random_vector(rng, dim, pure=False, doubly=False):
    return {i: c for i in range(dim)
            if (c := rng.randint(-2, 2)) and not (i == 0 and (pure or doubly))
            and not (doubly and i == dim // 2)}


class PaperSections(unittest.TestCase):
    def test_02_01_graph_definitions_and_projective_scaling(self):
        A = Algebra(4)
        a,b,c,d = witness(A)
        x,y = pair(a,b,8), pair(c,d,8)
        self.assertTrue(A.orth(x,y))
        self.assertEqual(line(x), line(scale(x, -3)))
        self.assertTrue(A.orth(scale(x, -3), scale(y, Fraction(2,7))))
        self.assertEqual(line(x), line(add(x,x)))
        self.assertRaises(ValueError, line, {})
        # Euclidean perpendicularity does not imply graph orthogonality.
        self.assertEqual(A.dot(e(1),e(2)), 0)
        self.assertFalse(A.orth(e(1),e(2)))
        self.assertNotEqual(line(x),line(y))

    def test_02_02_definition_2_4_twist_basis_and_oracle(self):
        for n in range(7):
            A,R = Algebra(n),Algebra(n,reference=True)
            for i,j in it.product(range(A.dim),repeat=2):
                expected = recursive_basis(i,j,R.gammas)
                self.assertEqual(CK.twist(i,j),expected,(n,i,j))
                for s,t in it.product((-1,1),repeat=2):
                    self.assertEqual(CK.basis(s*(i+1),t*(j+1)),
                                     s*t*expected*((i^j)+1),(n,i,j,s,t))
                self.assertEqual(A.mul(e(i),e(j)),R.mul(e(i),e(j)))
        rng=random.Random(260828176)
        for n in (32,128,1024):
            for _ in range(24):
                i,j=rng.getrandbits(n),rng.getrandbits(n)
                self.assertEqual(CK.twist(i,j),recursive_basis_iterative(i,j,n))

    def test_02_03_01_norm_involution_bilinear_form_and_flexibility(self):
        rng=random.Random(276)
        for n in range(1,6):
            A=Algebra(n)
            B=Algebra(n-1)
            for _ in range(12):
                x,y,z=[random_vector(rng,A.dim) for _ in range(3)]
                a,b=parts(x,A.dim//2)
                self.assertEqual(conj(conj(x)),x)
                self.assertEqual(conj(A.mul(x,y)),A.mul(conj(y),conj(x)))
                self.assertEqual(A.norm(x),sum(c*c for c in x.values()))
                self.assertEqual(A.norm(x),B.norm(a)+B.norm(b))
                self.assertEqual(add(x,conj(x)),scale(e(0),2*x.get(0,0)))
                self.assertEqual(add(A.mul(x,conj(y)),A.mul(y,conj(x))),scale(e(0),2*A.dot(x,y)))
                self.assertEqual(A.dot(x,A.mul(y,z)),A.dot(A.mul(x,conj(z)),y))
                self.assertEqual(A.dot(x,A.mul(y,z)),A.dot(A.mul(conj(y),x),z))
                self.assertFalse(A.assoc(x,y,x))
                self.assertEqual(A.assoc(x,y,z),neg(A.assoc(z,y,x)))
                if n<=3:
                    self.assertFalse(A.assoc(x,x,y))
                    self.assertFalse(A.assoc(y,x,x))
                    self.assertEqual(A.norm(A.mul(x,y)),A.norm(x)*A.norm(y))
        A=Algebra(4)
        a,b,c,d=witness(A)
        x,y=pair(a,b,8),pair(c,d,8)
        # Controls: global alternativity and norm multiplicativity fail in M4.
        self.assertTrue(any(A.assoc(x,x,e(i)) for i in range(16)))
        self.assertNotEqual(A.norm(A.mul(x,y)),A.norm(x)*A.norm(y))

    def test_02_03_02_lemma_2_14_anticentralizers(self):
        A=Algebra(4)
        for a in (add(e(0),e(1)), e(1), add(e(1),e(10))):
            columns=[add(A.mul(a,e(i)),A.mul(e(i),a)) for i in range(16)]
            self.assertEqual(16-rank(columns),0 if a.get(0) else 14)
            if not a.get(0):
                for i in range(1,16):
                    self.assertEqual(not add(A.mul(a,e(i)),A.mul(e(i),a)),A.dot(a,e(i))==0)
        # Lemma 2.14(2) is impossible in the positive definite main sequence.
        S=Algebra(2,[-1,1],reference=True)
        a=add(e(0),e(2))
        self.assertEqual(S.norm(a),0)
        self.assertFalse(add(S.mul(a,conj(a)),S.mul(conj(a),a)))
        self.assertEqual(4-rank([add(S.mul(a,e(i)),S.mul(e(i),a)) for i in range(4)]),1)

    def test_02_03_03_lemma_2_16_conditional_norm(self):
        for n in (3,4,5):
            A=Algebra(n)
            for i,j in it.product(range(A.dim),repeat=2):
                self.assertTrue(A.strong(e(i),e(j)))
                self.assertEqual(A.norm(A.mul(e(i),e(j))),A.norm(e(i))*A.norm(e(j)))

    def test_03_01_lemmas_3_2_to_3_8_hexagon(self):
        for n in (3,4):
            B,A=Algebra(n),Algebra(n+1)
            for a,b,c,d in examples(n):
                ac,da=B.mul(a,c),B.mul(d,a)
                self.assertTrue(all(B.strong(x,y) for x,y in it.product((a,b),(c,d))))
                for x,y in it.product((ac,da),(a,b,c,d)):
                    self.assertTrue(B.strong(x,y))
                self.assertEqual(B.norm(c),B.norm(d))
                h=[pair(a,b,B.dim),pair(c,d,B.dim),pair(conj(ac),da,B.dim),
                   pair(a,neg(b),B.dim),pair(c,neg(d),B.dim),pair(conj(ac),neg(da),B.dim)]
                self.assertEqual(len({line(x) for x in h}),6)
                for i in range(6):
                    self.assertFalse(A.mul(h[i],h[(i+1)%6]),(n,i,h))
                    self.assertEqual(A.norm(h[i]),2)
                # Proposition 3.8, gamma=-1, chi=-1, BOTH alternative expressions.
                self.assertEqual(pair(conj(ac),da,B.dim),pair(B.mul(conj(b),d),neg(B.mul(b,conj(c))),B.dim))
                self.assertEqual(pair(conj(ac),neg(da),B.dim),pair(B.mul(conj(b),d),B.mul(b,conj(c)),B.dim))

    def test_03_01_reference_only_degenerate_cycle_and_zero_norm_branch(self):
        # Corollary 3.7 as written permits a repeated-vertex zero-product sequence.
        # All hypotheses hold, but a six-cycle needs six distinct lines.
        B=Algebra(1,[-1],reference=True)
        A=Algebra(2,[-1,1],reference=True)
        a=b=c=d=e(1)
        self.assertTrue(all(B.strong(x,y) for x,y in it.product((a,b),(c,d))))
        self.assertEqual(B.norm(c),B.norm(d))
        self.assertNotEqual(B.norm(c),0)
        ac,da=B.mul(a,c),B.mul(d,a)
        h=[pair(a,b,2),pair(c,d,2),pair(conj(ac),neg(da),2),
           pair(a,neg(b),2),pair(c,neg(d),2),pair(conj(ac),da,2)]
        self.assertEqual(len({line(x) for x in h}),4)
        for i in range(6):
            self.assertFalse(A.mul(h[i],h[(i+1)%6]))
        # Lemma 3.3/Remark 3.4: n(c)=n(d)=0 allows arbitrary chi.
        B=Algebra(2,[-1,1],reference=True)
        A=Algebra(3,[-1,1,-1],reference=True)
        a=add(e(0),e(2))
        c=conj(a)
        b=d={}
        self.assertEqual(B.norm(c),0)
        self.assertFalse(A.mul(pair(a,b,4),pair(c,d,4)))
        for chi in (-2,0,Fraction(3,7),1):
            self.assertFalse(A.mul(pair(c,d,4),pair(conj(B.mul(a,c)),scale(B.mul(d,a),-chi),4)))

    def test_03_to_04_nonbasis_rational_witness(self):
        B,A=Algebra(3),Algebra(4)
        a,b,c0,d0=witness(A)
        # Rational rotation of a known annihilator and its tilde. Every
        # component has norm one, and c,d now have two terms each.
        c=add(scale(c0,Fraction(3,5)),scale(d0,Fraction(-4,5)))
        d=add(scale(d0,Fraction(3,5)),scale(c0,Fraction(4,5)))
        self.assertTrue(all(B.strong(x,y) for x,y in it.product((a,b),(c,d))))
        self.assertEqual(B.norm(c),1)
        self.assertEqual(B.norm(d),1)
        self.assertFalse(A.mul(pair(a,b,8),pair(c,d,8)))
        f,t=double_hexagon(B,a,b,c,d)
        expected=table1(f,t)
        F=f[:7]+t[:7]
        for i,j in it.product(range(14),repeat=2):
            self.assertEqual(A.mul(F[i],F[j]),expected[i][j])
        self.assertEqual(rank(F),14)

    def test_03_02_lemmas_3_13_3_14_full_annihilator_spaces(self):
        for n in (3,4):
            B,A=Algebra(n),Algebra(n+1)
            for a,b,c,d in examples(n,4):
                x=pair(a,b,B.dim)
                left_candidates=[]
                right_candidates=[]
                orth_candidates=[]
                for i in range(B.dim):
                    c=e(i)
                    left=pair(c,scale(B.mul(B.mul(b,c),a),-Fraction(1,B.norm(a))),B.dim)
                    right=pair(c,scale(B.mul(B.mul(b,conj(c)),conj(a)),-Fraction(1,B.norm(a))),B.dim)
                    lc=B.mul(b,B.mul(c,a))==neg(B.mul(B.mul(b,c),a))
                    rc=B.mul(B.mul(a,c),conj(b))==neg(B.mul(a,B.mul(c,conj(b))))
                    self.assertEqual(not A.mul(left,x),lc)
                    self.assertEqual(not A.mul(x,right),rc)
                    if lc: left_candidates.append(left)
                    if rc: right_candidates.append(right)
                    if lc and i:
                        orth_candidates.append(left)
                        self.assertTrue(A.orth(x,left))
                    d=e(i)
                    c2=scale(B.mul(B.mul(a,d),conj(b)),Fraction(1,B.norm(b)))
                    condition=(B.dot(d,B.mul(b,a))==0 and
                               B.mul(a,B.mul(d,conj(b)))==neg(B.mul(B.mul(a,d),conj(b))))
                    self.assertEqual(A.orth(x,pair(c2,d,B.dim)),condition)
                # Full matrix kernels certify completeness for these examples.
                self.assertEqual(rank(left_candidates),nullity(A,x,left=True))
                self.assertEqual(rank(right_candidates),nullity(A,x))
                self.assertEqual(rank(orth_candidates),nullity(A,x))

    def test_03_02_proposition_3_10_and_corollary_3_16(self):
        A=Algebra(4)
        for a,b,c,d in examples(3):
            B=Algebra(3)
            ac,da=B.mul(a,c),B.mul(d,a)
            h=[pair(a,b,8),pair(c,d,8),pair(conj(ac),da,8),pair(a,neg(b),8),pair(c,neg(d),8),pair(conj(ac),neg(da),8)]
            for i,j in it.product(range(6),repeat=2):
                self.assertEqual(A.orth(h[i],h[j]),(i-j)%6 in (1,5))
        # Nonpure zero-norm conjugate pair: Proposition 3.10(1), reference only.
        S=Algebra(2,[-1,1],reference=True)
        x=add(e(0),e(2))
        self.assertTrue(S.orth(x,conj(x)))
        # Pure self-orthogonality, Proposition 3.5, chi=+1.
        y=add(e(1),e(3))
        self.assertTrue(S.orth(y,y))
        self.assertEqual(S.norm(y),0)

    def test_04_01_lemmas_4_1_4_3_corollaries_4_2_4_4(self):
        for n in (3,4):
            B,A=Algebra(n),Algebra(n+1)
            for a,b,c,d in examples(n):
                x,y=pair(a,b,B.dim),pair(c,d,B.dim)
                self.assertTrue(A.orth(x,y))
                self.assertFalse(A.mul(x,A.tilde(y)))
                self.assertTrue(all(z.get(0,0)==0 for z in (a,b,c,d)))
                self.assertEqual(B.norm(a),B.norm(b))
                self.assertEqual(B.norm(c),B.norm(d))
                self.assertEqual(B.mul(a,c),neg(B.mul(d,b)))
                self.assertEqual(B.mul(d,a),B.mul(b,c))
                for u,v,w,r in ((a,c,b,d),(c,a,d,b)):
                    self.assertEqual(B.assoc(u,v,w),scale(r,2))
                self.assertEqual(B.assoc(a,d,b),scale(c,-2))
                self.assertEqual(B.assoc(c,b,d),scale(a,-2))
                for triple in ((a,c,b),(a,d,b),(b,c,a),(b,d,a),(c,a,d),(c,b,d),(d,a,c),(d,b,c)):
                    self.assertFalse(B.antiassoc(*triple))
                for u,v in it.combinations((a,b,c,d),2):
                    self.assertEqual(B.dot(u,v),0)
                for u,v in it.product((c,d),(e(0),a,b,B.mul(a,b))):
                    self.assertEqual(B.dot(u,v),0)
                self.assertEqual(B.mul(a,c),B.mul(b,d))
                self.assertEqual(B.mul(a,d),neg(B.mul(b,c)))

    def test_04_02_01_lemmas_4_5_4_6_4_10_and_table_1(self):
        for n in (3,4):
            B,A=Algebra(n),Algebra(n+1)
            for a,b,c,d in examples(n):
                ac,ad=B.mul(a,c),B.mul(a,d)
                orthonormal=[e(0),a,b,c,d,ac,ad]
                for i,j in it.product(range(7),repeat=2):
                    self.assertEqual(B.dot(orthonormal[i],orthonormal[j]),int(i==j))
                self.assertEqual(A.mul(pair(a,b,B.dim),pair(c,neg(d),B.dim)),scale(pair(ac,ad,B.dim),2))
                self.assertEqual(A.mul(pair(a,neg(b),B.dim),pair(c,d,B.dim)),scale(pair(ac,neg(ad),B.dim),2))
                f,t=double_hexagon(B,a,b,c,d)
                F=f[:7]+t[:7]
                expected=table1(f,t)
                for i,j in it.product(range(14),repeat=2):
                    self.assertEqual(A.mul(F[i],F[j]),expected[i][j],(n,i,j))
                    self.assertEqual(A.dot(F[i],F[j]),(1 if i in (0,7) else 2) if i==j else 0)
                self.assertEqual(rank(F),14)
                self.assertTrue(A.assoc(f[1],f[1],f[2]))

    def test_04_02_02_description_4_8_double_hexagon_graph(self):
        for n in (3,4):
            B,A=Algebra(n),Algebra(n+1)
            for abcd in examples(n):
                f,t=double_hexagon(B,*abcd)
                order=[1,5,3,6,2,4]
                corners=[[f[i],neg(t[i])] for i in order]
                self.assertEqual(len({line(x) for corner in corners for x in corner}),12)
                edges=0
                for i,j in it.product(range(6),repeat=2):
                    for x,y in it.product(corners[i],corners[j]):
                        adjacent=(i-j)%6 in (1,5)
                        self.assertEqual(A.orth(x,y),adjacent)
                        edges+=adjacent
                self.assertEqual(edges//2,24)  # six K2,2 blocks, degree four.
                self.assertEqual(rank([x for corner in corners for x in corner]),12)

    def check_table(self,A,images,table):
        for i,j in it.product(range(len(images)),repeat=2):
            coefficient,k=table[i][j]
            self.assertEqual(A.mul(images[i],images[j]),scale(images[k],coefficient),(i,j))

    def test_05_01_lemma_5_1_corollary_5_2(self):
        rng=random.Random(501)
        for n in (3,4,5):
            A=Algebra(n)
            for i in range(A.dim):
                expected=e(i+A.dim//2) if i<A.dim//2 else neg(e(i-A.dim//2))
                self.assertEqual(A.tilde(e(i)),expected)
            for _ in range(12):
                a,b=[random_vector(rng,A.dim,doubly=True) for _ in range(2)]
                self.assertEqual(A.tilde(A.tilde(a)),neg(a))
                self.assertEqual(A.mul(A.tilde(a),b),neg(A.tilde(A.mul(a,b))))
                self.assertEqual(A.dot(A.tilde(a),a),0)
                self.assertEqual(not add(A.mul(A.tilde(a),b),A.mul(A.tilde(b),a)),A.dot(a,b)==0)
                self.assertEqual(not add(neg(A.mul(a,b)),A.mul(A.tilde(b),A.tilde(a))),A.dot(A.tilde(a),b)==0)

    def test_05_02_lemma_5_3_table_2(self):
        for n in (3,4,5):
            A=Algebra(n)
            for a in (e(1),add(e(1),e(3)),add(e(2),scale(e(A.dim-1),2))):
                images=[e(0),a,e(A.dim//2),A.tilde(a)]
                self.check_table(A,images,quaternion_table(A.norm(a),1))
                for x,y,z in it.product(images,repeat=3):
                    self.assertFalse(A.assoc(x,y,z))
    def test_05_04_lemma_5_6_table_3(self):
        for n in (3,4,5):
            A=Algebra(n)
            for i,j in ((1,2),(1,A.dim-1),(3,A.dim//2)):
                a,b=e(i),scale(e(j),2)
                self.assertTrue(A.strong(a,b))
                self.assertEqual(A.dot(a,b),0)
                images=[e(0),a,b,A.mul(a,b)]
                self.check_table(A,images,quaternion_table(A.norm(a),A.norm(b)))
                for x,y,z in it.product(images,repeat=3):
                    self.assertFalse(A.assoc(x,y,z))

    def test_05_03_corollary_5_5_strong_alternativity_polarization(self):
        for n in (3,4,5):
            A=Algebra(n)
            t=e(A.dim//2)
            for i in range(A.dim):
                self.assertTrue(A.strong(t,e(i)))
                for j in range(i+1,A.dim):
                    # [x,x,t]=0 for ALL x follows by polarization on this basis.
                    self.assertFalse(add(A.assoc(e(i),e(j),t),A.assoc(e(j),e(i),t)))

    def test_05_05_lemma_5_8_table_4_alternativity_and_moufang(self):
        rng=random.Random(508)
        for n in (3,4,5):
            A=Algebra(n)
            a,b=e(1),scale(e(2),2)
            t=e(A.dim//2)
            self.assertTrue(A.strong(a,b))
            self.assertTrue(all(A.dot(b,x)==0 for x in (e(0),a,t,A.tilde(a))))
            ab=A.mul(a,b)
            images=[e(0),a,b,ab,t,A.tilde(a),A.tilde(b),A.tilde(ab)]
            self.assertEqual(rank(images),8)
            self.check_table(A,images,octonion_table(A.norm(a),A.norm(b),1))
            # Check the claimed homomorphism independently against Definition 2.4
            # at parameters (-norm(a),-norm(b),-1), not just the printed table.
            source=Algebra(3,[-A.norm(a),-A.norm(b),-1],reference=True)
            for i,j in it.product(range(8),repeat=2):
                expected={}
                for k,c in source.mul(e(i),e(j)).items():
                    expected=add(expected,scale(images[k],c))
                self.assertEqual(A.mul(images[i],images[j]),expected)
                self.assertFalse(A.assoc(images[i],images[i],images[j]))
                for k in range(j+1,8):
                    self.assertFalse(add(A.assoc(images[j],images[k],images[i]),A.assoc(images[k],images[j],images[i])))
            for _ in range(12):
                def combination():
                    return add(*(scale(x,rng.randint(-2,2)) for x in images))
                x,y,z=[combination() for _ in range(3)]
                self.assertFalse(A.assoc(x,x,y))
                self.assertFalse(A.assoc(y,x,x))
                self.assertEqual(A.mul(z,A.mul(x,A.mul(z,y))),A.mul(A.mul(A.mul(z,x),z),y))
                self.assertEqual(A.mul(x,A.mul(z,A.mul(y,z))),A.mul(A.mul(A.mul(x,z),y),z))
                lhs=A.mul(A.mul(z,x),A.mul(y,z))
                self.assertEqual(lhs,A.mul(A.mul(z,A.mul(x,y)),z))
                self.assertEqual(lhs,A.mul(z,A.mul(A.mul(x,y),z)))

    def test_05_06_corollary_5_9(self):
        for n in (3,4,5):
            B,A=Algebra(n-1),Algebra(n)
            for i,j in it.product(range(B.dim),repeat=2):
                x,y=e(i),e(j)
                self.assertTrue(B.strong(x,y))
                self.assertEqual(A.mul(x,A.tilde(y)),A.tilde(B.mul(y,x)))
                self.assertEqual(A.mul(A.tilde(x),y),A.tilde(B.mul(x,conj(y))))
                self.assertEqual(A.mul(A.tilde(x),A.tilde(y)),neg(B.mul(conj(y),x)))

    def test_05_07_reference_only_split_degenerate_homomorphisms(self):
        S=Algebra(3,[-1,-1,1],reference=True)
        a=add(e(1),e(5))
        self.assertEqual(S.norm(a),0)
        self.assertEqual(S.tilde(a),a)
        images=[e(0),a,e(4),S.tilde(a)]
        self.check_table(S,images,quaternion_table(0,-1))
        self.assertEqual(rank(images),3)  # Remark 5.4: nontrivial kernel.
        # Remark 5.7: nonzero orthogonal strongly alternating split-octonions
        # with ab=0; these generate a degenerate associative image.
        b=add(e(2),neg(e(6)))
        self.assertTrue(S.strong(a,b))
        self.assertEqual(S.dot(a,b),0)
        self.assertFalse(S.mul(a,b))
        self.check_table(S,[e(0),a,b,{}],quaternion_table(0,0))

    def test_06_01_01_lemmas_6_2_6_3_proposition_6_5_components(self):
        for n in (1,2,3,4,5):
            A,B=Algebra(n+1),Algebra(n)
            G=ordinary_graph(n)
            active={v for v in G if G[v]}
            while active:
                start=min(active)
                stack=[start]
                component=set()
                while stack:
                    v=stack.pop()
                    if v in component: continue
                    component.add(v)
                    stack.extend(G[v]-component)
                active-=component
                labels={v[0]^v[1] for v in component}
                self.assertEqual(len(labels),1)
                # Proposition 6.5: main sequence chi=-1, gamma=-1.
                k=next(iter(labels))
                self.assertEqual(-B.norm(e(k)),-1)
                for i,j,s in component:
                    self.assertEqual(B.mul(e(i),scale(e(j),s)).keys(),e(k).keys())
            if n<3:
                self.assertFalse(any(G.values()))
            if n==3:
                self.assertEqual(sum(bool(ws) for ws in G.values()),84)
                self.assertEqual(sum(len(ws) for ws in G.values())//2,168)
                self.assertEqual(sorted({len(ws) for ws in G.values()}),[0,4])

    def test_06_01_02_remark_6_6_four_types(self):
        for n in (2,3,4):
            m,h=1<<n,1<<(n-1)
            families=[set() for _ in range(4)]
            for i in range(1,m):
                families[0].add((i,i))
                families[1].add((i,i^h))
            for c in range(1,h):
                families[2].update(((c,0),(c+h,h),(h,c+h)))
                families[3].update(((c+h,0),(c,h),(h,c)))
            for a,b in it.permutations(range(1,h),2):
                families[2].update(((a,b),(a+h,b+h)))
                families[3].update(((a,b+h),(a+h,b)))
            for i in range(1,m):
                for j in range(m):
                    k=i^j
                    category=0 if k==0 else 1 if k==h else 2 if k<h else 3
                    self.assertIn((i,j),families[category])
                    self.assertFalse(any((i,j) in f for q,f in enumerate(families) if q!=category))

    def test_06_02_01_proposition_6_7_all_pure_basis_pairs(self):
        for n in (2,3,4):
            A=Algebra(n+1)
            m=1<<n
            # Include all candidates, not only known edges (both implications).
            for i,j,k in it.product(range(1,m),repeat=3):
                l=i^j^k
                if l==0: continue
                for s,t in it.product((-1,1),repeat=2):
                    x=pair(e(i),scale(e(j),s),m)
                    y=pair(e(k),scale(e(l),t),m)
                    transformed=pair(scale(e(l),t),neg(e(k)),m)
                    self.assertEqual(A.orth(x,y),A.orth(x,transformed))

    def test_06_02_02_proposition_6_8_independent_eight_units(self):
        # n=4 is the first possible component dimension for eight independent
        # tilde-related units. Exhaust all ordered lower-half four-tuples.
        A,B=Algebra(5),Algebra(4)
        successes=failures=0
        for ids in it.permutations(range(1,8),4):
            a,b,c,d=map(e,ids)
            units=[a,b,c,d]+[B.tilde(x) for x in (a,b,c,d)]
            self.assertEqual(rank(units),8)
            x,y=pair(a,b,16),pair(c,d,16)
            tx,ty=pair(B.tilde(a),B.tilde(b),16),pair(B.tilde(c),B.tilde(d),16)
            result=A.orth(x,y)
            self.assertEqual(result,A.orth(x,ty))
            self.assertEqual(result,A.orth(tx,ty))
            successes+=result
            failures+=not result
        self.assertGreater(successes,0)
        self.assertGreater(failures,0)

    def test_06_02_03_lemmas_6_9_to_6_12_and_6_15_neighbors(self):
        for n in (2,3,4):
            A,B=Algebra(n+1),Algebra(n)
            G=ordinary_graph(n)
            m,h=B.dim,B.dim//2
            for i in range(1,m):
                for s in (-1,1):
                    self.assertFalse(G[(i,i,s)])  # Lemma 6.9, chi=-1.
                    self.assertFalse(G[(i,0,s)])  # Lemma 6.11, chi=-1.
            # Lemma 6.10: tilde-related components, exact neighbor sets.
            units=range(1,h)
            for i in units:
                a=e(i)
                ta=B.tilde(a)
                x=pair(ta,a,m)
                canonical=line(x)
                v=next(v for v in G if line(vector(v,m))==canonical)
                expected={line(pair(B.tilde(e(j)),neg(e(j)),m))
                          for j in range(1,m) if j not in (h,i,i+h)}
                actual={line(vector(w,m)) for w in G[v]}
                self.assertEqual(actual,expected)
            # Lemma 6.12: c,tilde(1) and tilde(1),-c share all neighbors.
            for i in range(1,h):
                v=(i,h,1)
                w=(h,i,-1)
                self.assertEqual(G[v],G[w])
                expected=set()
                c=e(i)
                for candidate in G:
                    a,b=parts(vector(candidate,m),m)
                    if (all(B.dot(a,z)==0 for z in (e(0),c,e(h),B.tilde(c)))
                        and B.mul(a,b)==scale(B.tilde(c),-B.norm(b))):
                        expected.add(candidate)
                self.assertEqual(G[v],expected)
            # Lemma 6.15 at all independent a,b,tilde(a),tilde(b).
            checked=0
            for i,j in it.permutations([k for k in range(1,m) if k!=h],2):
                a,b=e(i),e(j)
                if rank([a,b,B.tilde(a),B.tilde(b)])!=4: continue
                x=pair(a,b,m)
                self.assertTrue(A.orth(x,pair(B.tilde(a),B.tilde(b),m)))
                self.assertTrue(A.orth(x,pair(B.tilde(b),neg(B.tilde(a)),m)))
                checked+=1
            if n>=3: self.assertGreater(checked,0)

    def test_06_02_04_lemma_6_14_reduction(self):
        for n in (3,4):
            A,B=Algebra(n+1),Algebra(n)
            lower=Algebra(n-1)
            circ=Algebra(n)
            bullet=Algebra(n,[-1]*(n-1)+[1],reference=True)
            yes=no=0
            for ids in it.permutations(range(1,lower.dim),4):
                a,b,c,d=map(e,ids)
                # Part (1), as printed: y includes (c,gamma_n*d).
                xs=[pair(a,b,B.dim),pair(b,neg(a),B.dim),
                    pair(B.tilde(a),B.tilde(b),B.dim),pair(B.tilde(b),neg(B.tilde(a)),B.dim)]
                ys=[pair(c,d,B.dim),pair(c,neg(d),B.dim),
                    pair(B.tilde(c),B.tilde(d),B.dim),pair(B.tilde(c),neg(B.tilde(d)),B.dim)]
                target=circ.orth(pair(a,b,lower.dim),pair(c,d,lower.dim))
                # The printed y list is recorded separately: see dedicated
                # counterexample below. Verify the natural Proposition 6.7
                # orbit (d,gamma_n*c) for the computational reduction.
                corrected=[ys[0],pair(d,neg(c),B.dim),ys[2],pair(B.tilde(d),neg(B.tilde(c)),B.dim)]
                for x,y in it.product(xs,corrected):
                    self.assertEqual(A.orth(x,y),target)
                xs2=[pair(a,B.tilde(b),B.dim),pair(B.tilde(b),neg(a),B.dim),
                     pair(B.tilde(a),neg(b),B.dim),pair(b,B.tilde(a),B.dim)]
                ys2=[pair(c,B.tilde(d),B.dim),pair(B.tilde(d),neg(c),B.dim),
                     pair(B.tilde(c),neg(d),B.dim),pair(d,B.tilde(c),B.dim)]
                target2=bullet.orth(pair(a,b,lower.dim),pair(c,d,lower.dim))
                for x,y in it.product(xs2,ys2):
                    self.assertEqual(A.orth(x,y),target2)
                yes+=target
                no+=not target
            if n==4:
                self.assertGreater(yes,0)
                self.assertGreater(no,0)

    def test_06_02_05_lemma_6_14_1_printed_counterexample(self):
        # Keep this as an explicit counterexample regression, never silently
        # replace the statement or mark the printed claim verified.
        A,C=Algebra(5),Algebra(4)
        a,b,c,d=witness(C)
        self.assertTrue(C.orth(pair(a,b,8),pair(c,d,8)))
        x,y=pair(a,b,16),pair(c,neg(d),16)
        self.assertFalse(A.orth(x,y))
        self.assertTrue(A.mul(x,y))
        R=Algebra(5,reference=True)
        self.assertEqual(A.mul(x,y),R.mul(x,y))

    def test_06_02_06_theorem_6_16_exact_kernel_classification(self):
        for n in (1,2,3,4):
            A=Algebra(n+1)
            G=ordinary_graph(n)
            for v,neighbors in G.items():
                i,j,s=v
                expected=not(n<=2 or i==j or j==0)
                self.assertEqual(bool(neighbors),expected,(n,v))
                # An empty finite neighbor set alone cannot rule out arbitrary
                # dense annihilators: exact full multiplication rank does.
                x=vector(v,1<<n)
                right=nullity(A,x)
                left=nullity(A,x,left=True)
                self.assertEqual(right>0,expected,(n,v,'right'))
                self.assertEqual(left,right)
        # Larger finite graph classification, no full rank claim at n=5.
        for v,ws in ordinary_graph(5).items():
            self.assertEqual(bool(ws),v[0]!=v[1] and v[1]!=0)

    def test_06_reference_only_all_signatures_classification_and_neighbors(self):
        # All 2^(n+1) signatures through total dimension 16. This is an
        # independent paper check; it does NOT validate a split CK product.
        positive=negative=0
        for n in (1,2,3):
            for gammas in it.product((-1,1),repeat=n+1):
                A=Algebra(n+1,gammas,reference=True)
                B=Algebra(n,gammas[:-1],reference=True)
                gamma=gammas[-1]
                G=graph(A)
                for v,ws in G.items():
                    i,j,s=v
                    a,b=e(i),scale(e(j),s)
                    chi=gamma*B.norm(a)/B.norm(b)
                    expected=not((n<=2 and chi==-1) or
                                 (i==j and gamma==-1) or (j==0 and chi==-1))
                    actual=nullity(A,vector(v,B.dim))>0
                    self.assertEqual(actual,expected,(gammas,v))
                    self.assertEqual(bool(ws),expected,(gammas,v))
                    positive+=chi==1
                    negative+=chi==-1
                    if chi==1:
                        self.assertTrue(A.orth(vector(v,B.dim),vector(v,B.dim)))
                    for w in ws:
                        k,l,t=w
                        self.assertEqual(i^j,k^l)
                        self.assertEqual(chi,gamma*B.norm(e(i^j)))
                        self.assertEqual(chi,gamma*B.norm(e(k))/B.norm(e(l)))
                    if i==j: # Lemma 6.9, including chi=+1 neighbors.
                        required={(k,k,-s) for k in range(1,B.dim) if k!=i}
                        if gamma==1:
                            self.assertTrue(required<=ws)
                            self.assertIn(v,ws)
                        else: self.assertFalse(ws)
                    if j==0 and s==1: # Lemma 6.11, full iff.
                        for w in G:
                            k,l,t=w
                            c,d=e(k),scale(e(l),t)
                            predicted=B.mul(c,d)==scale(a,B.norm(d))
                            self.assertEqual(w in ws,chi==1 and predicted)
                # Lemmas 6.10 and 6.12 chi=+1 constructions.
                h=B.dim//2
                for i in range(1,h):
                    a=e(i)
                    ta=B.tilde(a)
                    for s in (-1,1):
                        x=pair(ta,scale(a,s),B.dim)
                        chi=gamma*B.norm(ta)/B.norm(a)
                        if chi==1:
                            self.assertTrue(A.orth(x,x))
                            self.assertTrue(A.orth(x,pair(e(h),scale(e(0),s),B.dim)))
                            self.assertTrue(A.orth(x,pair(a,scale(ta,s*gamma),B.dim)))
                    x=pair(a,e(h),B.dim)
                    chi=gamma*B.norm(a)/B.norm(e(h))
                    if chi==1:
                        for y in (x,pair(e(h),scale(a,gamma),B.dim),
                                  pair(ta,scale(e(0),-gammas[-2]),B.dim)):
                            self.assertTrue(A.orth(x,y))
        self.assertGreater(positive,0)
        self.assertGreater(negative,0)


def recursive_basis_iterative(i,j,n):
    """Uncached ordinary oracle for arbitrary width; no recursion limit."""
    sign=1
    for bit in range(n-1,-1,-1):
        h=1<<bit
        u,v=bool(i&h),bool(j&h)
        i,j=i&(h-1),j&(h-1)
        if u and j: sign=-sign
        if u and v: sign=-sign
        if v: i,j=j,i
    return sign


class GPUVerification(unittest.TestCase):
    def test_06_complexified_all_signatures(self):
        from tests.papers.type_replay import replay_signatures
        replay_signatures(self,(g for n in range(1,5) for g in it.product((-1,1),repeat=n)))

    def setUp(self):
        if CK.cp is None:
            raise unittest.SkipTest('multiply(): CuPy is not installed; CPU checks still run')
        if not CK.loaded():
            raise unittest.SkipTest('multiply(): cuda_kernel.so missing; run make kernel')
        try:
            count=CK.cp.cuda.runtime.getDeviceCount()
        except CK.cp.cuda.runtime.CUDARuntimeError as exc:
            raise unittest.SkipTest('multiply(): CUDA device unavailable: '+str(exc))
        if not count:
            raise unittest.SkipTest('multiply(): no CUDA devices')

    def batch_check(self,A,pairs):
        cp=CK.cp
        left=cp.asarray([[float(x.get(i,0)) for i in range(A.dim)] for x,y in pairs],dtype=cp.float64)
        right=cp.asarray([[float(y.get(i,0)) for i in range(A.dim)] for x,y in pairs],dtype=cp.float64)
        actual=cp.asnumpy(replay_product(left,right)).tolist()
        for row,(x,y) in zip(actual,pairs):
            expected=A.mul(x,y)
            self.assertEqual(row,[expected.get(i,0) for i in range(A.dim)])

    def test_02_multiply_all_basiss_and_dense_oracle(self):
        rng=random.Random(260828176)
        for n in range(1,6):
            A=Algebra(n,reference=True)
            pairs=[(e(i),e(j)) for i,j in it.product(range(A.dim),repeat=2)]
            pairs += [(random_vector(rng,A.dim),random_vector(rng,A.dim)) for _ in range(16)]
            self.batch_check(A,pairs)

    def test_03_to_06_multiply_hexagon_tables_and_all_sedenion_candidates(self):
        for n in (3,4):
            A,B=Algebra(n+1),Algebra(n)
            a,b,c,d=witness(A)
            f,t=double_hexagon(B,a,b,c,d)
            F=f[:7]+t[:7]
            self.batch_check(A,list(it.product(F,repeat=2)))
        # Every pure two-term sedenion candidate against every candidate,
        # including XOR-incompatible nonzero products and both signs/orders.
        from .algebra import vertices
        A=Algebra(4)
        vs=[vector(v,8) for v in vertices(3)]
        self.batch_check(A,list(it.product(vs,repeat=2)))


def load_tests(loader, tests, pattern):
    # Keep the main section-by-section narrative first. The optional device
    # replay follows it, using the same mathematics on a different backend.
    return unittest.TestSuite([loader.loadTestsFromTestCase(PaperSections),
                               loader.loadTestsFromTestCase(GPUVerification)])


if __name__=='__main__':
    unittest.main(verbosity=2)
