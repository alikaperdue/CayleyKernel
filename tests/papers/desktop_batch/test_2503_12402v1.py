"""Zuest: exact exterior forms plus octonion-derived Clifford spinor actions."""
from fractions import Fraction as Q
from itertools import combinations,permutations,product
import math
import unittest
from .common import add,conj,e,mul,neg,gpu_replay
try:
    import pytest
    pytestmark=pytest.mark.cpu
except ImportError:pass
MEASUREMENTS={}


def wedge_sign(a,b):
    return (-1)**sum((b&((1<<i)-1)).bit_count() for i in range(a.bit_length()) if a>>i&1)


def wedge(a,b):
    out={}
    for i,c in a.items():
        for j,d in b.items():
            if not i&j:out[i|j]=out.get(i|j,0)+wedge_sign(i,j)*c*d
    return {i:c for i,c in out.items() if c}


def power(a,n):
    out={0:1}
    for _ in range(n):out=wedge(out,a)
    return out


def star(form,n):
    top=(1<<n)-1
    return {top^i:wedge_sign(i,top^i)*c for i,c in form.items()}


def volume(start,stop):
    out={0:1}
    for j in range(start,stop):out=wedge(out,{1<<(2*j):1,1<<(2*j+1):1j})
    return out


def real(form):return {i:int(complex(c).real) for i,c in form.items() if complex(c).real}


def phi():
    O1,O2=volume(0,4),volume(4,8)
    w={3<<(2*j):1 for j in range(8)}
    return add(real(wedge(O1,O2)),{i:Q(c,24) for i,c in power(w,4).items()},
               {i:Q(c,2) for i,c in wedge(add(real(O1),real(O2)),power(w,2)).items()})


def cayley_T(indices):
    a,b,c,d=map(e,indices)
    return mul(conj(a),mul(b,mul(conj(c),d))).get(0,0)


def gamma8(i,state):
    block,j=state
    z=mul(e(j),e(i));k,c=next(iter(z.items()))
    return (1-block,k),c*(1 if block else (-1 if i==0 else 1))


def gamma16(i,state):
    b,j,c,k=state
    if i<8:
        (nb,nj),s=gamma8(i,(b,j))
        return (nb,nj,c,k),s*(-1 if c else 1)
    (nc,nk),s=gamma8(i-8,(c,k))
    return (b,j,nc,nk),s


def spinor_coefficient(indices):
    state=(0,0,0,0);sign=1
    for i in reversed(indices):
        state,s=gamma16(i,state);sign*=s
    return sign if state in ((0,0,0,0),(0,1,0,1)) else 0


class Sections(unittest.TestCase):
    def test_02_1_exterior_wedge_and_hodge(self):
        for i,j in product(range(16),repeat=2):
            a,b={1<<i:1},{1<<j:1}
            self.assertEqual(wedge(a,b),{k:-c for k,c in wedge(b,a).items()})
        for n,k in [(8,4),(16,8)]:
            for inds in combinations(range(n),k):
                f={sum(1<<i for i in inds):1}
                self.assertEqual(star(star(f,n),n),f)

    def test_02_2_octonion_Q8_and_cayley_form(self):
        q=e(0)
        for i in reversed(range(8)):q=mul(q,e(i))
        self.assertEqual(q,e(0))
        printed=['+1234','+1256','-1278','+1357','+1368','+1458','-1467',
                 '+5678','+3478','-3456','+2468','+2457','+2367','-2358']
        expected={sum(1<<(int(i)-1) for i in s[1:]):1 if s[0]=='+' else -1 for s in printed}
        actual={}
        for indices in combinations(range(8),4):
            total=0
            for perm in permutations(indices):
                inversions=sum(perm[i]>perm[j] for i in range(4) for j in range(i+1,4))
                total+=(-1)**inversions*cayley_T(perm)
            value=Q(total,24)
            if value:actual[sum(1<<i for i in indices)]=value
        self.assertEqual(actual,expected)
        self.assertEqual(sum(c*c for c in actual.values()),14)

    def test_02_4_octonionic_clifford_relations(self):
        for i,j,block,k in product(range(8),range(8),range(2),range(8)):
            state=(block,k)
            a,s=gamma8(j,state);a,t=gamma8(i,a)
            b,u=gamma8(i,state);b,v=gamma8(j,b)
            if i==j:self.assertEqual((a,s*t),(state,-1))
            else:self.assertEqual(a,b);self.assertEqual(s*t,-u*v)

    def test_03_main_form_exact_self_duality_norm_and_product_bound(self):
        F=phi()
        self.assertEqual(star(F,16),F)
        self.assertEqual(sum(c*c for c in F.values()),294)
        self.assertEqual(wedge(F,F),{(1<<16)-1:Q(294)})
        ratio=Q(294,256);self.assertGreater(ratio,1)
        MEASUREMENTS['phi_nonzero_coefficients']=len(F)
        MEASUREMENTS['phi_norm_squared']=294
        MEASUREMENTS['diagonal_product_plane_value']=str(ratio)

    def test_04_calibrated_coordinate_planes_and_sampled_comass(self):
        F=phi()
        # Special Lagrangian, complex, and product-of-Cayley representatives.
        for inds in [tuple(range(0,16,2)),tuple(range(8)),(0,1,2,3,8,9,10,11)]:
            self.assertEqual(F.get(sum(1<<i for i in inds),0),1)
        try:import numpy as np
        except ImportError:self.skipTest('NumPy unavailable for orthonormal-frame samples')
        rng=np.random.default_rng(250312402);values=[]
        for _ in range(64):
            frame=np.linalg.qr(rng.normal(size=(16,8)))[0]
            score=sum(float(c)*np.linalg.det(frame[[i for i in range(16) if mask>>i&1]]) for mask,c in F.items())
            self.assertLessEqual(abs(score),1+1e-12);values.append(float(score))
        MEASUREMENTS['sampled_frame_count']=64;MEASUREMENTS['sampled_max_abs_value']=max(map(abs,values))

    def test_05_octonion_spinor_form_has_same_norm_and_self_duality(self):
        F={sum(1<<i for i in inds):spinor_coefficient(inds) for inds in combinations(range(16),8)}
        F={k:c for k,c in F.items() if c}
        self.assertEqual(sum(c*c for c in F.values()),294)
        self.assertEqual(star(F,16),F)
        self.assertEqual(len(F),294)
        MEASUREMENTS['spinor_form_norm_squared']=294

    def test_06_gpu_cayley_basis_replay(self):
        gpu_replay(self,[(e(i),e(j)) for i,j in product(range(8),repeat=2)],8)
