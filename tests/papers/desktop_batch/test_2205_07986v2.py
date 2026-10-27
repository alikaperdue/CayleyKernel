"""Ren--Zhao explicit twist, independent doubling, and split correction."""
from itertools import product
import random
import unittest
import CayleyKernel as CK
from .common import Algebra,add,conj,e,neg,pair,scale,mul,basis_product,recursive_basis,gpu_replay
try:
    import pytest
    pytestmark=pytest.mark.cpu
except ImportError:
    pass
MEASUREMENTS={}


def degree(a):
    if a<=0:raise ValueError('degree is defined for positive indices')
    return (a&-a).bit_length()-1


def sigma(a,b):
    """Literal (4.1), arithmetic in F2; phi is evaluated bit by bit."""
    if not a or not b:return 0
    if a==b:return 1
    da,db=degree(a),degree(b)
    ell=max(da,db,degree(a^b))
    parity=sum(bool((a>>i)&1 or (b>>i)&1)
               for i in range(ell,max(a.bit_length(),b.bit_length())))
    if da>db:parity+=(b>>ell)&1
    elif da<db:parity+=1+((a>>ell)&1)
    else:parity+=(a>>ell)&1
    return parity%2


def independent_sign(a,b):
    """Iterative block doubling; no closed sign formula or module calls."""
    s=1
    while a or b:
        h=1<<(max(a.bit_length(),b.bit_length())-1)
        u,v=a>=h,b>=h
        a,b=a%h,b%h
        if u and v:
            s*=1 if b else -1
            a,b=b,a
        elif u:
            s*=-1 if b else 1
        elif v:a,b=b,a
    return s


def assoc_parity(a,b,c):
    return (sigma(a,b)+sigma(a^b,c)+sigma(b,c)+sigma(a,b^c))%2


class Sections(unittest.TestCase):
    def test_08_gpu_complex_split_last_parameter(self):
        from tests.papers.type_replay import replay_signatures
        replay_signatures(self,([-1]*(n-1)+[1] for n in range(1,7)))

    def test_02_doubling_generator_relations_and_involution(self):
        rng=random.Random(220507986)
        for n in range(1,7):
            h=1<<(n-1);g=e(h)
            for _ in range(8):
                a={i:rng.randrange(-2,3) for i in range(h)}
                b={i:rng.randrange(-2,3) for i in range(h)}
                a={i:c for i,c in a.items() if c};b={i:c for i,c in b.items() if c}
                self.assertEqual(mul(a,g),mul(g,conj(a)))
                self.assertEqual(mul(mul(g,a),b),mul(g,mul(b,a)))
                self.assertEqual(mul(a,mul(b,g)),mul(mul(b,a),g))
                self.assertEqual(mul(mul(g,a),mul(g,b)),neg(mul(b,conj(a))))
                self.assertEqual(mul(mul(a,g),mul(b,g)),neg(mul(conj(b),a)))
            for index in range(1<<n):
                x=e(0)
                for k in range(n):
                    if index>>k&1:x=mul(x,e(1<<k))
                self.assertEqual(x,e(index))

    def test_03_2_proposition_1_recursive_parity(self):
        for n in range(7):
            h=1<<n
            for a,b,u,v in product(range(h),range(h),(0,1),(0,1)):
                rhs=(sigma(a,b)*(1+v)+sigma(b,a)*v+sigma(b,b)*u+u*v)%2
                self.assertEqual(sigma(a+u*h,b+v*h),rhs)

    def test_03_1_twisted_group_is_not_associative_cocycle(self):
        witnesses=[(a,b,c) for a,b,c in product(range(8),repeat=3) if assoc_parity(a,b,c)]
        self.assertTrue(witnesses)
        for a,b,c,d in product(range(8),repeat=4):
            self.assertEqual((assoc_parity(b,c,d)+assoc_parity(a^b,c,d)+assoc_parity(a,b^c,d)
                              +assoc_parity(a,b,c^d)+assoc_parity(a,b,c))%2,0)
        MEASUREMENTS['nonassociative_basis_triple']=witnesses[0]

    def test_04_1_lemma_4_and_theorem_1_exhaustive(self):
        for a,b in product(range(1024),repeat=2):
            s=(-1)**sigma(a,b)
            self.assertEqual(CK.twist(a,b),s,(a,b))
            self.assertEqual(independent_sign(a,b),s,(a,b))
            self.assertEqual(basis_product(a+1,b+1),s*((a^b)+1))
            if a and b and a!=b:
                ell=max(degree(a),degree(b),degree(a^b))
                self.assertTrue(((a|b)>>ell)&1)
                if degree(a)==degree(b):self.assertEqual(((a>>ell)^(b>>ell))&1,1)

    def test_04_2_example_1_and_equation_4_11_discrepancy(self):
        self.assertEqual((degree(5),degree(6),degree(5^6)),(0,1,0))
        self.assertEqual(sigma(5,6),1)
        self.assertEqual(mul(e(5),e(6)),{3:-1})
        mismatches=[]
        for a,b in product(range(1,8),repeat=2):
            if a==b:continue
            parity=sum(bool((a|b)>>i&1) for i in range(3))
            printed=(parity+(((b if degree(a)>degree(b) else a)>>2)&1)
                     +(degree(a)<degree(b)))%2
            if printed!=sigma(a,b):mismatches.append([a,b])
        self.assertTrue(mismatches)
        MEASUREMENTS['equation_4_11_mismatches']=mismatches
        MEASUREMENTS['worked_example_correct_degrees']=[0,1,0]

    def test_05_split_last_parameter_and_proposition_2(self):
        for n in range(1,7):
            h=1<<(n-1);params=(-1,)*(n-1)+(1,)
            for a,b in product(range(1<<n),repeat=2):
                s=(-1)**(sigma(a,b)^int(a>=h and b>=h))
                self.assertEqual(s,recursive_basis(a,b,params))
        # Split complex (1+g)(1-g)=0, illustrating changed norm signature.
        A=Algebra(1,gammas=[1],reference=True)
        self.assertEqual(A.mul({0:1,1:1},{0:1,1:-1}),{})

    def test_06_extension_high_indices_and_padding(self):
        rng=random.Random(220507987)
        rows=[]
        for bits in (20,32,128,1024,4096):
            for _ in range(24):
                a,b=rng.getrandbits(bits),rng.getrandbits(bits)
                s=(-1)**sigma(a,b)
                self.assertEqual(s,CK.twist(a,b))
                self.assertEqual(s,independent_sign(a,b))
                self.assertEqual(basis_product(-(a+1),b+1),-s*((a^b)+1))
            rows.append(dict(index_bits=bits,samples=24))
        MEASUREMENTS['high_index_coverage']=rows

    def test_07_gpu_basis_and_dense_doubling(self):
        rng=random.Random(220507988)
        vectors=[(e(a),e(b)) for a,b in product(range(16),repeat=2)]
        for _ in range(16):
            vectors.append(tuple({i:rng.randrange(-2,3) for i in range(16)} for _ in range(2)))
        gpu_replay(self,vectors,16)
