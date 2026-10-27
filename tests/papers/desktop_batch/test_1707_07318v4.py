"""Bales v4: enumerate 32 products; distinguish pseudo from octonion."""
from itertools import combinations, product
from functools import lru_cache
import unittest
from .common import add, conj, e, neg, gpu_replay
from .test_1602_02843v3 import omega, kernel_shuffle
from .test_1474295358 import tree2
try:
    import pytest
    pytestmark=pytest.mark.cpu
except ImportError:
    pass


@lru_cache(None)
def sign(p,q,f,g):
    if not p and not q:return 1
    a,b=p>>1,q>>1; u,v=p&1,q&1
    s=1
    if not u and not v: swap=f%4<2
    elif u and v:
        s= (1 if a else -1) if f<4 else (1 if b else -1)
        swap=f%2==1
    elif not u and v:
        s=-1 if g in (0,1) and a else 1
        swap=g in (0,3)
    else:
        s=-1 if g in (2,3) and b else 1
        swap=g in (1,2)
    return s*(sign(b,a,f,g) if swap else sign(a,b,f,g))


def pmul(x,y,f,g):
    out={}
    for p,a in x.items():
        for q,b in y.items():
            k=p^q; out[k]=out.get(k,0)+a*b*sign(p,q,f,g)
    return {k:a for k,a in out.items() if a}


PAIRS=[(0,0),(1,1),(2,0),(3,1),(4,2),(5,3),(6,2),(7,3)]
TRIPLES=[[(1,2,3),(1,4,5),(1,6,7),(2,6,4),(7,2,5),(5,6,3),(3,4,7)],
         [(1,2,3),(1,4,5),(1,6,7),(2,6,4),(5,2,7),(7,4,3),(3,6,5)],
         [(1,2,3),(1,4,5),(1,6,7),(2,4,6),(7,2,5),(5,6,3),(3,4,7)],
         [(1,2,3),(1,4,5),(1,6,7),(2,4,6),(5,2,7),(7,4,3),(3,6,5)]]


class Sections(unittest.TestCase):
    def test_02_05_basic_properties_all_32(self):
        for f,g in product(range(8),range(4)):
            m=lambda x,y:pmul(x,y,f,g)
            for p,q in product(range(16),repeat=2):
                self.assertEqual(sign(p,0,f,g),1)
                if p:self.assertEqual(sign(p,p,f,g),-1)
                if p and q and p!=q:self.assertEqual(sign(p,q,f,g),-sign(q,p,f,g))
                x=add(e(p),e(q),{0:2}); y=add(e(3),{4:2})
                self.assertEqual(m(x,conj(x)),{0:sum(a*a for a in x.values())})
                self.assertEqual(conj(m(x,y)),m(conj(y),conj(x)))

    def test_06_07_initial_elimination_16(self):
        survivors=[]
        for f,g in product(range(8),range(4)):
            if all(sign(p,q,f,g)==sign(q,p^q,f,g)==sign(p^q,p,f,g)
                   for s in range(1,32)
                   for p,q in [(2*s,1),(1,2*s),(2*s,2*s+1)]):survivors.append((f,g))
        self.assertEqual(survivors,[(f,g) for f,g in product(range(8),range(4)) if (f<4)==(g<2)])

    def test_08_09_full_elimination_8(self):
        survivors=[]
        for f,g in product(range(8),range(4)):
            if all(sign(p,q,f,g)==sign(q,p^q,f,g)==sign(p^q,p,f,g)
                   for p,q in product(range(1,32),repeat=2) if p!=q):survivors.append((f,g))
        self.assertEqual(survivors,sorted(PAIRS))
        for kind,(f,g) in enumerate(PAIRS):
            for p,q in product(range(64),repeat=2):self.assertEqual(sign(p,q,f,g),omega(p,q,kind))

    def test_10_11_fano_triples_and_pseudo_zero_divisors(self):
        for kind,(f,g) in enumerate(PAIRS):
            triples=TRIPLES[kind%4]
            if kind>=4:triples=[(p,r,q) for p,q,r in triples]
            for p,q,r in triples:
                self.assertEqual(p^q,r)
                for a,b in [(p,q),(q,r),(r,p)]:self.assertEqual(sign(a,b,f,g),1)
            z=pmul(add(e(2),e(5)),add(e(3),e(4)),f,g)
            if kind in (1,2,5,6):self.assertEqual(z,{})
            else:self.assertTrue(z)

    def test_11_octonion_vs_pseudo_alternativity_polarization(self):
        for kind,(f,g) in enumerate(PAIRS):
            m=lambda x,y:pmul(x,y,f,g)
            failures=0
            # Polarize the quadratic variable, plus diagonal cases. This
            # exhausts coefficients of left/right alternativity in R^8.
            for i,j in combinations(range(8),2):
                x=add(e(i),e(j))
                for k in range(8):
                    y=e(k)
                    failures += m(m(x,x),y)!=m(x,m(x,y))
                    failures += m(y,m(x,x))!=m(m(y,x),x)
            for i,j in product(range(8),repeat=2):
                x,y=e(i),e(j)
                failures += m(m(x,x),y)!=m(x,m(x,y))
                failures += m(y,m(x,x))!=m(m(y,x),x)
            self.assertEqual(failures==0,kind in (0,3,4,7))

    def test_12_15_module_and_twist_tables(self):
        for p,q in product(range(128),repeat=2):
            self.assertEqual(sign(p,q,7,3),kernel_shuffle(p,q))
        # Rebuilding from 2x2 parity blocks agrees with direct tables,
        # including every boundary type (C,L,T,D,interior).
        for kind,(f,g) in enumerate(PAIRS):
            for r,s in product(range(16),repeat=2):
                for u,v in product(range(2),repeat=2):
                    self.assertEqual(sign(2*r+u,2*s+v,f,g),omega(2*r+u,2*s+v,kind))

    def test_16_tree_example_and_extension_families(self):
        self.assertEqual((93^37,tree2(93,37)),(120,-1))
        for n,m in [(n,m) for n in range(6) for m in range(n+1,8)]+[(20,32),(128,1024),(1024,4096)]:
            for r,s in product(range(min(1<<n,8)),repeat=2):
                for p,q in [(2**n+r,2**m+s),(2**m+2**n+r,2**n+s),(2**m+r,2**m+2**n+s)]:
                    self.assertEqual(tree2(p,q),1)

    def test_17_gpu_standard_octonions(self):
        gpu_replay(self,[(add(e(2),e(5)),add(e(3),e(4)))],8)
