"""Bales: all eight shuffle products, twist trees and periodicity."""
from itertools import product
import random
import unittest
from CayleyKernel import twist
from .common import basis_product, e, gpu_replay
try:
    import pytest
    pytestmark = pytest.mark.cpu
except ImportError:
    pass


def omega(p,q,kind):
    """Table 1 doubling, LSB shuffle basis, independent of twist trees."""
    s=1
    while p or q:
        u,v=p&1,q&1
        a,b=p>>1,q>>1
        k=kind%4
        if not u and not v:
            swap=k<2
        elif not u and v:
            if kind<4:
                s *= -1 if a else 1
                swap=k in (0,2)
            else:
                swap=k in (1,3)
        elif u and not v:
            if kind<4:
                swap=k in (1,3)
            else:
                s *= -1 if b else 1
                swap=k in (0,2)
        else:
            s *= (1 if a else -1) if kind<4 else (1 if b else -1)
            swap=k in (1,3)
        p,q=(b,a) if swap else (a,b)
    return s


BASE={'C':(('C',1),('T',1),('L',1),('D',-1)),
      'T':(('T',1),('T',1),('I',1),('I',-1)),
      'L':(('L',1),('I',-1),('L',1),('I',1)),
      'D':(('D',1),('I',-1),('I',1),('D',1))}


def tree(p,q,k):
    node,s='C',1
    inner=[(-1,1,1,1),(-1,-1,-1,-1),(1,1,1,1),(1,-1,-1,-1)][k]
    for i in reversed(range(max(p.bit_length(),q.bit_length()))):
        d=2*((p>>i)&1)+((q>>i)&1)
        if node=='I': s*=inner[d]
        else:
            node,t=BASE[node][d]
            s*=t
    return s


def kernel_shuffle(p,q):
    n=max(1,p.bit_length(),q.bit_length())
    rev=lambda x:int(format(x,'0%db'%n)[::-1],2)
    a,b=rev(p),rev(q)
    label=basis_product(a+1,b+1)
    assert abs(label)-1==rev(p^q)
    assert (1 if label>0 else -1)==twist(a,b)
    return twist(a,b)


class Sections(unittest.TestCase):
    def test_02_table_1_transposes_and_module_conversion(self):
        for p,q in product(range(64),repeat=2):
            for k in range(4): self.assertEqual(omega(p,q,k),omega(q,p,k+4))
            self.assertEqual(omega(p,q,7),kernel_shuffle(p,q))

    def test_03_04_05_tree_navigation(self):
        for p,q in product(range(128),repeat=2):
            for k in range(4): self.assertEqual(omega(p,q,k),tree(p,q,k),(p,q,k))
        self.assertEqual((3^1,omega(3,1,3)),(2,1))
        self.assertEqual((25^17,omega(25,17,3)),(8,1))

    def test_06_theorem_1(self):
        for k,p in product(range(8),range(128)):
            self.assertEqual(omega(p,0,k),1)
            self.assertEqual(omega(0,p,k),1)
            if p: self.assertEqual(omega(p,p,k),-1)
            for q in range(1,128):
                if p and p!=q: self.assertEqual(omega(p,q,k),-omega(q,p,k))

    def test_06_theorems_2_3_periodicity(self):
        for n in range(1,7):
            h=1<<n
            for kind,p,q,k in product(range(8),range(h//2,h),range(h,2*h),(0,1,2,7,31)):
                self.assertEqual(omega(p,q,kind),omega(p,q+k*h,kind))
            for kind,p,q,k in product(range(8),range(h//2,h),range(h//2,h),(0,1,3,16)):
                self.assertEqual(omega(p,q,kind),omega(p+k*h,q+k*h,kind))

    def test_07_theorems_4_5_and_high_level_module(self):
        rng=random.Random(160202843)
        for bits in (8,20,32,128,1024,4096):
            for _ in range(24):
                p=rng.getrandbits(bits-1) or 1
                n=p.bit_length(); h=1<<n
                q=h+rng.getrandbits(bits+2)
                for kind in range(8):
                    self.assertEqual(omega(p,q,kind),omega(p,h+q%h,kind))
                self.assertEqual(kernel_shuffle(p,q),kernel_shuffle(p,h+q%h))
                p=(1<<(bits-1))|rng.getrandbits(bits-1)
                q=(1<<(bits-1))|rng.getrandbits(bits-1)
                if p==q: q^=1
                h=1<<(p^q).bit_length()
                for kind in range(8):
                    self.assertEqual(omega(p,q,kind),omega(h+p%h,h+q%h,kind))
                self.assertEqual(kernel_shuffle(p,q),kernel_shuffle(h+p%h,h+q%h))

    def test_08_gpu_standard_products(self):
        gpu_replay(self,[(e(i),e(j)) for i,j in product(range(8),repeat=2)],8)
