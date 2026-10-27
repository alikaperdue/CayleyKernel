"""Bales 2016: P2 shuffle twist vs its independent doubling and tree."""
from itertools import product
import random
import unittest
from CayleyKernel import twist
from .common import basis_product, e, gpu_replay, recursive_basis
try:
    import pytest
    pytestmark = pytest.mark.cpu
except ImportError:
    pass


def omega2(p, q):
    if not p or not q: return 1
    if p == q: return -1
    if p > q: return -omega2(q, p)
    n = p.bit_length()-1
    return 1 if q.bit_length() == p.bit_length() else (-1 if (q >> n)&1 else 1)


def doubling2(p, q):
    """LSB shuffle recursion of P2, implemented iteratively for deep indices."""
    sign = 1
    while p or q:
        a,b = p>>1,q>>1
        if p&1 and q&1:
            sign *= 1 if a else -1
        elif not p&1 and q&1:
            sign *= -1 if a else 1
            a,b = b,a
        p,q = a,b
    return sign


TREE = {'C':('C','T','L','-D'), 'T':('T','T','1','-1'),
        'L':('L','-1','L','1'), '-D':('-D','1','-1','-D'),
        '1':('1',)*4, '-1':('-1',)*4}


def tree2(p,q):
    node='C'
    for k in reversed(range(max(p.bit_length(),q.bit_length()))):
        node=TREE[node][2*((p>>k)&1)+((q>>k)&1)]
    return -1 if node in ('-1','-D') else 1


def reverse_bits(i,n):
    return int(format(i, '0%db'%n)[::-1],2)


class Sections(unittest.TestCase):
    def test_01_02_standard_product_and_shuffle_conventions(self):
        # The module is P3 transpose with block basis, not P2 shuffle.
        mismatches=0
        for p,q in product(range(64), repeat=2):
            self.assertEqual(twist(p,q), recursive_basis(p,q,(-1,)*6))
            self.assertEqual(basis_product(p+1,q+1), twist(p,q)*((p^q)+1))
            mismatches += twist(p,q) != omega2(p,q)
        self.assertGreater(mismatches,0)
        for p in range(128):
            self.assertEqual(omega2(p,0),1)
            self.assertEqual(omega2(0,p),1)
            if p: self.assertEqual(omega2(p,p),-1)

    def test_03_equations_3_1_3_2_all_pairs_through_1024(self):
        for p,q in product(range(1024), repeat=2):
            self.assertEqual(omega2(p,q), doubling2(p,q), (p,q))

    def test_04_tree_and_printed_examples(self):
        for p,q in product(range(128),repeat=2):
            self.assertEqual(tree2(p,q),doubling2(p,q),(p,q))
        for p,q,index,sign in [(3,14,13,-1),(35,55,20,1),(87,340,259,-1),(51,12,63,-1)]:
            self.assertEqual((p^q,omega2(p,q)),(index,sign))

    def test_05_theorem_high_indices_and_fixed_bit_reversal(self):
        r=random.Random(1474295358)
        for n in (16,20,32,128,1024,4096):
            for _ in range(16):
                p,q=r.getrandbits(n),r.getrandbits(n)
                self.assertEqual(omega2(p,q),doubling2(p,q))
                self.assertEqual(omega2(p,q),tree2(p,q))
        # Shuffle embedding is bit reversal at a FIXED level. The module
        # tests here check XOR compatibility only, not a P2 isomorphism.
        for p,q in product(range(128),repeat=2):
            self.assertEqual(reverse_bits(p^q,7),reverse_bits(p,7)^reverse_bits(q,7))

    def test_06_gpu_standard_convention(self):
        gpu_replay(self,[(e(p),e(q)) for p,q in product(range(16),repeat=2)],16)
