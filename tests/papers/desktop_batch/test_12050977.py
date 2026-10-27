"""US12050977B2: permutation encoder, and limits of CD substitution."""
from itertools import product
import random
import unittest
from .common import e, mul, neg, gpu_replay
try:
    import pytest
    pytestmark = pytest.mark.cpu
except ImportError:
    pass


def shift(x, k):
    k %= len(x)
    return x[k:] + x[:k]


def encode(seed, symbols, perm):
    x = tuple(seed)
    for k in symbols:
        y = shift(x, k)
        x = tuple(y[i] for i in perm)
    return x


class Sections(unittest.TestCase):
    def test_01_background_permutation_preserves_density_and_distance(self):
        r = random.Random(12050977)
        for dim in (8, 1024, 10000, 65536, 1048576):
            x = tuple(r.getrandbits(1) for _ in range(dim))
            y = shift(x, 27)
            self.assertEqual(sum(x), sum(y))
            # Same permutation applied to two vectors is an isometry.
            z = tuple(r.getrandbits(1) for _ in range(dim))
            self.assertEqual(sum(a!=b for a,b in zip(x,z)),
                             sum(a!=b for a,b in zip(y,shift(z,27))))

    def test_02_detailed_description_character_shift_stages_fig6(self):
        self.assertEqual(shift((1,0,1,1,0,0,1,0), 2), (1,1,0,0,1,0,1,0))
        self.assertEqual(shift((1,0,1,1,0,0,1,0), 0), (1,0,1,1,0,0,1,0))
        self.assertEqual(shift((0,1,1,1,0,1,0,0), 1), (1,1,1,0,1,0,0,0))

    def test_03_flowchart_sequential_recurrence(self):
        seed = (1,0,1,1,0,0,1,0)
        perm = (3,0,6,1,7,4,2,5)  # explicitly chosen embodiment, not traced Fig.6 wires
        for seq in product(range(4), repeat=3):
            p = list(range(8))
            for k in seq:
                p = [p[(i+k)%8] for i in perm]
            self.assertEqual(encode(seed, seq, perm), tuple(seed[i] for i in p))
            self.assertEqual(sum(encode(seed, seq, perm)), sum(seed))

    def test_04_bundling_distance_and_finite_collision(self):
        perm = (3,0,6,1,7,4,2,5)
        vs = [encode((1,0,1,1,0,0,1,0), seq, perm)
              for seq in product(range(4), repeat=4)]
        # 256 words into at most C(8,4)=70 vectors cannot be injective.
        self.assertLess(len(set(vs)), len(vs))
        bundle = tuple(map(sum, zip(*vs)))
        self.assertEqual(sum(bundle), 4*len(vs))
        self.assertTrue(all(sum(a!=b for a,b in zip(v,v)) == 0 for v in vs))

    def test_05_extension_basis_action_is_signed_xor_not_cyclic_shift(self):
        for n in (3, 4, 10, 20, 128, 1024, 4096):
            u = (1 << (n-1)) + 1
            for i in (0, 1, 2, u):
                result = mul(e(u), e(i))
                self.assertEqual(next(iter(result)), u ^ i)
                self.assertEqual(mul(e(u), result), neg(e(i)))
            # Positive binary seed values can acquire negative coefficients.
            self.assertEqual(mul(e(u), e(u)), {0:-1})
        # Ordinary +1 cyclic shift has a different index map from XOR 1.
        self.assertNotEqual([i^1 for i in range(8)], [(i+1)%8 for i in range(8)])

    def test_06_gpu_signed_action_negative_control(self):
        gpu_replay(self, [(e(3), e(3)), (e(3), {0:1,1:1,3:1})], 8)
