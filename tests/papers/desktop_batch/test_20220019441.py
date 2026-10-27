"""Nine-part HDC patent: selected software primitives, no hardware claims."""
from itertools import product
import unittest
from .test_1703_00144v4 import np
from .common import e,mul,gpu_replay
try:
    import pytest
    pytestmark=pytest.mark.cpu
except ImportError:
    pass
MEASUREMENTS={}


def encode_dna(text,bases):
    out=np.ones(len(next(iter(bases.values()))),dtype=np.int64)
    for i,char in enumerate(text):out*=np.roll(bases[char],i)
    return out


def edit_distance(a,b):
    D=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):D[i][0]=i
    for j in range(len(b)+1):D[0][j]=j
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            D[i][j]=min(D[i-1][j]+1,D[i][j-1]+1,D[i-1][j-1]+(a[i-1]!=b[j-1]))
    return D[-1][-1]


@unittest.skipIf(np is None,'NumPy unavailable')
class Sections(unittest.TestCase):
    def test_01_privehd_linear_encoding_reconstruction_and_quantization(self):
        B=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]])
        v=np.array([1,2,3,4]);H=v@B
        np.testing.assert_array_equal(H@B.T//4,v)
        # Sign-only encoding is many-to-one; this is not a privacy proof.
        np.testing.assert_array_equal(np.sign(v@B),np.sign((2*v)@B))
        self.assertFalse(np.array_equal(v,2*v))

    def test_02_hyperrec_xor_majority(self):
        for a,b in product(range(256),repeat=2):self.assertEqual((a^b)^a,b)
        # All bit triples: majority agrees with at least two constituents.
        for bits in product((0,1),repeat=3):
            majority=int(sum(bits)>=2)
            self.assertGreaterEqual(sum(x==majority for x in bits),2)

    def test_03_hpu_blend_dot_product_has_cross_term(self):
        rng=np.random.default_rng(20220019441);D=4096
        H1=rng.choice([-1,1],D);H2=rng.choice([-1,1],D);d=1000
        mixed=np.r_[H1[:d],H2[d:]]
        exact=d+int(H2[d:]@H1[d:])
        self.assertEqual(int(mixed@H1),exact)
        MEASUREMENTS['blend_dot_vs_claimed_d']={'d':d,'dot':exact,'cross_term':exact-d}
        # Literal dot=d is only an expectation under independent random inputs.
        self.assertNotEqual(exact,d)

    def test_04_searchhd_bit_substitution_moves_toward_sample(self):
        rng=np.random.default_rng(20220019442)
        model=rng.integers(0,2,1024);sample=rng.integers(0,2,1024)
        chosen=rng.choice(1024,200,replace=False);updated=model.copy();updated[chosen]=sample[chosen]
        self.assertLessEqual(np.count_nonzero(updated!=sample),np.count_nonzero(model!=sample))
        self.assertEqual(np.count_nonzero(updated[chosen]!=sample[chosen]),0)

    def test_05_associative_search_normalization_and_safe_bound(self):
        rng=np.random.default_rng(20220019443)
        memory=rng.normal(size=(16,128));memory/=np.linalg.norm(memory,axis=1)[:,None]
        q=rng.normal(size=128);scores=memory@q
        cos=scores/np.linalg.norm(q)
        self.assertEqual(int(scores.argmax()),int(cos.argmax()))
        for row,score in zip(memory,scores):
            # A general, safe suffix bound; not a hardware emulation.
            partial=row[:64]@q[:64]
            bound=np.linalg.norm(row[64:])*np.linalg.norm(q[64:])
            self.assertLessEqual(abs(score-partial),bound+1e-13)

    def test_06_geniehd_rolling_window_vs_naive(self):
        rng=np.random.default_rng(20220019444);D=4096
        bases={b:rng.choice([-1,1],D) for b in 'ACGT'}
        ref='TCTAGAATCGTACGTACCGT';rows=[]
        for width in (1,3,6,10):
            old=encode_dna(ref[:width],bases)
            for t in range(len(ref)-width):
                old=np.roll(old*bases[ref[t]],-1)*np.roll(bases[ref[t+width]],width-1)
                np.testing.assert_array_equal(old,encode_dna(ref[t+1:t+width+1],bases))
            rows.append(dict(width=width,windows=len(ref)-width+1))
        MEASUREMENTS['dna_rolling_windows']=rows

    def test_07_rapid_unit_edit_distance_against_exhaustive_paths(self):
        # Restricted unit-cost global alignment, not RAPID affine-gap hardware.
        from functools import lru_cache
        @lru_cache(None)
        def brute(a,b):
            if not a:return len(b)
            if not b:return len(a)
            return min(1+brute(a[1:],b),1+brute(a,b[1:]),(a[0]!=b[0])+brute(a[1:],b[1:]))
        words=['']+[''.join(w) for n in range(1,4) for w in product('AC',repeat=n)]
        for a,b in product(words,repeat=2):self.assertEqual(edit_distance(a,b),brute(a,b))
        MEASUREMENTS['example_unit_edit_distance']=edit_distance('ATGTTATA','ATCGTCC')

    def test_08_workload_markov_probability(self):
        P=np.array([[.8,.2,0],[.1,.7,.2],[0,.3,.7]])
        p=np.array([0.,1.,0.]);states=[]
        for _ in range(20):
            p=p@P
            self.assertAlmostEqual(float(p.sum()),1.)
            self.assertTrue(np.all(p>=0));states.append(p.tolist())
        MEASUREMENTS['illustrative_markov_forecast']=states[-1]

    def test_09_scrimp_stochastic_multiply_requires_independence(self):
        # Cartesian product gives exact independent bipolar probabilities.
        a=[-1,1,1,1];b=[-1,-1,1,1]
        mean=lambda xs:sum(xs)/len(xs)
        self.assertEqual(mean([x*y for x,y in product(a,b)]),mean(a)*mean(b))
        self.assertNotEqual(mean([x*x for x in a]),mean(a)**2)

    def test_10_cd_is_not_elementwise_binding(self):
        self.assertEqual(mul(e(1),e(1)),{0:-1})
        # Componentwise e1*e1 would be e1, a different operation.
        self.assertNotEqual(mul(e(1),e(1)),e(1))

    def test_11_gpu_distinction(self):
        gpu_replay(self,[(e(1),e(1))],2)
