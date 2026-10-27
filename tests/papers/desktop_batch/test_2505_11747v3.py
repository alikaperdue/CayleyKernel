"""Wilmot: exhaustive finite triads, subspaces and two-term zero products."""
from collections import Counter
from functools import lru_cache
from itertools import combinations,product
from fractions import Fraction as Q
import math,random,unittest
import CayleyKernel as CK
from .common import add,e,mul,neg,scale,Algebra,gpu_replay
from tests.papers.arxiv_2608_28176.algebra import line
try:
    import pytest
    pytestmark=pytest.mark.cpu
except ImportError:pass
MEASUREMENTS={}


@lru_cache(65536)
def sign(i,j):return CK.twist(i,j)


def associative(i,j,k):return sign(i,j)*sign(i^j,k)==sign(j,k)*sign(i,j^k)


def kind(b,c,d):
    flags=(not associative(b,d,c),not associative(b,c,d),not associative(c,b,d))
    return {(0,0,0):'H',(1,0,0):'A',(0,1,0):'B',(0,0,1):'C',(1,1,1):'X'}.get(flags,'INVALID')


def support_zero_pairs(n,params=None):
    """No theorem assumed: expand all four terms of XOR-compatible pairs.

    Quotient the relative-sign alternatives only for the paper support count;
    also retain every signed projective edge, including both relative signs.
    """
    dim=1<<n;A=Algebra(n,gammas=params,reference=params is not None)
    sg=(lambda i,j:A.mul(e(i),e(j))[i^j]) if params is not None else sign
    groups={};records=[];edges=0
    for i,j in combinations(range(1,dim),2):
        if params is not None or sg(i,i)==sg(j,j):groups.setdefault(i^j,[]).append((i,j))
    for group in groups.values():
        for (i,j),(k,l) in combinations(group,2):
            patterns=[]
            for s,t in product((-1,1),repeat=2):
                if sg(i,k)+s*t*sg(j,l)==0 and t*sg(i,l)+s*sg(j,k)==0:
                    patterns.append((s,t))
            if patterns:records.append((i,j,k,l,patterns));edges+=len(patterns)
    return records,edges


def graded(s):
    sign=-1 if s.startswith('-') else 1
    return {sum(1<<(int(c)-1) for c in s.lstrip('-')):sign}


class Sections(unittest.TestCase):
    def test_08_gpu_complex_split_signatures(self):
        from tests.papers.type_replay import replay_signatures
        replay_signatures(self,(g for n in range(1,5) for g in product((-1,1),repeat=n)))

    def test_02_graded_basis_and_blade_laws(self):
        for n in range(1,9):
            for i in range(1,1<<n):
                self.assertEqual(sign(i,i),-1)
                self.assertEqual(mul(e(i),e(1<<n)),e(i+(1<<n)))
            for i,j in combinations(range(1,1<<n),2):
                self.assertEqual(sign(i,j),-sign(j,i))
                self.assertTrue(associative(i,i,j));self.assertTrue(associative(j,i,i))

    def test_03_1_theorem_5_and_table_2_triads(self):
        rows=[]
        for n in range(2,8):
            d=1<<n;counts=Counter(kind(*t) for t in combinations(range(1,d),3))
            self.assertNotIn('INVALID',counts)
            expected=(d-1)*(d-2)//6
            self.assertEqual(counts['H'],expected)
            self.assertEqual(sum(counts.values()),math.comb(d-1,3))
            rows.append(dict(level=n,counts=dict(counts)))
            if n==4:
                self.assertEqual(counts,Counter(H=35,A=84,C=84,X=252))
                self.assertNotEqual(counts,Counter(H=35,A=84,C=112,X=224))
        MEASUREMENTS['exhaustive_triads']=rows

    def test_03_2_theorem_7_cycle_silos(self):
        names=['AAA','BBA','ACC','XBB','BXC','CAB','CCX','XXX']
        expected=[[0,0,0,0,0,0,0,8],[28,0,0,0,0,0,28,64],
                  [252,168,84,0,0,0,336,400],[1988,1848,1344,672,168,0,2660,2480]]
        rows=[]
        for n in range(3,7):
            counts=Counter()
            for b,c in combinations(range(1,1<<n),2):
                bc=b^c
                if bc<=c:continue
                for d in range(bc+1,1<<n):counts[kind(b,c,d)+kind(b,bc,d)+kind(c,bc,d)]+=1
            self.assertEqual(counts,Counter(dict(zip(names,expected[n-3]))))
            rows.append(dict(level=n,silos=dict(counts)))
        MEASUREMENTS['cycle_silos']=rows

    def test_03_3_table_5_subalgebra_counts(self):
        rows=[]
        expected=[(1,0,0,0),(8,7,0,0),(50,63,42,0),(310,413,504,168)]
        labels={(0,0,0,28):'O',(12,0,12,4):'P4',(8,8,8,4):'P12',(7,10,7,4):'P14'}
        for n in range(3,7):
            spaces=set()
            for a,b,c in combinations(range(1,1<<n),3):
                if a^b^c:spaces.add(tuple(sorted((a,b,c,a^b,a^c,b^c,a^b^c))))
            counts=Counter()
            for space in spaces:
                kinds=Counter(kind(*t) for t in combinations(space,3))
                counts[labels.get(tuple(kinds[t] for t in ('A','B','C','X')),'OTHER')]+=1
            self.assertNotIn('OTHER',counts)
            self.assertEqual(tuple(counts[k] for k in ('O','P4','P12','P14')),expected[n-3])
            rows.append(dict(level=n,subalgebras=dict(counts)))
        MEASUREMENTS['subalgebras']=rows

    def test_03_4_actual_malcev_identity_vs_associative_exclusion(self):
        bracket=lambda x,y:scale(add(mul(x,y),neg(mul(y,x))),Q(1,2))
        jac=lambda x,y,z:add(bracket(bracket(x,y),z),bracket(bracket(y,z),x),bracket(bracket(z,x),y))
        for i,j,k in product(range(4),repeat=3):
            x,y,z=e(i),e(j),e(k)
            self.assertEqual(jac(x,y,bracket(x,z)),bracket(jac(x,y,z),x))
        # Associative quaternion commutators are Lie, hence also Malcev.
        # This checks the actual identity, not simplified (10)'s restricted triads.

    def test_04_1_theorem_10_counts_and_complete_expanded_products(self):
        rows=[]
        for n in range(3,8):
            records,edges=support_zero_pairs(n);N=(1<<n)-1
            expected=(N-1)*(N-3)*(N-7)//16
            self.assertEqual(len(records),expected)
            self.assertEqual(edges,2*expected)
            # Replay every sign pattern through basis products at manageable levels.
            if n<=5:
                for i,j,k,l,patterns in records:
                    for s,t in patterns:
                        x,y={i:1,j:s},{k:1,l:t}
                        self.assertEqual(mul(x,y),{});self.assertEqual(mul(y,x),{})
            rows.append(dict(level=n,paper_support_pairs=len(records),signed_projective_edges=edges))
        MEASUREMENTS['zero_divisor_counts']=rows

    def test_04_2_table_8_all_prime_and_mode_products(self):
        rows=[('1','2','34','-1234'),('1','12','34','234'),('1','3','24','1234'),
              ('1','13','24','-234'),('1','23','24','134'),('1','123','24','-34'),
              ('2','12','34','-134'),('2','3','14','-1234'),('2','13','14','234'),
              ('2','23','14','-134'),('2','123','14','34'),('12','3','14','234'),
              ('12','13','14','1234'),('12','23','14','-34'),('12','123','14','-134'),
              ('3','13','24','-124'),('3','23','14','124'),('3','123','14','-24'),
              ('13','23','14','24'),('13','123','14','124'),('23','123','24','124')]
        for bs,cs,ds,as_ in rows:
            b,c,d,a=map(graded,(bs,cs,ds,as_))
            self.assertEqual(a,mul(mul(b,c),d))
            db={i:abs(v) for i,v in mul(d,b).items()};ap=mul(mul(b,c),db)
            for x,y in [(add(a,b),add(c,d)),(add(neg(d),b),add(c,a)),
                        (add(ap,b),add(c,db)),(add(neg(db),b),add(c,ap))]:
                self.assertEqual(mul(x,y),{});self.assertEqual(mul(y,x),{})

    def test_05_split_trace_and_selected_pair_counts(self):
        for n in range(1,6):
            for params in product((-1,1),repeat=n):
                A=Algebra(n,gammas=params,reference=True)
                trace=sum(A.mul(e(i),e(i))[0] for i in range(1,1<<n))
                self.assertEqual(trace,1 if 1 in params else -((1<<n)-1))
        rows=[]
        for params,expected in [((1,)*3,12),((1,)*4,180),((-1,-1,-1,1),84)]:
            records,edges=support_zero_pairs(len(params),params)
            self.assertEqual(len(records),expected)
            rows.append(dict(parameters=params,support_pairs=len(records),signed_edges=edges))
        MEASUREMENTS['split_counts']=rows

    def test_06_extension_high_index_type_patterns(self):
        rng=random.Random(250511747)
        for n in (20,32,128,1024,4096):
            for _ in range(128):
                b,c,d=sorted(rng.sample(range(1,1<<20),3)) if n==20 else sorted(rng.getrandbits(n) or 1 for _ in range(3))
                self.assertNotEqual(kind(b,c,d),'INVALID')

    def test_07_gpu_table_8_example(self):
        gpu_replay(self,[({1:1,15:-1},{2:1,12:1})],16)
