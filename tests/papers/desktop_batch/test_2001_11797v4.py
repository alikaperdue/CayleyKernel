"""VSA comparison ingredients and a deliberately separate CD extension."""
from itertools import product
import unittest
from .common import Algebra, add, conj, e, mul, neg, scale, pair, gpu_replay
from tests.papers.arxiv_2608_28176.algebra import witness
try:
    import numpy as np
except ImportError:
    np=None
try:
    import pytest
    pytestmark=pytest.mark.cpu
except ImportError:
    pass
MEASUREMENTS={}


@unittest.skipIf(np is None,'NumPy unavailable')
class Sections(unittest.TestCase):
    def test_02_1_similarity_bundling_and_binding_baselines(self):
        rng=np.random.default_rng(200111797)
        for dim in (128,1024,65536):
            a=rng.choice([-1,1],size=dim); b=rng.choice([-1,1],size=dim)
            np.testing.assert_array_equal(a*(a*b),b)
            bitsa=(a>0); bitsb=(b>0)
            np.testing.assert_array_equal((a*b)<0,bitsa^bitsb)
            self.assertAlmostEqual(float(a@b/dim),1-2*np.mean(bitsa!=bitsb))
            t=rng.uniform(-np.pi,np.pi,size=dim); u=rng.uniform(-np.pi,np.pi,size=dim)
            z=np.exp(1j*t)*np.exp(1j*u)
            np.testing.assert_allclose(z*np.exp(-1j*t),np.exp(1j*u),atol=2e-15)

    def test_02_5_analogy_map_i(self):
        rng=np.random.default_rng(200111798); dim=65536
        name,currency,capital,usa,dollar,wdc,mex,peso,mxc=rng.choice([-1,1],size=(9,dim)).astype(float)
        ru=name*usa+currency*dollar+capital*wdc
        rm=name*mex+currency*peso+capital*mxc
        answer=dollar*ru*rm
        candidates=np.stack([usa,dollar,wdc,mex,peso,mxc])
        scores=candidates@answer/dim
        self.assertEqual(int(scores.argmax()),4)
        MEASUREMENTS['map_i_analogy_scores']=scores.tolist()

    def test_03_1_bundling_capacity_small_reproduction(self):
        rng=np.random.default_rng(200111799); rows=[]
        for dim,k in product((128,1024,4096),(2,8,32)):
            memory=rng.choice([-1,1],size=(1000,dim)).astype(float)
            chosen=rng.choice(1000,k,replace=False)
            query=memory[chosen].sum(axis=0)
            recovered=np.argsort(memory@query)[-k:]
            accuracy=len(set(chosen)&set(recovered))/k
            rows.append(dict(dimension=dim,k=k,accuracy=accuracy))
            self.assertTrue(0<=accuracy<=1)
            # Independent scalar scoring checks the retrieval implementation.
            for i in (0,1,999):
                self.assertEqual(float(memory[i]@query),sum(float(a*b) for a,b in zip(memory[i],query)))
        MEASUREMENTS['map_i_capacity_one_trial_per_cell']=rows

    def test_03_2_cd_binding_exact_for_basis_roles_not_all_roles(self):
        x={i:i-5 for i in range(16) if i!=5}
        for n in (4,8,20,128,1024,4096):
            role=e((1<<(n-1))+3)
            self.assertEqual(mul(conj(role),mul(role,x)),x)
        a,b,c,d=witness(Algebra(4)); role=pair(a,b,8); lost=pair(c,d,8)
        self.assertTrue(role);self.assertTrue(lost)
        self.assertEqual(mul(role,lost),{})
        self.assertNotEqual(mul(conj(role),mul(role,lost)),scale(lost,2))
        MEASUREMENTS['cd_noninvertible_binding']={'role':role,'lost_filler':lost,'bound':{}}

    def test_03_3_cd_role_filler_retrieval(self):
        rng=np.random.default_rng(200111800); dim=128
        memory=[{i:int(v) for i,v in enumerate(rng.choice([-1,1],size=dim))} for _ in range(20)]
        roles=[e(1),e(2),e(3)]
        record=add(*(mul(roles[i],memory[i]) for i in range(3)))
        outcomes=[]
        for i in range(3):
            query=mul(conj(roles[i]),record)
            scores=[sum(query.get(j,0)*x.get(j,0) for j in range(dim)) for x in memory]
            outcomes.append(int(np.argmax(scores))==i)
            # Check bilinear decomposition, rather than demanding statistical success.
            expected=add(*(mul(conj(roles[i]),mul(roles[j],memory[j])) for j in range(3)))
            self.assertEqual(query,expected)
        MEASUREMENTS['cd_basis_role_retrieval']=outcomes

    def test_04_1_cd_ngram_parentheses_are_required(self):
        x,y,z=e(1),e(2),e(4)
        left=mul(mul(x,y),z);right=mul(x,mul(y,z))
        self.assertNotEqual(left,right)
        MEASUREMENTS['cd_trigram_parentheses']={'left':left,'right':right}

    def test_05_gpu_binding_counterexample(self):
        a,b,c,d=witness(Algebra(4))
        gpu_replay(self,[(pair(a,b,8),pair(c,d,8))],16)
