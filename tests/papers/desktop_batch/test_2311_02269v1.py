"""Independent Clifford blade algebra and the paper's modified products."""
from itertools import product,combinations
import unittest
from .common import add,conj,e,neg,scale,mul,Algebra,gpu_replay
try:
    import pytest
    pytestmark=pytest.mark.cpu
except ImportError:
    pass
MEASUREMENTS={}


def gp(x,y,signature):
    out={}
    for a,u in x.items():
        for b,v in y.items():
            swaps=sum((b&((1<<i)-1)).bit_count() for i in range(3) if a>>i&1)
            c=(-1)**swaps
            for i in range(3):
                if (a&b)>>i&1:c*=signature[i]
            k=a^b;out[k]=out.get(k,0)+u*v*c
    return {k:c for k,c in out.items() if c}


def invol(x,kind):
    def sign(mask):
        g=mask.bit_count()
        return {'reverse':(-1)**(g*(g-1)//2),'grade':(-1)**g,
                'clifford':(-1)**(g*(g+1)//2),'full':1 if not mask else -1}[kind]
    return {k:c*sign(k) for k,c in x.items()}


def parity(x,even):return {k:c for k,c in x.items() if (k.bit_count()%2==0)==even}


def bullet(x,y,signature,variant=1):
    a,b=parity(x,True),parity(x,False)
    c,d=parity(y,True),parity(y,False)
    return add(gp(a,c,signature),scale(gp(invol(d,'clifford'),b,signature),variant),
               gp(d,a,signature),gp(b,invol(c,'clifford'),signature))


def N(x,signature,variant=1):
    return sum(c*c*(1 if k.bit_count()%2==0 else variant)
               *product_sign(k,signature) for k,c in x.items())


def product_sign(mask,signature):
    s=1
    for i in range(3):
        if mask>>i&1:s*=signature[i]
    return s


def basis_map(signature,variant):
    m=lambda x,y:bullet(x,y,signature,variant)
    images=[e(0),e(3),e(6),m(e(3),e(6)),e(1)]
    images.extend(m(images[i],images[4]) for i in (1,2,3))
    return images


def translate(x,images):return add(*(scale(images[i],c) for i,c in x.items()))


class Sections(unittest.TestCase):
    def test_07_gpu_complex_split_signatures(self):
        from tests.papers.type_replay import replay_signatures
        replay_signatures(self,product((-1,1),repeat=3))

    def test_02_hurwitz_subalgebras_via_module_map(self):
        images=basis_map((1,1,1),1)
        for i,j in product(range(8),repeat=2):
            self.assertEqual(translate(mul(e(i),e(j)),images),bullet(images[i],images[j],(1,1,1)))

    def test_03_geometric_product_and_involutions(self):
        for sig in product((-1,1),repeat=3):
            for a,b,c in product(range(8),repeat=3):
                self.assertEqual(gp(gp(e(a),e(b),sig),e(c),sig),gp(e(a),gp(e(b),e(c),sig),sig))
            for a,b in product(range(8),repeat=2):
                x,y=e(a),e(b)
                for kind in ('reverse','clifford'):
                    self.assertEqual(invol(gp(x,y,sig),kind),gp(invol(y,kind),invol(x,kind),sig))
                self.assertEqual(invol(gp(x,y,sig),'grade'),gp(invol(x,'grade'),invol(y,'grade'),sig))

    def test_04_1_even_and_pseudoscalar_subalgebras(self):
        for sig in product((-1,1),repeat=3):
            I=e(7)
            self.assertEqual(gp(I,I,sig),{0:-sig[0]*sig[1]*sig[2]})
            for i in range(8):self.assertEqual(gp(I,e(i),sig),gp(e(i),I,sig))
            even=[0,3,6,5]
            for i,j in product(even,repeat=2):self.assertTrue(all(k in even for k in gp(e(i),e(j),sig)))

    def test_04_2_norm_composition_and_all_eight_signature_isomorphisms(self):
        forms=[e(i) for i in range(8)]+[add(e(i),e(j)) for i,j in combinations(range(8),2)]
        for sig,variant in product(product((-1,1),repeat=3),(-1,1)):
            images=basis_map(sig,variant)
            params=[-sig[0]*sig[1],-sig[1]*sig[2],-variant*sig[0]]
            A=Algebra(3,gammas=params,reference=True)
            for i,j in product(range(8),repeat=2):
                self.assertEqual(translate(A.mul(e(i),e(j)),images),bullet(images[i],images[j],sig,variant))
            for x,y in product(forms,repeat=2):
                z=bullet(x,y,sig,variant)
                self.assertEqual(N(z,sig,variant),N(x,sig,variant)*N(y,sig,variant))
                self.assertEqual(bullet(x,invol(x,'full'),sig,variant),{0:N(x,sig,variant)} if N(x,sig,variant) else {})
        MEASUREMENTS['signature_variants_checked']=16

    def test_04_3_equation_4_5_odd_part_counterexample(self):
        sig=(1,1,1);x=add(e(0),e(1),e(3));rev=invol(x,'reverse')
        lhs=parity(gp(x,rev,sig),False);rhs=parity(gp(rev,x,sig),False)
        self.assertNotEqual(lhs,rhs)
        MEASUREMENTS['equation_4_5_counterexample']={'x':x,'lhs':lhs,'rhs':rhs}

    def test_04_4_table_2_norm_signatures(self):
        results=[]
        for sig in [(1,1,1),(1,1,-1),(-1,-1,1),(-1,-1,-1)]:
            for variant in (1,-1):
                signs=[N(e(i),sig,variant) for i in range(8)]
                positive=signs.count(1)
                self.assertIn(positive,(4,8))
                self.assertEqual(positive==8,(sig==(1,1,1) and variant==1) or (sig==(-1,-1,-1) and variant==-1))
                results.append(dict(signature=sig,variant=variant,norm_signs=signs))
        MEASUREMENTS['table_2_norm_signatures']=results

    def test_05_extension_standard_subalgebra_embedding_through_level_4096(self):
        # Embed a known standard octonion subalgebra at dyadically shifted indices.
        from CayleyKernel import twist
        for n in (8,20,128,1024,4096):
            shift=n-3
            for i,j in product(range(8),repeat=2):
                self.assertEqual(twist(i<<shift,j<<shift),twist(i,j))

    def test_06_gpu_octonion_products(self):
        gpu_replay(self,[(e(i),e(j)) for i,j in product(range(8),repeat=2)],8)
