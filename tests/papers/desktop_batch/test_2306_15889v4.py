"""Alternating roots, scalar norm equation, and doubling correspondence."""
from fractions import Fraction as Q
from itertools import product
import random
import unittest
from .common import add,conj,e,mul,neg,pair,scale,rank,close,gpu_replay
from tests.papers.arxiv_2608_28176.algebra import witness,Algebra
try:
    import pytest
    pytestmark=pytest.mark.cpu
except ImportError:
    pass
MEASUREMENTS={}


def norm(x):return sum(c*c for c in x.values())


def alt(coeff,x):
    """Use the norm-reduced definition displayed in Section 3."""
    N=norm(x)
    return add(*(scale(mul(a,x) if i%2 else a,N**(i//2)) for i,a in enumerate(coeff)))


def regular(coeff,x):
    power=e(0);out={}
    for a in coeff:
        out=add(out,mul(a,power));power=mul(power,x)
    return out


def friend(coeff,half,gamma=-1):
    result=[]
    for i,a in enumerate(coeff):
        c=scale(conj(a),Q(1,gamma**((i+1)//2)))
        result.append(pair({},c,half) if i%2 else c)
    return result


def norm_poly(coeff):
    """Norm of a central-variable polynomial, exact coefficients."""
    out=[0]*(max(1,2*len(coeff)-1))
    for i,a in enumerate(coeff):
        for j,b in enumerate(coeff):out[i+j]+=mul(conj(a),b).get(0,0)
    return out


def scalar_equation(coeff):
    even=norm_poly(coeff[::2]);odd=norm_poly(coeff[1::2]);odd=[0]+odd
    return [(even[i] if i<len(even) else 0)-(odd[i] if i<len(odd) else 0)
            for i in range(max(len(even),len(odd)))]


def alternating_roots(coeff):
    """Numerical Section 3 algorithm for real composition algebras only.

    Returns isolated roots and squared radii for entire spherical families.
    Repeated/ill-conditioned scalar roots need more careful numerical analysis.
    """
    import numpy as np
    p=np.trim_zeros(np.array(scalar_equation(coeff),dtype=float),'b')
    if not len(p):raise ValueError('identically zero norm equation needs separate analysis')
    roots=[];spheres=[];seen=[]
    for z in np.polynomial.polynomial.polyroots(p):
        if abs(z.imag)>1e-8 or z.real < -1e-8:continue
        N=max(0.,float(z.real))
        if any(abs(N-t)<1e-8 for t in seen):continue
        seen.append(N)
        odd=add(*(scale(a,N**i) for i,a in enumerate(coeff[1::2])))
        even=add(*(scale(a,N**i) for i,a in enumerate(coeff[::2])))
        if norm(odd)<1e-16:
            if norm(even)>1e-12:raise ArithmeticError('invalid scalar norm root')
            spheres.append(N)
        else:
            root=scale(mul(conj(odd),neg(even)),1/norm(odd))
            if norm(alt(coeff,root))>1e-12:raise ArithmeticError('root residual too large')
            roots.append(root)
    return roots,spheres


class Sections(unittest.TestCase):
    def test_02_quadratic_identity_and_power_associativity(self):
        rng=random.Random(230615889)
        for n in range(1,8):
            for _ in range(8):
                x={i:rng.randrange(-2,3) for i in range(1<<n)}
                x={i:c for i,c in x.items() if c}
                self.assertEqual(add(mul(x,x),scale(x,-2*x.get(0,0)),{0:norm(x)}),{})
                self.assertEqual(mul(mul(x,x),x),mul(x,mul(x,x)))

    def test_03_1_central_norm_polynomial_and_constructed_roots(self):
        rng=random.Random(230615890)
        for dim in (2,4,8):
            for _ in range(12):
                x={i:rng.randrange(-2,3) for i in range(dim)}
                coeff=[{}]+[{i:rng.randrange(-2,3) for i in range(dim)} for j in range(5)]
                coeff[0]=neg(alt(coeff,x))
                self.assertEqual(alt(coeff,x),{})
                p=scalar_equation(coeff);N=norm(x)
                self.assertEqual(sum(c*N**i for i,c in enumerate(p)),0)
                for subset in (coeff[::2],coeff[1::2]):
                    # Full convolution coefficients, including imaginary cancellations.
                    for k in range(2*len(subset)-1):
                        c=add(*(mul(conj(a),b) for i,a in enumerate(subset)
                                for j,b in enumerate(subset) if i+j==k))
                        self.assertFalse(any(c.get(index,0) for index in c if index))

    def test_03_2_examples_3_4_3_5_3_7_and_spherical_case(self):
        # Under the module convention i=e1, j=e2, ij=e3.
        coeff=[{0:-1,3:-1},e(1),e(0)]
        self.assertEqual(scalar_equation(coeff),[2,-3,1])
        for x in [e(2),add(e(1),e(2))]:self.assertEqual(alt(coeff,x),{})
        for i in range(4):self.assertTrue(alt([e(0),{},e(0)],e(i)))
        # f=x^2-1 has every norm-one element as an alternating root.
        for x in [e(i) for i in range(4)]+[{0:Q(3,5),2:Q(4,5)}]:
            self.assertEqual(alt([{0:-1},{},e(0)],x),{})

    def test_03_3_remark_3_6_nondivision_counterexample(self):
        a,b,c,d=witness(Algebra(4));a1=pair(a,b,8)
        columns=[mul(a1,e(i)) for i in range(16)];r=rank(columns)
        target=next(e(i) for i in range(16) if rank(columns+[e(i)])>r)
        self.assertLess(r,16)
        self.assertEqual(rank(columns+[target]),r+1)
        # f(x)=a1*x-target has odd degree but no root in the sedenions.
        MEASUREMENTS['remark_3_6_counterexample']={'linear_coefficient':a1,'target':target,'rank':r,'augmented_rank':r+1}

    def test_03_4_python_port_of_octave_root_algorithm(self):
        try:import numpy
        except ImportError:self.skipTest('NumPy unavailable for scalar polynomial roots')
        roots,spheres=alternating_roots([{0:-1,3:-1},e(1),e(0)])
        self.assertEqual(spheres,[]);self.assertEqual(len(roots),2)
        for x,y in zip(sorted(roots,key=norm),[e(2),add(e(1),e(2))]):close(self,x,y)
        roots,spheres=alternating_roots([{0:-1},{},e(0)])
        self.assertEqual(roots,[]);self.assertEqual(len(spheres),1);self.assertAlmostEqual(spheres[0],1.)
        roots,spheres=alternating_roots([e(0),{},e(0)])
        self.assertEqual((roots,spheres),([],[]))

    def test_04_theorem_4_1_and_root_correspondence(self):
        rng=random.Random(230615891)
        for n in range(1,8):
            h=1<<n
            for _ in range(8):
                x={i:rng.randrange(-1,2) for i in range(h)}
                coeff=[{i:rng.randrange(-1,2) for i in range(h)} for j in range(6)]
                z=pair({},x,h)
                self.assertEqual(regular(friend(coeff,h),z),conj(alt(coeff,x)))
                self.assertFalse(any(i>=h for i in regular(friend(coeff,h),z)))
                coeff[0]=add(coeff[0],neg(alt(coeff,x)))
                self.assertEqual(regular(friend(coeff,h),z),{})

    def test_05_theorem_5_2_requires_conjugate_lambda_for_g(self):
        rng=random.Random(230615892)
        for n in (2,3,4,5):
            h=1<<n;v=e(h)
            for _ in range(8):
                x={i:rng.randrange(-1,2) for i in range(h)};z=pair({},x,h)
                f=[{i:rng.randrange(-1,2) for i in range(h)} for j in range(5)]
                g=[{i:rng.randrange(-1,2) for i in range(h)} for j in range(5)]
                fc,gc=friend(f,h),friend(g,h)
                H=[add(a,mul(b,v)) for a,b in zip(fc,gc)]
                self.assertEqual(regular(H,z),pair(conj(alt(f,x)),conj(alt(g,conj(x))),h))
                f[0]=add(f[0],neg(alt(f,x)));g[0]=add(g[0],neg(alt(g,conj(x))))
                H=[add(a,mul(b,v)) for a,b in zip(friend(f,h),friend(g,h))]
                self.assertEqual(regular(H,z),{})

    def test_06_extension_sparse_correspondence_through_level_4096(self):
        for n in (8,20,128,1024,4096):
            half=1<<n
            x={1:2,(1<<(n-1))+3:1};a={2:1,(1<<(n-1))+5:-1}
            coeff=[e(3),a,{},e(0),a,e(1)]
            self.assertEqual(regular(friend(coeff,half),pair({},x,half)),conj(alt(coeff,x)))

    def test_07_gpu_root_and_friend_products(self):
        coeff=[{0:-1,3:-1},e(1),e(0)]
        gpu_replay(self,[(e(1),e(2)),(e(1),add(e(1),e(2))),
                         (friend(coeff,4)[1],pair({},e(2),4))],8)
