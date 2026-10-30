# SPDX-License-Identifier: Apache-2.0
"""Finite exact verification in paper order; extensions are named separately."""
import itertools as it
import random
import unittest
from fractions import Fraction
try:
    import numpy as np
    import sympy as s
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("paper companion requires NumPy and SymPy: " + str(exc))
import CayleyKernel as CK
from .algebra import (
    PAPER, TRIPLES, basis_product, mul, add, neg, scale, conj, e,
    vector, operator, operators, words, flat, span_rank, comm, centralizer,
    lie_model, physical, phases, embedding, hopf, oracle,
)
try:
    import pytest
    pytestmark = pytest.mark.cpu
except ImportError:
    pass

EVIDENCE = {}


class Sections(unittest.TestCase):
    def test_01_section1_clifford_0_8_and_bott_dimensions(self):
        lo,_=operators(8); z=s.zeros(8); eye=s.eye(8)
        gs=[s.BlockMatrix([[z,g],[g,z]]).as_explicit() for g in lo[1:]]
        gs.append(s.BlockMatrix([[z,eye],[-eye,z]]).as_explicit())
        for i,a in enumerate(gs):
            for j,b in enumerate(gs):
                self.assertEqual(a*b+b*a, -2*s.eye(16) if i==j else s.zeros(16))
        # Trace-orthogonal 256 words establish linear independence exactly.
        ws=np.array([list(flat(x)) for x in words(gs)],dtype=np.int64)
        np.testing.assert_array_equal(ws@ws.T,16*np.eye(256,dtype=np.int64))
        for n in range(5):
            self.assertEqual(16**(2*n),2**(8*n))
        EVIDENCE['clifford_0_8']={'generators':8,'independent_words':256,'matrix_size':16}

    def test_02_section3_1_left_right_maps_and_nonassociativity(self):
        for d in (1,2,4,8):
            units=PAPER if d==8 else [e(i) for i in range(d)]
            left,right=operators(d)
            for i,a in enumerate(units):
                for j,b in enumerate(units):
                    self.assertEqual(left[i]*vector(b,d),vector(mul(a,b),d))
                    self.assertEqual(right[i]*vector(b,d),vector(mul(b,a),d))
            self.assertEqual(sum(x!=y for x,y in zip(left,right)),0 if d<=2 else d-1)
            self.assertEqual(span_rank([x-y for x,y in zip(left,right)]),0 if d<=2 else d-1)
        lo,_=operators(8)
        self.assertNotEqual(lo[1]*lo[2],operator(mul(PAPER[1],PAPER[2]),8))
        a,b,c=lo[1],lo[2],lo[3]
        self.assertEqual((a*b)*c,a*(b*c))

    def test_03_section3_1_equation8_su3_derivations(self):
        l,_=operators(8)
        p=lambda i,j:l[i]*l[j]
        # Omit nonzero normalization factors, including sqrt(3): same real span.
        gs=[p(3,4)-p(1,5),p(1,4)+p(3,5),p(1,3)-p(4,5),
            -p(2,5)-p(4,6),p(2,4)-p(5,6),-p(1,6)-p(2,3),
            -p(1,2)-p(3,6),-p(1,3)-p(4,5)+2*p(2,6)]
        self.assertEqual(span_rank(gs),8)
        for g in gs:
            self.assertEqual(g.T,-g)
            self.assertEqual(g*vector(PAPER[7],8),s.zeros(8,1))
            for a,b in it.product(PAPER,repeat=2):
                ga={i:v for i,v in enumerate(g*vector(a,8)) if v}
                gb={i:v for i,v in enumerate(g*vector(b,8)) if v}
                self.assertEqual(g*vector(mul(a,b),8),vector(add(mul(ga,b),mul(a,gb)),8))
        self.assertEqual(span_rank(gs+[comm(a,b) for a,b in it.product(gs,repeat=2)]),8)

    def test_04_section3_2_fano_table_and_api_oracle(self):
        expected={(0,i):(1,i) for i in range(8)}
        expected.update({(i,0):(1,i) for i in range(8)})
        for i in range(1,8): expected[i,i]=(-1,0)
        for a,b,c in TRIPLES:
            for i,j,k in ((a,b,c),(b,c,a),(c,a,b)):
                expected[i,j]=(1,k); expected[j,i]=(-1,k)
        self.assertEqual(len(expected),64)
        for (i,j),(sign,k) in expected.items():
            self.assertEqual(mul(PAPER[i],PAPER[j]),scale(PAPER[k],sign))
        for n in range(7):
            for i in range(1<<n):
                for j in range(1<<n):
                    sign=oracle(i,j,n)
                    self.assertEqual(CK.twist(i,j),sign)
                    for a,b in it.product((-1,1),repeat=2):
                        self.assertEqual(basis_product(a*(i+1),b*(j+1)),a*b*sign*((i^j)+1))
        EVIDENCE['paper_basis']=[list(x.items()) for x in PAPER]

    def test_05_section3_2_direct_sum_componentwise(self):
        # Four block identities are orthogonal, unlike their shifted images in S.
        blocks=[8,4,2,1]
        ps=[]; offset=0
        for d in blocks:
            p=s.zeros(15)
            p[offset:offset+d,offset:offset+d]=s.eye(d)
            ps.append(p); offset+=d
        self.assertEqual(sum(ps,s.zeros(15)),s.eye(15))
        for i,p in enumerate(ps):
            for j,q in enumerate(ps): self.assertEqual(p*q,p if i==j else s.zeros(15))
        for d in blocks:
            a={i:i+1 for i in range(d)}; b={i:(-1)**i for i in range(d)}
            self.assertEqual(operator(a,d)*vector(b,d),vector(mul(a,b),d))

    def test_06_section3_3_multiplication_algebra_dimensions(self):
        rows=[]
        for d in (1,2,4,8):
            l,r=operators(d)
            # For O the six Clifford generators supply all 64 independent words.
            lb=words(l[1:7]) if d==8 else l
            rb=words(r[1:7]) if d==8 else r
            full=[a*b for a in l for b in r] if d!=8 else lb
            dims=[span_rank(lb),span_rank(rb),span_rank(full)]
            self.assertEqual(dims,{1:[1,1,1],2:[2,2,2],4:[4,4,16],8:[64,64,64]}[d])
            intersection=dims[0]+dims[1]-span_rank(lb+rb)
            self.assertEqual(intersection,{1:1,2:2,4:1,8:64}[d])
            rows.append({'dimension':d,'left_right_full':dims,'intersection':intersection})
        EVIDENCE['multiplication_algebras']=rows

    def test_07_section3_3_equation12_all_seven_units(self):
        l,r=operators(8)
        idx=lambda i:(i-1)%7+1
        for i in range(1,8):
            self.assertEqual(2*r[i],-l[i]+l[idx(i+1)]*l[idx(i+3)]
                             +l[idx(i+2)]*l[idx(i+6)]+l[idx(i+4)]*l[idx(i+5)])
            self.assertEqual(2*l[i],-r[i]+r[idx(i+3)]*r[idx(i+1)]
                             +r[idx(i+6)]*r[idx(i+2)]+r[idx(i+5)]*r[idx(i+4)])

    def test_08_section4_clifford_signatures_and_volumes(self):
        lh,rh=operators(4); lo,ro=operators(8)
        cases=[([lh[1],lh[2]],[-1,-1],lh[3]),
               ([rh[1],rh[2]],[-1,-1],-rh[3]),
               ([lh[1]*rh[i] for i in (1,2,3)]+[lh[2]],[1,1,1,-1],-lh[3]),
               (lo[1:7],[-1]*6,lo[7]),(ro[1:7],[-1]*6,-ro[7])]
        for gs,signs,vol in cases:
            d=gs[0].rows; actual=s.eye(d)
            for i,a in enumerate(gs):
                actual*=a
                for j,b in enumerate(gs):
                    self.assertEqual(a*b+b*a,2*signs[i]*s.eye(d) if i==j else s.zeros(d))
            self.assertEqual(actual,vol); self.assertEqual(vol*vol,-s.eye(d))
        # Eq.15: real End(C), two +1 generators and imaginary volume.
        a=s.diag(1,-1); b=s.Matrix([[0,1],[1,0]])
        self.assertEqual(a*b+b*a,s.zeros(2)); self.assertEqual((a*b)**2,-s.eye(2))
        self.assertEqual(span_rank(words([a,b])),4)

    def test_09_section5_centralizers_and_even_words(self):
        counts={}
        for d,both,want in ((2,False,2),(4,False,8),(8,False,32),(8,True,20)):
            bs=centralizer(d,both)
            self.assertEqual(len(bs),want)
            counts[str((d,both))]=len(bs)
            if d in (4,8) and not both:
                gs=operators(4)[0][1:3] if d==4 else operators(8)[0][1:7]
                # H full algebra uses the Cl(3,1) realization rather than Cl(0,2).
                if d==4:
                    l,r=operators(4); gs=[l[1]*r[i] for i in (1,2,3)]+[l[2]]
                even=[]
                for bits in it.product((0,1),repeat=len(gs)):
                    x=s.eye(d)
                    for use,g in zip(bits,gs):
                        if use:x*=g
                    if sum(bits)%2==0: even.append(x)
                self.assertEqual(span_rank(bs+even),want)
        EVIDENCE['centralizer_real_dimensions']=counts

    def test_10_section6_second_quaternion_centralizer(self):
        self.assertEqual(len(centralizer(4,True)),4)
        p,q,ph,qh=phases()
        self.assertEqual([p.rank(),q.rank(),ph.rank(),qh.rank()],[2,6,2,2])
        l,r=operators(4)
        bs=[s.eye(4),l[3],r[3],l[3]*r[3]]
        self.assertEqual(span_rank(bs+centralizer(4,True)),4)

    def test_11_section7_lie_jordan_splitting(self):
        for low,want in ((False,15),(True,13)):
            bs=lie_model(low); self.assertEqual(len(bs),want)
            self.assertEqual(span_rank(bs+[comm(a,b) for a,b in it.product(bs,repeat=2)]),want)
        # Exhaust a real basis of M3(C) to check all four bilinear rules, Eq.27.
        hs=[]; us=[]
        for i in range(3):
            a=s.zeros(3); a[i,i]=1; hs.append(a); us.append(s.I*a)
            for j in range(i+1,3):
                a=s.zeros(3); a[i,j]=a[j,i]=1; hs.append(a); us.append(s.I*a)
                a=s.zeros(3); a[i,j]=s.I; a[j,i]=-s.I; hs.append(a); us.append(s.I*a)
        for aa,bb,sign in ((hs,hs,1),(us,us,-1),(hs,us,-1),(us,hs,1)):
            for a,b in it.product(aa,bb):
                m=a*b+b*a.conjugate().T
                self.assertEqual(m.conjugate().T,sign*m)
        EVIDENCE['lie_dimensions_before_conditions']={'SM':15,'LE':13}

    def test_12_section8_1_xor_grading(self):
        for n in range(8):
            for i in range(1<<n):
                for j in range(1<<n): self.assertEqual(abs(basis_product(i+1,j+1))-1,i^j)
        # Under the convention conversion, paper labels are not literal binary grades.
        self.assertEqual(mul(PAPER[1],PAPER[2]),PAPER[4])

    def test_13_section8_2_full_symmetry_constraints(self):
        for low,want in ((False,12),(True,9)):
            bs=physical(low)
            self.assertEqual(len(bs),want)
            self.assertEqual(span_rank(bs+[comm(a,b) for a,b in it.product(bs,repeat=2)]),want)
            derived=[comm(a,b) for a,b in it.product(bs,repeat=2)]
            self.assertEqual(span_rank(derived),8 if low else 11)
        EVIDENCE['constrained_lie_dimensions']={'SM':12,'LE':9,'derived_SM':11,'derived_LE':8}

    def test_14_section8_3_projectors(self):
        for p,q in (phases()[:2],phases()[2:]):
            self.assertEqual(p*p,p); self.assertEqual(q*q,q)
            self.assertEqual(p*q,s.zeros(p.rows)); self.assertEqual(p+q,s.eye(p.rows))
            self.assertEqual(p.T,p); self.assertEqual(q.T,q)
        p,q,ph,qh=phases()
        self.assertEqual(p*vector(PAPER[7],8),vector(PAPER[7],8))
        self.assertEqual(ph*vector(e(3),4),vector(e(3),4))

    def test_15_sections8_4_8_5_trace_constraints_and_charges(self):
        po,qo,ph,qh=phases()
        y=s.diag(qo/3,s.eye(4)/2,s.eye(2),s.zeros(1))
        q=s.diag(qo/3,qh,s.eye(2),s.zeros(1))
        jo=operators(8)[0][7]; jh=operators(4)[0][3]; jc=operators(2)[0][1]
        J=s.diag(jo,jh,jc,s.zeros(1))
        for low,x in ((False,y),(True,q)):
            bs=physical(low)
            self.assertEqual(span_rank(bs+[J*x]),len(bs))
            self.assertEqual([(j*(j*p)).trace() for j,p in
                              ((jo,x[:8,:8]),(jh,x[8:12,8:12]),(jc,x[12:14,12:14]))],[-2,-2,-2])
            self.assertEqual(x.T,x)
        # Solve phase constraints independently, rather than assume charge coefficients.
        a,b,c=s.symbols('a b c')
        self.assertEqual(s.linsolve([3*a-2*b,2*b-c,c-1],(a,b,c)),s.FiniteSet((s.Rational(1,3),s.Rational(1,2),1)))
        self.assertEqual(s.linsolve([3*a-b,b-c,c-1],(a,b,c)),s.FiniteSet((s.Rational(1,3),1,1)))
        t3=q-y
        self.assertEqual(t3[8:12,8:12].trace(),0)
        self.assertEqual(t3[8:12,8:12].eigenvals(),{s.Rational(-1,2):2,s.Rational(1,2):2})
        self.assertEqual(span_rank(physical(False)+[J*t3]),12)
        EVIDENCE['charges']={'Y_complex_eigenvalues':['0','1/3','1/2','1'],
                             'Q_complex_eigenvalues':['0','1/3','1'],'real_trace':6}

    def test_16_sections8_6_8_7_scalar_shift_spectra_and_entropy(self):
        _,qo,_,qh=phases()
        y=s.diag(qo/3,s.eye(4)/2,s.eye(2),s.zeros(2))
        q=s.diag(qo/3,qh,s.eye(2),s.zeros(2))
        for x in (y,q):
            self.assertEqual(x.trace(),6)
            self.assertEqual((x-s.Rational(3,8)*s.eye(16)).trace(),0)
            for i,j in it.product(range(16),repeat=2):
                unit=s.zeros(16); unit[i,j]=1
                self.assertEqual(comm(x,unit),comm(x-s.Rational(3,8)*s.eye(16),unit))
        for n in (1,2,3):
            rho=s.eye(n)/n
            self.assertEqual(rho.trace(),1)
            self.assertEqual(sum(-v*s.log(v)*m for v,m in rho.eigenvals().items()),s.log(n))
        EVIDENCE['Y_real_spectrum']={str(k):v for k,v in y.eigenvals().items()}
        EVIDENCE['Q_real_spectrum']={str(k):v for k,v in q.eigenvals().items()}

    def test_16b_section8_6_left_right_complex_structures(self):
        l,r=operators(8); p,q,_,_=phases()
        self.assertEqual((r[7]-l[7])*p,s.zeros(8))
        self.assertEqual((r[7]+l[7])*q,s.zeros(8))
        self.assertEqual(l[7]**2,-s.eye(8)); self.assertEqual(r[7]**2,-s.eye(8))
        # The two choices reverse the complex orientation on the triplet space.

    def test_16c_section9_1_other_vector_space_embeddings(self):
        ts=embedding(); tc=ts.copy()
        # In C tensor O, scalar i times o has upper coefficients o, not conjugate(o).
        for j in range(8):tc[:,j]=vector(e(8+j),16)
        swap=s.BlockMatrix([[s.zeros(8),s.eye(8)],[s.eye(8),s.zeros(8)]]).as_explicit()
        td=swap*tc  # O direct-sum O: unshifted O in the first copy.
        for t in (tc,td):
            self.assertEqual(t.T*t,s.eye(16)); self.assertEqual(t[:,:15].rank(),15)
            for a in physical(False):
                transported=t*s.diag(a,s.zeros(1))*t.T
                self.assertEqual(transported*t[:,:15],t[:,:15]*a)
        # These checks concern vector-space embeddings and induced actions only.

    def test_17_section9_2_embedding_and_nested_subalgebras(self):
        t=embedding()
        self.assertEqual(t.T*t,s.eye(16))
        self.assertEqual(t[:,:15].rank(),15)
        for labels in ((0,7),(0,2,6,7),tuple(range(8))):
            allowed={next(iter(PAPER[j])) for j in labels}
            for a,b in it.product(labels,repeat=2):
                self.assertTrue(set(mul(PAPER[a],PAPER[b]))<=allowed)
        # Positive and negative controls: shifted summands are vector spaces only.
        image=t[:,0]; x={i:v for i,v in enumerate(image) if v}
        self.assertEqual(mul(x,x),{0:-1})
        self.assertNotEqual(vector(mul(x,x),16),image)
        EVIDENCE['embedding_columns']=[[int(v) for v in t[:,j]] for j in range(16)]

    def test_18_section9_2_equivariance_global_complex_structure(self):
        t=embedding(); j=operator(PAPER[7],16)
        self.assertEqual(j*j,-s.eye(16))
        for low in (False,True):
            for a in physical(low):
                base=s.diag(a,s.zeros(1)); transported=t*base*t.T
                self.assertEqual(transported*t[:,:15],t[:,:15]*a)
                self.assertEqual(transported*vector(e(0),16),s.zeros(16,1))
                self.assertEqual(comm(transported,j),s.zeros(16))

    def test_19_section9_3_peirce_blocks_and_grading(self):
        # Complex dimensions in Fig.6: 1+3+2+1+1=8 (SM); split H for LE.
        spectra={}
        for sizes,ys,qs in (((1,3,2,1,1),(0,Fraction(1,3),Fraction(1,2),1,0),None),
                            ((1,3,1,1,1,1),None,(0,Fraction(1,3),0,1,1,0))):
            offsets=np.cumsum((0,)+sizes)
            self.assertEqual(sum(sizes),8)
            self.assertEqual(sum(a*b for a,b in it.product(sizes,repeat=2)),64)
            # Every matrix-unit product either vanishes or has grade g_ij+g_jk=g_ik.
            for i,j,k in it.product(range(8),repeat=3):
                bi=int(np.searchsorted(offsets[1:],i,side='right'))
                bj=int(np.searchsorted(offsets[1:],j,side='right'))
                bk=int(np.searchsorted(offsets[1:],k,side='right'))
                grade=lambda a,b:(1<<a)^(1<<b)
                self.assertEqual(grade(bi,bj)^grade(bj,bk),grade(bi,bk))
                if len(sizes)==5:
                    labels=(1,2,4,8,0)
                    self.assertEqual((labels[bi]^labels[bj])^(labels[bj]^labels[bk]),labels[bi]^labels[bk])
            weights=ys or qs
            spectra['Y' if ys else 'Q']={str(v):sum(sizes[i]*sizes[j] for i,j in it.product(range(len(sizes)),repeat=2)
                                      if weights[i]-weights[j]==v)
                                      for v in sorted({a-b for a,b in it.product(weights,repeat=2)})}
        EVIDENCE['commutator_charge_spectra_complex_multiplicities']=spectra

    def test_20_section9_4_tensor_clifford_and9_5_e8_dimensions(self):
        self.assertEqual((2*4*8)**2,16**2*4**2)
        self.assertEqual((4*8)**2*2,16**2*4*2)
        self.assertEqual(28+28+3*8*8,248)
        self.assertEqual(8*(4+2+1+1),64)
        # Eq.53 is a vector space decomposition, so no unspecified e8 bracket assumed.

    def test_21_section9_6_hopf_norm_identity(self):
        rng=random.Random(141078)
        for d in (1,2,4,8):
            for _ in range(40):
                a={i:rng.randrange(-3,4) for i in range(d)}
                b={i:rng.randrange(-3,4) for i in range(d)}
                scalar,z=hopf(a,b)
                total=sum(v*v for v in a.values())+sum(v*v for v in b.values())
                self.assertEqual(scalar*scalar+sum(v*v for v in z.values()),total*total)
        EVIDENCE['hopf_norm_samples']={'per_dimension':40,'dimensions':[1,2,4,8],'exact':True}

    def test_21b_section9_6_symbolic_norm_composition(self):
        # Polynomial equality establishes the norm identity for every real input
        # in these fixed algebras, not only the seeded examples in test21.
        for d in (1,2,4,8):
            aa=s.symbols('a:'+str(d)); bb=s.symbols('b:'+str(d))
            product=mul(dict(enumerate(aa)),dict(enumerate(bb)))
            defect=sum(v*v for v in product.values())-sum(v*v for v in aa)*sum(v*v for v in bb)
            self.assertEqual(s.expand(defect),0)
        EVIDENCE['symbolic_norm_composition_dimensions']=[1,2,4,8]

    def test_22_section9_6_hopf_north_pole_derivative(self):
        for d in (1,2,4,8):
            # Normalized path (1,t*b)/sqrt(1+t^2*N(b)); derivative (0,2*conj(b)).
            columns=[]
            t=s.Symbol('t',real=True)
            for i in range(d):
                scalar,z=hopf(e(0),scale(e(i),t))
                denominator=1+t*t
                self.assertEqual(s.diff(scalar/denominator,t).subs(t,0),0)
                derivative={k:s.diff(v/denominator,t).subs(t,0) for k,v in z.items()}
                self.assertEqual(derivative,scale(conj(e(i)),2))
                columns.append(vector(derivative,d))
            self.assertEqual(s.Matrix.hstack(*columns).rank(),d)
        self.assertEqual(8+4+2+1,15)

    def test_23_extension_higher_hopf_obstruction(self):
        # Deterministically find a sedenion zero product; normalize both halves by 1/2.
        a=add(e(1),e(10)); partner=None
        for j,k in it.combinations(range(16),2):
            for sign in (-1,1):
                b=add(e(j),scale(e(k),sign))
                if not mul(a,b):partner=b;break
            if partner is not None:break
        self.assertIsNotNone(partner)
        b=conj(partner)
        witnesses=[]
        for n in (4,5,8,16,20,32,128,1024):
            # Low-index embedding preserves the exact obstruction at every tested level.
            scalar,z=hopf(scale(a,Fraction(1,2)),scale(b,Fraction(1,2)))
            self.assertEqual(scalar,0); self.assertEqual(z,{})
            self.assertTrue(all(i<1<<n for i in a.keys()|b.keys()))
            witnesses.append({'level':n,'input_norm_sq':1,'image_norm_sq':0})
        EVIDENCE['hopf_obstruction']={'a':a,'conjugated_partner':b,'levels':witnesses}

    def test_24_extension_high_bit_twists_and_embedded_model(self):
        rng=random.Random(2026141078); rows=[]
        for n in (4,8,16,20,32,64,128,256,1024):
            for _ in range(64):
                i,j=rng.getrandbits(n),rng.getrandbits(n)
                sign=oracle(i,j,n)
                self.assertEqual(CK.twist(i,j),sign)
                self.assertEqual(basis_product(i+1,j+1),sign*((i^j)+1))
            # Embedded 16-dimensional model: all 256 basis products, fixed exact result.
            for i,j in it.product(range(16),repeat=2):
                self.assertEqual(CK.twist(i,j),oracle(i,j,n))
            rows.append({'level':n,'high_bit_samples':64,'embedded_pairs':256})
        EVIDENCE['sparse_high_levels']=rows

    def test_25_extension_basis_projectors_up_to_level12(self):
        rows=[]
        for n in range(1,13):
            d=1<<n; u=d//2
            # L_u R_u is diagonal: evaluate every column without d^2 allocation.
            p1=p2=0
            for j in range(d):
                result=mul(e(u),mul(e(j),e(u)))
                sign=result[j]
                self.assertIn(sign,(-1,1))
                self.assertEqual(mul(e(u),mul(e(u),e(j))),{j:-1})
                p1+=(1-sign)//2; p2+=(1+sign)//2
            self.assertEqual((p1,p2),(2,d-2))
            rows.append({'level':n,'dimension':d,'projector_ranks':[p1,p2],
                         'predicted_joint_centralizer_real_dim':2+(d-2)**2//2})
        # Predictions above use simultaneous complex eigenspaces; exhaust linear solve at d=16.
        self.assertEqual(len(centralizer(16,True)),100)
        EVIDENCE['higher_projector_ranks']=rows

    def test_25b_extension_high_bit_quaternion_subalgebras(self):
        rng=random.Random(14107825); rows=[]
        for n in (4,8,20,64,128,1024):
            for _ in range(32):
                i,j=rng.sample(range(1,1<<n),2) if n<32 else (rng.getrandbits(n)|1,rng.getrandbits(n)&~1|2)
                units=[e(0),e(i),e(j),mul(e(i),e(j))]
                self.assertEqual(mul(units[1],units[2]),units[3])
                self.assertEqual(mul(units[2],units[3]),units[1])
                self.assertEqual(mul(units[3],units[1]),units[2])
                for a,b,c in it.product(units,repeat=3):
                    self.assertEqual(mul(mul(a,b),c),mul(a,mul(b,c)))
            rows.append({'level':n,'quaternion_subalgebras':32,'associators_per_algebra':64})
        EVIDENCE['high_bit_quaternion_subalgebras']=rows


class GPUVerification(unittest.TestCase):
    def setUp(self):
        if CK.cp is None:self.skipTest('multiply(): CuPy unavailable')
        if not CK.loaded():self.skipTest('multiply(): compiled kernel unavailable')
        try:count=CK.cp.cuda.runtime.getDeviceCount()
        except CK.cp.cuda.runtime.CUDARuntimeError as exc:self.skipTest('multiply(): '+str(exc))
        if count<1:self.skipTest('multiply(): no CUDA device')

    def test_26_multiply_all_paper_basis_pairs_and_nested_operators(self):
        cp=CK.cp
        pairs=list(it.product(PAPER,repeat=2))
        for dtype in (np.float32,np.float64,np.complex64,np.complex128):
            a=np.array([list(vector(x,8)) for x,y in pairs],dtype=dtype)
            b=np.array([list(vector(y,8)) for x,y in pairs],dtype=dtype)
            expected=np.array([list(vector(mul(x,y),8)) for x,y in pairs],dtype=dtype)
            np.testing.assert_array_equal(cp.asnumpy(CK.multiply(cp.asarray(a),cp.asarray(b))),expected)
        # Actual nested products used by Eq.12, not a CPU substitute for multiply().
        batch=cp.eye(8,dtype=cp.float64)
        units={i:cp.asarray(np.array(list(vector(PAPER[i],8)),dtype=float)) for i in range(1,8)}
        idx=lambda i:(i-1)%7+1
        for i in range(1,8):
            left=-CK.multiply(units[i],batch)
            for a,b in ((i+1,i+3),(i+2,i+6),(i+4,i+5)):
                left+=CK.multiply(units[idx(a)],CK.multiply(units[idx(b)],batch))
            np.testing.assert_array_equal(cp.asnumpy(left),2*cp.asnumpy(CK.multiply(batch,units[i])))

    def test_27_multiply_sparse_witness_dense_arrays_through_level20(self):
        cp=CK.cp; rows=[]
        # Full-length arrays, but sparse coefficients. Dense multiplication still executes.
        for n in (4,8,12,16,18,20):
            d=1<<n
            pairs=[(add(e(1),e(10)),add(e(4),scale(e(15),-1))),
                   ({0:1,d-1:2,10:1},{1:-2,d//2:3,15:-1})]
            x=np.zeros((2,d)); y=np.zeros_like(x); expected=np.zeros_like(x)
            for row,(a,b) in enumerate(pairs):
                for i,v in a.items():x[row,i]=v
                for i,v in b.items():y[row,i]=v
                for i,v in mul(a,b).items():expected[row,i]=v
            actual=cp.asnumpy(CK.multiply(cp.asarray(x),cp.asarray(y)))
            np.testing.assert_array_equal(actual,expected)
            rows.append({'level':n,'dimension':d,'dtype':'float64','batch':2})
        EVIDENCE['gpu_dense_array_replay']=rows
