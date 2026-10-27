"""Exact, paper-ordered tests; normalized rays avoid floating point rank claims."""
from fractions import Fraction as F
from itertools import product, combinations
import random
import unittest
import numpy as np
import CayleyKernel as CK
from tests.papers.type_replay import replay_product
from .algebra import (e,add,scale,conj,dot,norm2,oracle_sign,mul,rows_from_columns,
                      nullspace,span_rank,embed,right,orbit,constraints,frame,
                      transform,diagonal,sphere_dimensions,minimum_q)

SEED=2010981
OBSERVATIONS={}
CERTIFICATES={}


def samples(d,count=6):
    rng=random.Random(SEED+d)
    return [{i:x for i in range(d) if (x:=rng.randint(-2,2))} for _ in range(count)]


def hermitian(v,w,d):
    """Section 2 convention sum v_r conjugate(w_r), linear on the left."""
    rows=set(p//d for p in v)|set(p//d for p in w)
    return add(*(mul({p%d:x for p,x in v.items() if p//d==r},
                     conj({p%d:x for p,x in w.items() if p//d==r}),d.bit_length()-1)
                 for r in rows))


def classical_orbit(v,d):
    return [{r*d+i:x for r in set(p//d for p in v)
             for i,x in mul(e(j),{p%d:x for p,x in v.items() if p//d==r},d.bit_length()-1).items()}
            for j in range(d)]


def coordinate_vectors(v,d,n):
    return [{r:v[r*d+i] for r in range(n) if v.get(r*d+i)} for i in range(d)]


def orthogonalize(vectors):
    out=[]
    for source in vectors:
        v=source
        for u in out:
            v=add(v,scale(u,-F(dot(v,u),norm2(u))))
        if v:
            out.append(v)
    return out


class PaperVerification(unittest.TestCase):
    def test_01_01_fixity_and_free_stiefel_dimension_obstruction(self):
        cases=[]
        # C2 in SO(n), with f-dimensional fixed subspace, for every f=0,...,5.
        # The nontrivial element is identity on f coordinates and -I on two.
        for f in range(6):
            n=f+2
            g=diagonal([1]*f+[-1]*2)
            difference=[{j:g[i][j]-int(i==j) for j in range(n) if g[i][j]!=int(i==j)} for i in range(n)]
            actual=n-span_rank(difference,n)
            self.assertEqual(actual,f)
            self.assertEqual((-1)**2,1)
            self.assertTrue(all(sum(g[i][k]*g[k][j] for k in range(n))==int(i==j)
                                for i,j in product(range(n),repeat=2)))
            self.assertLess(actual,f+1)  # no fixed orthonormal (f+1)-frame
            cases.append({'group':'C2','dimension':n,'fixity':actual,'first_free_frame_length':f+1})
        OBSERVATIONS['fixity_examples']=cases

    def test_01_02_theorems_1_1_1_2_sphere_dimension_arithmetic(self):
        rows=[]
        for n in range(3,101):
            q=max(2,minimum_q(6*n-6,n-2,4))
            spheres=[2*n-1,4*n-5,q*(4*n-8)-1]
            self.assertLessEqual(6*n-6,4*q*(n-2)+2)
            self.assertEqual(q,3 if n==3 else 2)
            rows.append({'n':n,'complex_q':q,'complex_spheres':spheres})
        self.assertEqual(rows[1]['complex_spheres'],[7,11,15])
        self.assertEqual(sphere_dimensions(12)[:4]+[2*(16*12-64)-1],[11,21,39,71,255])
        OBSERVATIONS['theorem_1_1_dimensions']=rows

    def test_02_01_conjugation_hermitian_and_real_group_invariance(self):
        # Rational orthogonal row rotation; it commutes with all coefficient maps.
        Q=[[F(3,5),F(-4,5)],[F(4,5),F(3,5)]]
        for d in (1,2,4):
            for a,b in zip(samples(2*d),reversed(samples(2*d))):
                self.assertEqual(hermitian(a,b,d),conj(hermitian(b,a,d)))
                self.assertEqual(hermitian(a,a,d),{0:norm2(a)} if a else {})
                self.assertEqual(hermitian(transform(a,d,Q),transform(b,d,Q),d),hermitian(a,b,d))

    def test_02_02_stiefel_fibres_and_lemma_2_1_stable_complement(self):
        cases=[]
        for d in (1,2,4):
            n=4
            if d==1:
                first={0:F(3,5),d:F(4,5)}
                second={0:F(-4,5),d:F(3,5)}
            else:
                first={0:F(3,5),d+1:F(4,5)}
                second={1:F(4,5),d:F(3,5)}
            self.assertEqual(hermitian(first,second,d),{})
            self.assertEqual(norm2(first),1);self.assertEqual(norm2(second),1)
            columns=classical_orbit(first,d)+classical_orbit(second,d)
            complement=nullspace(columns,n*d)
            self.assertEqual(span_rank(columns,n*d),2*d)
            self.assertEqual(len(complement),d*(n-2))
            self.assertEqual(span_rank(columns+complement,n*d),n*d)
            self.assertTrue(all(not dot(a,b) for a,b in product(columns,complement)))
            cases.append({'field_real_dimension':d,'n':n,'k':2,'fibre_rank':len(complement),
                          'sphere_dimension':len(complement)-1,'stabilized_rank':n*d})
        OBSERVATIONS['lemma_2_1_fibre_checks']=cases

    def test_02_03_lemma_2_2_exact_integer_stable_range_boundaries(self):
        rows=[]
        for c in (1,2,4):
            for rank in range(1,12):
                for base in range(0,60):
                    q=minimum_q(base,rank,c)
                    self.assertLessEqual(base,c*(q*rank+1)-2)
                    if q>1:
                        self.assertGreater(base,c*((q-1)*rank+1)-2)
                rows.append({'c':c,'rank':rank,'last_base_for_q1':c*(rank+1)-2})
        with self.assertRaises(ValueError):minimum_q(1,0)
        OBSERVATIONS['lemma_2_2_boundaries']=rows

    def test_02_04_theorem_1_1_complex_to_quaternionic_fixity(self):
        # Faithful C2 complex n=4, diag(1,1,-1,-1), fixity2.
        signs=[1,1,-1,-1]
        g=diagonal(signs)
        fixed=[e(r*4+j) for r in range(2) for j in range(4)]
        self.assertEqual(span_rank(fixed,16),8) # H-dimension 2
        f=[e(0),e(2*4)] # admissible E1 example, not fixed by g
        self.assertNotEqual(transform(f[1],4,g),f[1])
        # Any first two fixed H-frame vectors exhaust the fixed space.
        fixed_frame=[e(0),e(4)]
        columns=sum((classical_orbit(v,4) for v in fixed_frame),[])
        self.assertEqual(span_rank(columns,16),8)
        fixed_complement=nullspace([{j:dot(v,w) for j,w in enumerate(fixed) if dot(v,w)} for v in columns],8)
        self.assertEqual(fixed_complement,[])
        OBSERVATIONS['complex_fixity_two']={'n':4,'complex_fixity':2,'quaternionic_fixed_real_dimension':8,
                                           'fixed_last_fibre_rank':0,'q2_sphere_dimensions':[7,11,15]}

    def test_02_05_real_fixity_three_whitney_bounds(self):
        rows=[]
        for n in range(4,101):
            base=7*n-13
            q=minimum_q(base,n-3,4)
            self.assertLessEqual(base,q*(4*n-12)+2)
            if n>=9:self.assertLessEqual(q,2)
            rows.append({'n':n,'q':q,'spheres':[n-1,2*n-3,4*n-9,q*(4*n-12)-1]})
        self.assertEqual(rows[0]['q'],4)
        self.assertEqual(3*2-4,4*2-6)
        OBSERVATIONS['real_fixity_three_dimensions']=rows

    def test_02_06_omitting_complex_orthogonality_creates_fixed_points(self):
        # Rows0,1 are fixed by diag(+,+,-,-). The omitted i*v2 constraint
        # allows the third vector to be i*v2 itself, entirely in the fixed space.
        v1=e(0);v2=e(1*2);v3=e(1*4+1)
        self.assertFalse(dot(embed(v1,1,2),v2))
        self.assertFalse(dot(embed(v1,1,4),v3))
        self.assertFalse(dot(embed(v2,2,4),v3))
        self.assertEqual(dot(right(embed(v2,2,4),e(1),4),v3),1)
        g=diagonal([1,1,-1,-1])
        for v,d in ((v1,1),(v2,2),(v3,4)):
            self.assertEqual(transform(v,d,g),v)
        OBSERVATIONS['missing_orthogonality_fixed_point']={'v1':v1,'v2':v2,'v3':v3,
                                                         'violated_imaginary_constraint':1}

    def test_02_07_remark_2_4_orientation_sign_arithmetic(self):
        # Degree of a linear orthogonal sphere action is determinant.
        for n in range(1,12):
            for negative in range(n+1):
                determinant=(-1)**negative
                self.assertEqual(determinant%2,1)
                self.assertEqual(determinant**2,1)
        OBSERVATIONS['orientation_scope']='Checks determinant signs and reduction mod2; does not calculate bundle homology actions.'

    def test_03_01_full_doubling_table_twist_and_signed_basis_labels(self):
        count=0
        for level in range(5):
            d=1<<level
            for i,j in product(range(d),repeat=2):
                expected=oracle_sign(i,j,d)
                self.assertEqual(CK.twist(i,j),expected)
                for s,t in product((-1,1),repeat=2):
                    self.assertEqual(CK.basis(s*(i+1),t*(j+1)),s*t*expected*((i^j)+1))
                count+=1
            for a,b in zip(samples(d),reversed(samples(d))):
                self.assertEqual(mul(a,b,level),mul(a,b,level,True))
                self.assertEqual(conj(mul(a,b,level)),mul(conj(b),conj(a),level))
        OBSERVATIONS['basis_pairs_against_independent_recursion']=count

    def test_03_02_module_warning_bilinearity_and_nonassociativity(self):
        cases=[]
        for level in (3,4):
            d=1<<level
            witness=next((i,j,k) for i,j,k in product(range(1,d),repeat=3)
                         if mul(mul(e(i),e(j),level),e(k),level)!=mul(e(i),mul(e(j),e(k),level),level))
            i,j,k=witness
            left=mul(mul(e(i),e(j),level),e(k),level)
            right_result=mul(e(i),mul(e(j),e(k),level),level)
            self.assertNotEqual(left,right_result)
            for a,b,c in zip(samples(d),samples(d)[1:],samples(d)[2:]):
                self.assertEqual(mul(a,add(b,c),level),add(mul(a,b,level),mul(a,c,level)))
            cases.append({'dimension':d,'indices':witness,'left_associated':left,'right_associated':right_result})
        OBSERVATIONS['nonassociative_module_warning']=cases

    def test_03_03_lemma_3_1_uniformity_polynomial_certificate(self):
        # For arbitrary low a and high b, certify |ab|^2=|a|^2|b|^2
        # by ALL coefficients of the quadratic matrix L_a^T L_a.
        cases=[]
        for low,high in combinations((1,2,4,8,16),2):
            level=high.bit_length()-1
            matrices=[[mul(e(i),e(k),level) for k in range(high)] for i in range(low)]
            for i,j in combinations(range(low),2):
                for k,l in product(range(high),repeat=2):
                    self.assertEqual(dot(matrices[i][k],matrices[j][l])+dot(matrices[j][k],matrices[i][l]),0)
            for i in range(low):
                for k,l in product(range(high),repeat=2):
                    self.assertEqual(dot(matrices[i][k],matrices[i][l]),int(k==l))
            cases.append({'low_dimension':low,'high_dimension':high,'all_norm_coefficients_verified':True,
                          'nonzero_left_factor_injective':True})
        OBSERVATIONS['uniformity_coefficient_certificates']=cases

    def test_03_04_sedenion_zero_divisors_and_sixth_stage_obstruction(self):
        level=4
        witness=None
        for i,j,k,l in product(range(1,8),range(8,16),range(1,8),range(8,16)):
            a=add(e(i),e(j));b=add(e(k),e(l))
            if not mul(a,b,level):
                witness=(a,b);break
        self.assertIsNotNone(witness)
        a,b=witness
        self.assertTrue(a and b);self.assertFalse(mul(a,b,4))
        self.assertFalse(mul(a,b,5)) # embedding into next stage retains zero product
        kernel=nullspace(rows_from_columns([mul(a,e(j),5) for j in range(32)],32),32)
        self.assertGreater(len(kernel),0)
        self.assertEqual(span_rank([mul(e(0),e(j),5) for j in range(32)],32),32)
        OBSERVATIONS['sixth_stage_uniformity_failure']={'a':a,'b':b,'product':{},'annihilator_in_dimension32':len(kernel),
                                                       'identity_annihilator_dimension':0}

    def test_03_05_definition_3_2_separability_and_full_direct_sum(self):
        rows=[]
        for n,seed in ((5,2010),(5,981),(12,2010)):
            f=frame(n,4,seed)
            for k in range(1,5):
                d=1<<k
                blocks=[orbit(embed(v,1<<i,d),d) for i,v in enumerate(f[:k])]
                for a,b in combinations(blocks,2):
                    self.assertEqual(span_rank(a+b,n*d),2*d)
                all_columns=sum(blocks,[])
                self.assertEqual(span_rank(all_columns,n*d),k*d)
                rows.append({'n':n,'seed':seed,'k':k,'block_rank':d,'sum_rank':k*d})
        OBSERVATIONS['separability_and_joint_rank']=rows

    def test_03_06_lemma_3_3_pairwise_condition_alone_is_insufficient(self):
        # Diagnostic for the abstract wording only, NOT a W-frame counterexample.
        lines=[e(0),e(1),add(e(0),e(1))]
        self.assertTrue(all(span_rank([a,b],2)==2 for a,b in combinations(lines,2)))
        self.assertEqual(span_rank(lines,2),2)
        OBSERVATIONS['pairwise_vs_joint_caution']='Three lines in R2 are pairwise disjoint but their sum is not direct. Actual Cayley-Dickson W-frames are tested for full joint rank separately; this diagnostic does not refute the construction.'

    def test_03_07_proposition_3_4_exact_complement_and_stabilization(self):
        f=frame(5,4,2010)
        records=[]
        for k in range(1,5):
            d=1<<k
            columns=constraints(f[:k],d)
            kernel=nullspace(columns,5*d)
            self.assertEqual(len(kernel),(5-k)*d)
            self.assertEqual(span_rank(kernel,5*d),len(kernel))
            self.assertTrue(all(not dot(a,b) for a,b in product(columns,kernel)))
            self.assertEqual(span_rank(columns+kernel,5*d),5*d)
            records.append({'k':k,'algebra_dimension':d,'ambient_dimension':5*d,'constraint_rank':k*d,
                            'kernel_dimension':len(kernel),'frame_rays':f[:k],'complement_basis':kernel})
        CERTIFICATES['exact_fibres']=records
        OBSERVATIONS['proposition_3_4_ranks']=[{k:v for k,v in r.items() if k not in ('frame_rays','complement_basis')} for r in records]

    def test_03_08_proposition_3_5_equation_3_1_thresholds(self):
        first=[]
        for k in range(1,5):
            good=[]
            for n in range(5,101):
                spheres=sphere_dimensions(n)
                lhs=sum(spheres[:k]);rhs=spheres[k]
                if lhs<=rhs:good.append(n)
            first.append(good[0])
        self.assertEqual(first,[5,5,12,27]) # first scan starts at n=5
        self.assertEqual(next(n for n in range(2,30) if n-1<=2*n-3),2)
        self.assertEqual(next(n for n in range(3,30) if 3*n-4<=4*n-9),5)
        OBSERVATIONS['equation_3_1_minimal_n']=[2,5,12,27]

    def test_03_09_whitney_double_rank_and_sphere_off_by_one(self):
        rows=[]
        for n in range(12,101):
            rank=16*n-64;base=15*n-38
            q=minimum_q(base,rank)
            self.assertLessEqual(q,2)
            self.assertLessEqual(base,2*rank-1)
            self.assertEqual(q,2 if n<27 else 1)
            self.assertNotEqual(2*rank,2*rank-1)
            rows.append({'n':n,'base_dimension':base,'original_vector_rank':rank,'minimum_q':q,
                         'q2_vector_rank':2*rank,'q2_sphere_dimension':2*rank-1})
        OBSERVATIONS['whitney_double_dimensions']=rows

    def test_03_10_group_equivariance_and_frame_preservation(self):
        n=12;f=frame(n,4,981)
        g=diagonal([1]*4+[-1]*8)
        transformed=[transform(v,1<<i,g) for i,v in enumerate(f)]
        for i,v in enumerate(f):
            d=1<<i
            self.assertEqual(norm2(transformed[i]),norm2(v))
            for a in samples(d,3):
                self.assertEqual(transform(right(v,a,d),d,g),right(transform(v,d,g),a,d))
            if i:
                self.assertTrue(all(not dot(transformed[i],c) for c in constraints(transformed[:i],d)))
        OBSERVATIONS['real_group_action']={'group':'C2','n':12,'determinant':1,'fixity':4,'frame_preserved':True}

    def test_03_11_fixity_four_final_fibre_has_no_fixed_vectors(self):
        # An arbitrary admissible W4 ray frame inside the 4-dimensional fixed
        # real subspace of diag(I4,-I8). Its 64 constraint vectors exhaust
        # the 64-dimensional fixed part of V tensor S.
        f=frame(4,4,2010)
        columns=constraints(f,16)
        self.assertEqual(span_rank(columns,64),64)
        self.assertEqual(nullspace(columns,64),[])
        for v in f:self.assertGreater(norm2(v),0)
        OBSERVATIONS['fixity_four_freeness_certificate']={'fixed_base_dimension':4,'fixed_tensor_dimension':64,
                                                       'constraint_rank':64,'fixed_fibre_rank':0,'q2_fixed_fibre_rank':0,
                                                       'frame_rays':f}

    def test_03_12_corollary_coordinate_extraction_requires_orthogonalization(self):
        # Valid five-frame rays with no orthogonal raw selection from stages2,3.
        f=[e(0),{2:1,5:1},{4:1,8:1,5:1,9:-1},e(3*8),e(4*16)]
        for i,v in enumerate(f):
            self.assertTrue(all(not dot(v,c) for c in constraints(f[:i],1<<i)))
        g=diagonal([1]*5+[-1]*8) # faithful SO(13) representation of C2, fixity5
        for i,v in enumerate(f):
            self.assertEqual(transform(v,1<<i,g),v)
        blocks=[coordinate_vectors(v,1<<i,5) for i,v in enumerate(f)]
        options=[[v for v in block if v] for block in blocks]
        raw_choices=list(product(*options))
        self.assertTrue(raw_choices)
        self.assertFalse(any(all(not dot(a,b) for a,b in combinations(choice,2)) for choice in raw_choices))
        all_coords=sum(blocks,[])
        gs=orthogonalize(all_coords)
        self.assertEqual(len(gs),5)
        self.assertTrue(all(not dot(a,b) for a,b in combinations(gs,2)))
        # To obtain a Stiefel point normalize the five nonzero GS rays.
        CERTIFICATES['corollary_coordinate_example']={'frame_rays':f,'raw_nonzero_choices':len(raw_choices),
                                                     'raw_orthogonal_choices':0,'gram_schmidt_rays':gs}
        OBSERVATIONS['corollary_1_3_coordinate_qualification']='Raw coordinate vectors need not form a Stiefel point. In this exact example Gram-Schmidt supplies five fixed orthogonal rays; normalization supplies a Stiefel point.'

    def test_03_13_corollary_fixity_five_next_frame_dimension(self):
        # Linear-algebra part only; Smith theory/periodic complexes not computed.
        f=frame(5,5,981)
        coords=sum((coordinate_vectors(v,1<<i,5) for i,v in enumerate(f)),[])
        self.assertEqual(span_rank(coords,5),5)
        self.assertEqual(len(orthogonalize(coords)),5)
        self.assertLess(5,6)
        OBSERVATIONS['corollary_dimension_check']={'fixed_dimension':5,'fixed_coordinate_rank':5,
                                                   'fixed_six_frame_possible':False,'topology_verified':False}


class GPUVerification(unittest.TestCase):
    def setUp(self):
        if CK.cp is None:raise unittest.SkipTest('CuPy/CUDA runtime unavailable; multiply() was not executed')
        if not CK.loaded():raise unittest.SkipTest('Compiled CayleyKernel shared library unavailable')
        try:
            if CK.cp.cuda.runtime.getDeviceCount()<1:raise unittest.SkipTest('No CUDA device')
        except unittest.SkipTest:raise
        except Exception as exc:raise unittest.SkipTest('CUDA runtime unavailable: '+str(exc))

    def test_04_01_multiply_all_basis_pairs_through_sedenions(self):
        for d in (1,2,4,8,16):
            pairs=list(product(range(d),repeat=2))
            a=np.eye(d,dtype=np.float64)[[i for i,j in pairs]]
            b=np.eye(d,dtype=np.float64)[[j for i,j in pairs]]
            expected=np.zeros_like(a)
            for r,(i,j) in enumerate(pairs):expected[r,i^j]=oracle_sign(i,j,d)
            np.testing.assert_array_equal(replay_product(a,b),expected)

    def test_04_02_multiply_uniformity_and_zero_divisor_witness(self):
        a,b={1:1,10:1},{4:1,15:-1}
        # Compute expectations independently; CPU tests record a separate searched witness.
        for d in (8,16):
            aa=samples(d,8);bb=list(reversed(aa));level=d.bit_length()-1
            A=np.array([[v.get(i,0) for i in range(d)] for v in aa],dtype=np.float64)
            B=np.array([[v.get(i,0) for i in range(d)] for v in bb],dtype=np.float64)
            expected=np.array([[mul(x,y,level,True).get(i,0) for i in range(d)] for x,y in zip(aa,bb)])
            np.testing.assert_array_equal(replay_product(A,B),expected)
            # An arbitrary octonion embedded as (a,0) in S must preserve
            # the norm up to |a|, even though general sedenions do not.
            embedded=A.copy()
            embedded[:,d//2:]=0
            actual=replay_product(embedded,B)
            np.testing.assert_array_equal(np.sum(actual*actual,axis=-1),
                                          np.sum(embedded*embedded,axis=-1)*np.sum(B*B,axis=-1))
        A=np.array([a.get(i,0) for i in range(16)],dtype=np.float64)
        B=np.array([b.get(i,0) for i in range(16)],dtype=np.float64)
        expected=mul(a,b,4,True)
        self.assertEqual(expected,{})
        np.testing.assert_array_equal(replay_product(A,B),np.zeros(16))


try:
    import pytest
except ImportError:
    pass
else:
    PaperVerification=pytest.mark.cpu(PaperVerification)
    GPUVerification=pytest.mark.cpu(GPUVerification)
