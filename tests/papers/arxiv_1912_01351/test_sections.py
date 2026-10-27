"""Finite checks in paper order. Counterexample tests assert DISAGREEMENT.

Run with python3 -m tests.papers.arxiv_1912_01351.run.
"""
from fractions import Fraction as F
from itertools import product
import random
import unittest

import numpy as np
import CayleyKernel as CK
from tests.papers.type_replay import replay_product
from .algebra import (PAPER_TABLE, PAPER_TO_KERNEL, e, add, scale, conj, norm,
                      mul, recursive, inverse, q, dq, transformed, nmul, nq,
                      ndq, lattice, zeta, wp, jacobian, dirac)

OBSERVATIONS = {}


class PaperVerification(unittest.TestCase):
    def test_02_01_basis_twist_and_independent_doubling(self):
        for d in (1, 2, 4, 8, 16, 32):
            for i, j in product(range(d), repeat=2):
                expected = recursive(e(i, d), e(j, d))
                self.assertEqual(mul(e(i, d), e(j, d)), expected)
                self.assertEqual(expected[i ^ j], CK.twist(i, j))
                for s, t in product((-1, 1), repeat=2):
                    self.assertEqual(CK.basis(s*(i+1), t*(j+1)),
                                     s*t*CK.twist(i, j)*((i ^ j)+1))

    def test_02_02_printed_octonion_table(self):
        for i, j in product(range(1, 8), repeat=2):
            label = PAPER_TABLE[i-1][j-1]
            expected = scale(e(PAPER_TO_KERNEL[abs(label)-1]), 1 if label > 0 else -1)
            self.assertEqual(mul(e(PAPER_TO_KERNEL[i]), e(PAPER_TO_KERNEL[j])), expected)

    def test_02_03_involution_quadratic_power_and_inverse(self):
        rng = random.Random(191201351)
        for d in (1, 2, 4, 8, 16, 32):
            for _ in range(12):
                a, b = ([rng.randint(-2, 2) for _ in range(d)] for _ in range(2))
                self.assertEqual(conj(mul(a, b)), mul(conj(b), conj(a)))
                self.assertEqual(mul(a, conj(a)), scale(e(0, d), norm(a)))
                self.assertEqual(add(add(mul(a, a), scale(a, -2*a[0])),
                                     scale(e(0, d), norm(a))), [0]*d)
                self.assertEqual(mul(mul(a, a), mul(a, a)), mul(mul(mul(a, a), a), a))
                if norm(a):
                    self.assertEqual(mul(a, inverse(a)), e(0, d))
                    self.assertEqual(mul(inverse(a), a), e(0, d))

    def test_02_04_composition_alternativity_moufang_and_conjugate_rule(self):
        rng = random.Random(1912)
        for d in (2, 4, 8):
            for _ in range(20):
                a, b, c = ([rng.randint(-2, 2) for _ in range(d)] for _ in range(3))
                self.assertEqual(norm(mul(a, b)), norm(a)*norm(b))
                self.assertEqual(mul(mul(a, a), b), mul(a, mul(a, b)))
                self.assertEqual(mul(b, mul(a, a)), mul(mul(b, a), a))
                self.assertEqual(mul(mul(a, b), mul(c, a)), mul(a, mul(mul(b, c), a)))
                self.assertEqual(mul(mul(a, b), a), mul(a, mul(b, a)))
                target = scale(a, norm(b))
                self.assertEqual(mul(mul(a, conj(b)), b), target)
                self.assertEqual(mul(conj(b), mul(b, a)), target)

    def test_02_05_associativity_boundary_and_sedenion_failures(self):
        for i, j, k in product(range(4), repeat=3):
            self.assertEqual(mul(mul(e(i, 4), e(j, 4)), e(k, 4)),
                             mul(e(i, 4), mul(e(j, 4), e(k, 4))))
        self.assertNotEqual(mul(mul(e(1), e(2)), e(4)), mul(e(1), mul(e(2), e(4))))
        # Find a reproducible sparse zero divisor without assuming its sign.
        for i, j, k, s in product(range(1, 8), range(9, 16), range(1, 8), (-1, 1)):
            a = add(e(i, 16), e(j, 16))
            l = i ^ j ^ k
            if not 8 < l < 16:
                continue
            b = add(e(k, 16), scale(e(l, 16), s))
            if not any(mul(a, b)):
                self.assertNotEqual(norm(mul(a, b)), norm(a)*norm(b))
                self.assertNotEqual(mul(a, mul(a, b)), mul(mul(a, a), b))
                OBSERVATIONS['sedenion_zero_divisor'] = {'a': a, 'b': b}
                return
        self.fail('no sedenion zero divisor found')

    def test_03_01_trace_norm_polarization_and_brandt_conditions(self):
        for a, b in (([1,2,3,4,1,0,2,1], [2,0,1,3,2,1,0,1]),
                     (scale(e(1), F(1,2)), scale(e(2), F(1,3)))):
            self.assertEqual(norm(add(a,b)), norm(a)+norm(b)+2*sum(x*y for x,y in zip(a,b)))
            self.assertEqual(2*mul(a,b)[0], 2*sum(x*y for x,y in zip(a,conj(b))))
            self.assertEqual(norm(mul(a,b)), norm(a)*norm(b))
        # Each is integral by Definition 3.1, their sum is not.
        a = [F(1,2)]*4 + [0]*4
        b = e(1)
        self.assertEqual((2*a[0], norm(a), 2*b[0], norm(b)), (1,1,0,1))
        # This particular pair passes; contrast another integral unit below.
        c = scale(add(e(1), e(2)), 1/np.sqrt(2))
        self.assertAlmostEqual(norm(c), 1)
        self.assertFalse(float(2*sum(x*y for x,y in zip(c,b))).is_integer())
        self.assertFalse(float(norm(add(c,b))).is_integer())

    def test_03_02_exact_multiquadratic_lattice_structure_constants(self):
        # Kernel binary index is the subset mask for sqrt(2),sqrt(3),sqrt(5).
        ms = (2,3,5)
        for i, j in product(range(8), repeat=2):
            overlap = np.prod([ms[b] for b in range(3) if (i & j) >> b & 1], dtype=int)
            coefficient = CK.twist(i,j)*int(overlap)
            # Product sqrt(mi)*sqrt(mj) reduces exactly by their overlap.
            mi = int(np.prod([ms[b] for b in range(3) if i >> b & 1], dtype=int))
            mj = int(np.prod([ms[b] for b in range(3) if j >> b & 1], dtype=int))
            mk = int(np.prod([ms[b] for b in range(3) if (i^j) >> b & 1], dtype=int))
            self.assertEqual(coefficient**2*mk, mi*mj)
            self.assertEqual(abs(CK.basis(i+1,j+1))-1, i^j)
        # Lattice coordinates give integral trace and norm; conjugation preserves Z^8.
        coords = [1,2,-1,3,1,-2,0,1]
        weights = [1,2,3,6,5,10,15,30]
        self.assertIsInstance(sum(c*c*w for c,w in zip(coords,weights)), int)
        self.assertEqual(2*coords[0], 2)

    def test_03_03_rational_coefficients_do_not_ensure_lattice_closure(self):
        # omega1=sqrt(2)/2 e1 has square -1/2, outside scalar Z.
        self.assertEqual(-2*F(1,2)**2, F(-1,2))
        # Arbitrary independent integer scales also need divisibility conditions.
        scales = (1,2,1,1,1,1,1,1)
        self.assertEqual(F(3*scales[2]*scales[3],scales[2^3]), F(3,2))
        # Closure under multiplication is not the printed commutativity condition.
        self.assertNotEqual(mul(e(1),e(2)), mul(e(2),e(1)))

    def test_04_01_01_cauchy_kernel_left_right_regular_and_derivatives(self):
        z = [1,2,-1,1,0,1,0,-1]
        left = right = [0]*8
        for j in range(8):
            left = add(left,mul(e(j),dq(z,j)))
            right = add(right,mul(dq(z,j),e(j)))
        self.assertEqual(left,[0]*8)
        self.assertEqual(right,[0]*8)
        zn = np.array(z,dtype=float)
        np.testing.assert_allclose(dirac(jacobian(nq,zn)), 0, atol=1e-12)
        for j in (1,4,7):
            np.testing.assert_allclose(dirac(jacobian(lambda x:ndq(x,j),zn)),0,atol=1e-11)
        lap = sum((nq(zn+1e-3*np.eye(8)[j])-2*nq(zn)+nq(zn-1e-3*np.eye(8)[j]))/1e-6
                  for j in range(8))
        np.testing.assert_allclose(lap,0,atol=1e-9)

    def test_04_01_02_fueter_linear_polynomials_and_module_obstruction(self):
        for j in range(1,8):
            for a in map(e,range(8)):
                self.assertEqual(add(scale(mul(e(j),a),-1),mul(e(j),a)),[0]*8)
        # f=x1*a+x2*b, b=e2*(e1*a), has Df=0; f*c need not.
        for i,k in product(range(1,8),repeat=2):
            a,c=e(i),e(k)
            b=mul(e(2),mul(e(1),a))
            if any(add(mul(e(1),a),mul(e(2),b))):
                continue
            residual=add(mul(e(1),mul(a,c)),mul(e(2),mul(b,c)))
            if any(residual):
                OBSERVATIONS['regular_module_obstruction']={'a':a,'b':b,'c':c,'D_fc':residual}
                return
        self.fail('module obstruction not found')

    def test_04_01_03_cauchy_sphere_center_and_singularity_scaling(self):
        rng=np.random.default_rng(1912)
        w=rng.normal(size=(64,8));w/=np.linalg.norm(w,axis=1)[:,None]
        # q(z-w) at z=0: use inward normal -w, giving +f(0).
        np.testing.assert_allclose(nmul(nq(-w),-w),np.tile(e(0),(64,1)),atol=2e-15)
        for t in (F(1,2),F(1,3),F(1,4)):
            self.assertEqual(q(scale(e(1),t)),scale(q(e(1)),t**-7))
        # Area S^7 = 2*pi^4/Gamma(4)=pi^4/3, consistent with paper.
        import math
        self.assertEqual(2*math.pi**4/math.gamma(4),math.pi**4/3)

    def test_04_01_04_higher_algebra_regular_need_not_be_harmonic(self):
        # f=x1*xj*a - x0*x1*(ej*a) - x0*xj*(e1*a) + x0^2*d.
        # Df=0 coefficientwise; Laplacian=2*d may be nonzero in sedenions.
        for i,j,k in product(range(1,16),repeat=3):
            if i == j:
                continue
            a=e(k,16);u=e(i,16);v=e(j,16)
            b=scale(mul(v,a),-1);c=scale(mul(u,a),-1)
            two_d=scale(add(mul(u,b),mul(v,c)),-1)
            if any(two_d):
                self.assertEqual(two_d,add(recursive(u,recursive(v,a)),recursive(v,recursive(u,a))))
                self.assertEqual(add(b,mul(v,a)),[0]*16)
                self.assertEqual(add(c,mul(u,a)),[0]*16)
                self.assertEqual(add(two_d,add(mul(u,b),mul(v,c))),[0]*16)
                # Independently evaluate the polynomial and exact central
                # differences (exact derivatives for this quadratic).
                def polynomial(x):
                    return add(add(scale(a,x[i]*x[j]),scale(b,x[0]*x[i])),
                               add(scale(c,x[0]*x[j]),scale(two_d,F(x[0]**2,2))))
                point=list(range(1,17))
                df=[0]*16;lap=[0]*16
                for axis in range(16):
                    plus=add(point,e(axis,16))
                    minus=add(point,scale(e(axis,16),-1))
                    derivative=scale(add(polynomial(plus),scale(polynomial(minus),-1)),F(1,2))
                    df=add(df,recursive(e(axis,16),derivative))
                    lap=add(lap,add(add(polynomial(plus),polynomial(minus)),scale(polynomial(point),-2)))
                self.assertEqual(df,[0]*16)
                self.assertEqual(lap,two_d)
                OBSERVATIONS['nonharmonic_regular_sedenion_polynomial']={
                    'i':i,'j':j,'a_index':k,'laplacian':two_d}
                return
        self.fail('no nonharmonic regular polynomial found')

    def test_04_02_01_wp_even_zeta_odd_and_corrected_derivative(self):
        points=lattice(1,np.sqrt([1,2,3,6,5,10,15,30]))
        z=np.array([.23,.17,-.11,.09,.13,-.07,.05,.03])
        for correction in (-1,1):
            np.testing.assert_allclose(zeta(-z,points,correction),-zeta(z,points,correction),atol=1e-8)
        for j in (1,4,7):
            np.testing.assert_allclose(wp(-z,points,j),wp(z,points,j),atol=1e-7)
            derivative=jacobian(lambda x:zeta(x,points,-1),z,h=2e-6)[j]
            np.testing.assert_allclose(derivative,wp(z,points,j),rtol=2e-8,atol=1e-3)

    def test_04_02_02_printed_zeta_plus_sign_counterexample(self):
        # The discrepancy is exact for EACH nonzero summand, no truncation claim.
        w=[2,1,0,1,0,0,0,0];j=1
        # d/dxj of +Vj(z)*qj(w) is +qj(w), whereas wp needs -qj(w).
        printed_constant=add(dq(w,j),dq(w,j))
        self.assertTrue(any(printed_constant))
        OBSERVATIONS['printed_zeta_derivative_error_single_summand']=[str(v) for v in printed_constant]
        points=lattice(1,np.sqrt([1,2,3,6,5,10,15,30]))
        self.assertGreater(np.linalg.norm(2*np.sum(ndq(points,j),axis=0)),.1)
        z=[1,1,0,0,0,0,0,0]
        def summand(r,sign):
            omega=scale(w,r)
            ans=add(q(add(z,omega)),scale(q(omega),-1))
            for k in range(1,8):
                v=add(scale(e(0),z[k]),scale(e(k),-z[0]))
                ans=add(ans,scale(mul(v,dq(omega,k)),sign))
            return ans
        errors={}
        for sign in (1,-1):
            vals=[float(norm(summand(r,sign)))**.5 for r in (16,32,64)]
            errors[str(sign)]={'norms':vals,'decay_ratios':[vals[i]/vals[i+1] for i in (0,1)]}
        self.assertGreater(errors['-1']['decay_ratios'][-1],400)
        self.assertLess(errors['1']['decay_ratios'][-1],300)
        OBSERVATIONS['zeta_summand_decay']=errors

    def test_04_02_03_finite_lattice_periodicity_and_quasiperiod_experiment(self):
        z=np.array([.31,.27,-.19,.21,.13,-.17,.11,.09])
        period=np.eye(8)[1]
        observations=[]
        for cutoff in (1,2):
            points=lattice(cutoff,np.ones(8))
            eta=2*zeta(period/2,points,-1)
            observations.append({
                'cutoff':cutoff,'nonzero_lattice_points':len(points),
                'wp_period_defect':float(np.linalg.norm(wp(z+period,points,1)-wp(z,points,1))),
                'zeta_quasiperiod_defect':float(np.linalg.norm(zeta(z+period,points,-1)-zeta(z,points,-1)-eta)),
                'eta1':eta.tolist(),
            })
        self.assertTrue(all(np.isfinite(row['wp_period_defect']) for row in observations))
        OBSERVATIONS['finite_lattice_period_experiment']=observations

    def test_04_03_01_kernel_conjugation_and_one_sided_transformations(self):
        rng=random.Random(49)
        for _ in range(16):
            l,m,z=([rng.randint(-2,2) for _ in range(8)] for _ in range(3))
            self.assertEqual(q(mul(l,mul(z,m))),mul(mul(q(m),q(z)),q(l)))
            self.assertEqual(mul(m,q(mul(z,m))),scale(q(z),F(1,norm(m)**3)))
            self.assertEqual(mul(q(mul(l,z)),l),scale(q(z),F(1,norm(l)**3)))

    def test_04_03_02_proposition_4_9_exact_counterexample(self):
        # Kernel (e1,e2,e4) = PAPER (e1,e2,e3); all norms equal one.
        l,m,z=e(1),e(2),e(4)
        for fn in (mul,recursive):
            self.assertEqual(transformed(z,l,m,fn),e(4))
        self.assertEqual(q(z),scale(e(4),-1))
        self.assertNotEqual(transformed(z,l,m),q(z))
        OBSERVATIONS['proposition_4_9']={'lambda':l,'mu':m,'z':z,
                                       'lhs':transformed(z,l,m),'rhs':q(z)}

    def test_04_03_03_exhaust_basis_triples_and_associative_controls(self):
        passed=failed=0
        for l,m,z in product(map(e,range(8)),repeat=3):
            ok=transformed(z,l,m)==q(z)
            passed+=ok;failed+=not ok
        self.assertEqual((passed,failed),(344,168))
        for i,j,k in product(range(4),repeat=3):
            self.assertEqual(transformed(e(k),e(i),e(j)),q(e(k)))
        OBSERVATIONS['proposition_4_9_basis_triples']={'passed':passed,'failed':failed}

    def test_04_03_04_proposition_4_11_regular_counterexample(self):
        # For unit lambda=e1, mu=e2 the transform is A*z/|z|^8,
        # A is diagonal (+ on 0, - on 1..3, + on 4..7).
        columns=[transformed(e(j),e(1),e(2)) for j in range(8)]
        z=e(0)
        jac=[]
        for j in range(8):
            jac.append(add(columns[j],scale(columns[0],-8*z[j])))
        left=right=[0]*8
        for j in range(8):
            left=add(left,mul(e(j),jac[j]));right=add(right,mul(jac[j],e(j)))
        self.assertEqual(left,[-8,0,0,0,0,0,0,0])
        self.assertEqual(right,left)
        OBSERVATIONS['proposition_4_11_D_at_1']=left

    def test_04_03_05_cm_map_inverse_period_coefficients_and_count(self):
        for l,m in ((e(1),e(2)),(scale(e(1),2),e(2)),(add(e(0),e(1)),e(2))):
            rows=np.array([mul(l,mul(e(j),m)) for j in range(8)],dtype=float)
            n=norm(l)*norm(m)
            np.testing.assert_allclose(rows@rows.T,n*np.eye(8),atol=1e-12)
            self.assertAlmostEqual(abs(np.linalg.det(rows)),n**4,places=7)
            self.assertTrue(np.all(rows==np.round(rows)))
            for w in map(e,range(8)):
                v=mul(mul(inverse(l),w),inverse(m))
                self.assertEqual(mul(l,mul(v,m)),w)
        points=list(product((F(0),F(1,2)),repeat=8))
        self.assertEqual(len(set(points)),256)
        self.assertTrue(all(all((2*x).denominator==1 for x in v) for v in points))
        self.assertEqual(len(points),norm(scale(e(0),2))**4)

    def test_04_03_06_parentheses_and_laurent_cancellation_counterexample(self):
        l,m,z=e(1),e(2),e(4)
        self.assertNotEqual(mul(l,mul(z,m)),mul(mul(l,z),m))
        for t in (F(1,2),F(1,4),F(1,8)):
            defect=add(transformed(scale(z,t),l,m),scale(q(scale(z,t)),-1))
            self.assertEqual(defect,scale(z,2*t**-7))
        # These are legitimate nonreal multiplicators of the integer octonion lattice.
        # Their CM map is unimodular, V={0}; equation (2)'s f has a pole at 0.
        self.assertEqual(norm(l)*norm(m),1)

    def test_04_03_07_cofactor_elimination_for_arbitrary_data_only(self):
        # Audit Eq (8) and the extraction of Ci from Eq (7); NOT the trace theorem.
        w=np.eye(8);w[1,2]=2;w[3,5]=-1
        cof=np.linalg.det(w)*np.linalg.inv(w).T
        np.testing.assert_allclose(w.T@cof,np.eye(8)*np.linalg.det(w),atol=1e-14)
        rng=np.random.default_rng(414)
        constants=rng.integers(-3,4,size=(7,8))
        v=np.array([w[h,j]*np.eye(8)[0]-w[h,0]*np.eye(8)[j]
                    for h in range(8) for j in range(1,8)]).reshape(8,7,8)
        residual=nmul(v,constants).sum(axis=1)
        recovered=cof[:,1:].T@residual/np.linalg.det(w)
        np.testing.assert_allclose(recovered,constants,atol=1e-14)

    def test_04_03_08_trace_formula_unit_cm_numerical_diagnostic(self):
        # V={0}, so the trace over V minus {0} is EXACTLY zero.
        # Cubic signed coordinate symmetry determines all eta_h from eta_0.
        # This uses finite sums, not certified infinite-series values.
        observations=[]
        l,m=np.array(e(1)),np.array(e(2))
        rows=np.array([mul(e(1),mul(e(h),e(2))) for h in range(8)])
        for cutoff in (1,2):
            points=lattice(cutoff,np.ones(8))
            eta0=2*zeta(np.array(e(0))*.5,points,-1)
            eta=np.array([eta0[0]*np.array(conj(e(h))) for h in range(8)])
            # W=I, det=1, cofactors=I, N(lambda)N(mu)=1; i=4.
            prediction=-(nmul(nmul(m,rows[4]@eta),l)-eta[4])
            self.assertGreater(np.linalg.norm(prediction),1)
            observations.append({'cutoff':cutoff,'eta0':eta0.tolist(),
                                 'trace_lhs_exact':[0]*8,'trace_rhs_truncated_eta':prediction.tolist()})
        OBSERVATIONS['theorem_4_14_unit_cm_diagnostic']=observations


class GPUVerification(unittest.TestCase):
    def setUp(self):
        if CK.cp is None:
            raise unittest.SkipTest('CuPy unavailable; multiply() was not executed')
        if not CK.loaded():
            raise unittest.SkipTest('cuda_kernel.so unavailable; build with make kernel')
        try:
            count=CK.cp.cuda.runtime.getDeviceCount()
        except CK.cp.cuda.runtime.CUDARuntimeError as exc:
            raise unittest.SkipTest('CUDA device unavailable: '+str(exc))
        if not count:
            raise unittest.SkipTest('No CUDA device')

    def test_02_multiply_basis_and_dense_independent_oracle(self):
        rng=np.random.default_rng(191201351)
        for d in (2,4,8,16,32):
            pairs=list(product(range(d),repeat=2))
            a=np.array([e(i,d) for i,j in pairs]+rng.integers(-2,3,size=(16,d)).tolist(),dtype=np.float64)
            b=np.array([e(j,d) for i,j in pairs]+rng.integers(-2,3,size=(16,d)).tolist(),dtype=np.float64)
            expected=np.array([recursive(x.tolist(),y.tolist()) for x,y in zip(a,b)])
            np.testing.assert_array_equal(replay_product(a,b),expected)

    def test_04_03_multiply_proposition_4_9_counterexample(self):
        l,m,z=(np.array(e(j),dtype=np.float64) for j in (1,2,4))
        arg=replay_product(l,replay_product(z,m))
        lhs=replay_product(replay_product(m,nq(arg)),l)
        np.testing.assert_array_equal(lhs,z)
        self.assertFalse(np.array_equal(lhs,nq(z)))


# Integrate with repository GPU guards without requiring pytest for unittest.
try:
    import pytest
except ImportError:
    pass
else:
    PaperVerification=pytest.mark.cpu(PaperVerification)
    GPUVerification=pytest.mark.cpu(GPUVerification)
