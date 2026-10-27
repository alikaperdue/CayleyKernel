"""Section-ordered finite tests. Counterexample passes reproduce discrepancies."""
from fractions import Fraction as F
from itertools import product
import math
import random
import unittest

import numpy as np
import CayleyKernel as CK
from tests.papers.type_replay import replay_product
from .algebra import (e, add, scale, conj, norm2, mul, reference, recursive_sign,
                      power, quadratic_power, bracketed_powers, polynomial,
                      polymul, remainder, minimal_polynomial, radius_from_roots,
                      lp, unit_power_samples, bumped, paper_sqrt, stable_sqrt,
                      Angle, complex_argument, quaternion_argument, angle_vector)

OBSERVATIONS = {}
DIMS = (1, 2, 4, 8, 16, 32, 64)
SEED = 150501670


def vectors(dim, count=8):
    rng = random.Random(SEED+dim)
    return [[rng.randint(-2, 2) for _ in range(dim)] for _ in range(count)]


def paper_zero_divisors(dim=16):
    # Paper e_j is the vector with ONE-BASED coordinate j, no permutation.
    return (add(e(3, dim), e(10, dim)),
            add(e(6, dim), scale(e(15, dim), -1)))


class PaperVerification(unittest.TestCase):
    def test_01_01_doubling_twist_signed_basis_and_embeddings(self):
        for dim in DIMS:
            for i, j in product(range(dim), repeat=2):
                s = recursive_sign(i, j, dim)
                self.assertEqual(CK.twist(i, j), s)
                self.assertEqual(mul(e(i, dim), e(j, dim)), scale(e(i ^ j, dim), s))
                for u, v in product((-1, 1), repeat=2):
                    self.assertEqual(CK.basis(u*(i+1), v*(j+1)), u*v*s*((i ^ j)+1))
            if dim < 64:
                for a, b in zip(vectors(dim), reversed(vectors(dim))):
                    self.assertEqual(mul(a+[0]*dim, b+[0]*dim), mul(a,b)+[0]*dim)

    def test_01_02_distributivity_unit_and_conjugation(self):
        for dim in DIMS:
            samples = vectors(dim)
            for a, b, c in zip(samples, samples[1:], samples[2:]):
                self.assertEqual(mul(a,b), reference(a,b))
                self.assertEqual(mul(a,add(b,c)), add(mul(a,b),mul(a,c)))
                self.assertEqual(mul(add(a,b),c), add(mul(a,c),mul(b,c)))
                self.assertEqual(mul(e(0,dim),a),a)
                self.assertEqual(mul(a,e(0,dim)),a)
                self.assertEqual(conj(conj(a)),a)
                self.assertEqual(conj(mul(a,b)),mul(conj(b),conj(a)))

    def test_01_03_commutativity_associativity_and_alternativity_boundary(self):
        for dim in (1,2):
            for a,b in zip(vectors(dim), reversed(vectors(dim))):
                self.assertEqual(mul(a,b),mul(b,a))
        self.assertNotEqual(mul(e(1,4),e(2,4)),mul(e(2,4),e(1,4)))
        for i,j,k in product(range(4),repeat=3):
            self.assertEqual(mul(mul(e(i,4),e(j,4)),e(k,4)),mul(e(i,4),mul(e(j,4),e(k,4))))
        self.assertNotEqual(mul(mul(e(1,8),e(2,8)),e(4,8)),mul(e(1,8),mul(e(2,8),e(4,8))))
        for a,b in zip(vectors(8), reversed(vectors(8))):
            self.assertEqual(mul(mul(a,a),b),mul(a,mul(a,b)))
            self.assertEqual(mul(b,mul(a,a)),mul(mul(b,a),a))

    def test_01_04_lemma_1_1_quadratic_trace_and_norm(self):
        for dim in DIMS:
            for a in vectors(dim)+[[0]*dim,e(0,dim),scale(e(0,dim),-3)]:
                self.assertEqual(mul(conj(a),a),scale(e(0,dim),norm2(a)))
                self.assertEqual(mul(a,conj(a)),scale(e(0,dim),norm2(a)))
                self.assertEqual(add(a,conj(a)),scale(e(0,dim),2*a[0]))
                self.assertEqual(polynomial(a,[norm2(a),-2*a[0],1]),[0]*dim)

    def test_01_05_theorem_1_1_all_bracketings_and_recurrence(self):
        counts = {}
        for dim in (2,4,8,16,32):
            for a in vectors(dim,2):
                for k in range(1,8):
                    expected = quadratic_power(a,k)
                    all_values = bracketed_powers(a,k)
                    self.assertTrue(all(list(x)==expected for x in all_values))
                    counts[str(k)] = len(all_values)
                self.assertEqual(mul(mul(a,a),a),mul(a,mul(a,a)))
                self.assertEqual(mul(mul(a,a),mul(a,a)),power(a,4))
        OBSERVATIONS['binary_parenthesizations_per_exponent'] = counts

    def test_01_06_printed_sedenion_nonalternative_example(self):
        a,b = paper_zero_divisors()
        for fn in (mul,reference):
            self.assertEqual(fn(a,b),[0]*16)
            self.assertEqual(fn(a,fn(a,b)),[0]*16)
            self.assertEqual(fn(fn(a,a),b),scale(b,-2))
        OBSERVATIONS['printed_zero_divisors_kernel_indices'] = {'o1':[3,10],'o2':[[6,1],[15,-1]]}

    def test_02_01_theorem_2_1_minimal_polynomial_and_divisibility(self):
        for dim in DIMS:
            for a in vectors(dim)+[[0]*dim,scale(e(0,dim),-2)]:
                q = minimal_polynomial(a)
                self.assertEqual(polynomial(a,q),[0]*dim)
                if any(a[1:]):
                    self.assertEqual(len(q),3)  # nonzero imaginary coordinate rules out a linear annihilator
                else:
                    self.assertEqual(len(q),2)
                p = [norm2(a),-2*a[0],1]
                self.assertEqual(remainder(p,q),[])
                for factor in ([1,2,1],[-3,0,2,1]):
                    annihilator = polymul(q,factor)
                    self.assertEqual(polynomial(a,annihilator),[0]*dim)
                    self.assertEqual(remainder(annihilator,q),[])

    def test_02_02_theorems_2_2_2_3_radius_from_minimal_roots(self):
        max_error = 0
        for dim in DIMS:
            for a in vectors(dim)+[[0]*dim,scale(e(0,dim),-3)]:
                radius = radius_from_roots(a)
                expected = math.sqrt(norm2(a))
                max_error = max(max_error,abs(radius-expected))
                self.assertAlmostEqual(radius,expected,places=11)
                for scalar in (F(-3,2),F(0),F(2,3)):
                    self.assertAlmostEqual(radius_from_roots(scale(a,scalar)),float(abs(scalar))*radius,places=10)
        OBSERVATIONS['radius_root_max_absolute_error'] = max_error

    def test_02_03_power_norm_no_nilpotents_and_continuity_samples(self):
        for dim in DIMS:
            for a in vectors(dim):
                for k in range(1,13):
                    ak = power(a,k)
                    self.assertEqual(ak,quadratic_power(a,k))
                    self.assertEqual(norm2(ak),norm2(a)**k)
                    self.assertEqual(any(ak),any(a))
                delta = scale(e(0,dim),F(1,1000))
                difference = abs(radius_from_roots(add(a,delta))-radius_from_roots(a))
                self.assertLessEqual(difference, .001+1e-11)

    def test_02_04_radius_is_not_left_multiplication_spectral_radius(self):
        a,_ = paper_zero_divisors()
        left = np.array([mul(a,e(j,16)) for j in range(16)],dtype=float).T
        spectral = float(max(abs(np.linalg.eigvals(left))))
        radius = radius_from_roots(a)
        self.assertEqual(np.linalg.matrix_rank(left),12)
        self.assertAlmostEqual(spectral,2,places=12)
        self.assertAlmostEqual(radius,math.sqrt(2),places=12)
        self.assertNotAlmostEqual(spectral,radius)
        OBSERVATIONS['element_vs_left_operator_radius'] = {'element':radius,'left_operator':spectral,'rank':12}

    def test_03_01_lp_subnorm_definitions_and_nontriangle_example(self):
        # On A0 every lp is just absolute value, including p>2.
        for p in (.5,1,2,4,math.inf):
            self.assertEqual(lp([-2],p),2)
        for dim in (2,8,32):
            for a in vectors(dim):
                for p in (.5,1,1.5,2,4,math.inf):
                    self.assertGreater(lp(a,p),0)
                    self.assertAlmostEqual(lp(scale(a,F(-3,2)),p),1.5*lp(a,p),places=10)
            self.assertGreater(lp(add(e(0,dim),e(1,dim)),.5),lp(e(0,dim),.5)+lp(e(1,dim),.5))

    def test_03_02_gelfand_lp_and_continuous_angular_subnorm(self):
        exponents = (1,2,4,8,16,32,64,128,256)
        observations = []
        raw_drift = []
        for dim in (2,8,32):
            a = vectors(dim,1)[0]
            radius, samples = unit_power_samples(a,exponents)
            raw = e(0,dim)
            unit = scale(a,1/radius)
            for k in range(1,257):
                raw = mul(raw,unit)
                if k in (64,128,256):
                    self.assertTrue(all(math.isfinite(x) for x in raw))
                    raw_drift.append({'dimension':dim,'k':k,
                                      'squared_norm_error':abs(float(norm2(raw))-1),
                                      'max_coordinate_error':max(abs(x-y) for x,y in zip(raw,samples[k]))})
            for p in (.5,1,1.5,2,4,math.inf):
                ratios = []
                exponent = (0 if p==math.inf else 1/p)-.5
                for k,x in samples.items():
                    self.assertAlmostEqual(norm2(x),1,places=10)
                    root_ratio = math.exp(math.log(lp(x,p))/k)
                    low,high = sorted((1,dim**(exponent/k)))
                    self.assertGreaterEqual(root_ratio,low-1e-11)
                    self.assertLessEqual(root_ratio,high+1e-11)
                    ratios.append(root_ratio)
                observations.append({'dimension':dim,'p':str(p),'radius':radius,
                                     'exponents':list(exponents),'root_over_radius':ratios,
                                     'last_universal_relative_bound':math.expm1(abs(exponent)*math.log(dim)/256)})
            for k,x in samples.items():
                angular = math.sqrt(norm2(x))*(1+.5*x[0]**2/norm2(x))
                self.assertLessEqual(angular**(1/k),1.5**(1/k)+1e-11)
                self.assertGreaterEqual(angular**(1/k),1-1e-11)
        OBSERVATIONS['gelfand_experiments'] = observations
        OBSERVATIONS['raw_float_unit_power_drift'] = raw_drift

    def test_03_03_discontinuous_equivalent_bump_and_gelfand(self):
        kappa = 3
        a0 = e(0,4)
        values = []
        for m in (10,100,1000):
            x = add(a0,scale(e(1,4),F(1,m)))
            values.append(bumped(x,kappa))
            self.assertLess(bumped(x,kappa),1.01)
        self.assertEqual(bumped(a0,kappa),3)
        for a in vectors(4):
            r = math.sqrt(norm2(a))
            self.assertLessEqual(r,bumped(a,kappa))
            self.assertLessEqual(bumped(a,kappa),kappa*r)
            self.assertAlmostEqual(bumped(scale(a,-2),kappa),2*bumped(a,kappa))
        for k in (1,2,4,8,16,32,64,128):
            root = bumped(power(e(1,4),k),kappa)**(1/k)
            self.assertGreaterEqual(root,1)
            self.assertLessEqual(root,kappa**(1/k))
        OBSERVATIONS['bump_discontinuity'] = {'at_real_unit':3,'nearby_off_line_values':values}

    def test_03_04_additive_periodic_restriction_and_complex_argument(self):
        angles = [Angle(F(i,3),F(j,5)) for i,j in product(range(-2,3),repeat=2)]
        for a,b in zip(angles,reversed(angles)):
            self.assertEqual((a+b).phi(),a.phi()+b.phi())
            self.assertEqual(a.times(F(-7,3)).phi(),F(-7,3)*a.phi())
            self.assertEqual((a+Angle(F(7,2),F(0))).phi(),a.phi())
            self.assertEqual(complex_argument(a).phi(),a.phi())
        # Different prescribed values at pi and 1 exclude a continuous linear extension.
        self.assertEqual(Angle(F(1),F(0)).phi(),0)
        self.assertEqual(Angle(F(0),F(1)).phi(),1)

    def test_03_05_theorem_3_2_complex_log_gelfand_and_real_homogeneity(self):
        observations = []
        for angle in (Angle(F(0),F(1)),Angle(F(1),F(-1)),Angle(F(1,3),F(0))):
            z = angle_vector(angle)
            logs = []
            for k in (1,2,3,4,8,16,32,64):
                exact_angle = complex_argument(angle.times(k))
                actual = power(z,k)
                np.testing.assert_allclose(actual,angle_vector(exact_angle),atol=2e-13)
                self.assertEqual(exact_angle.phi(),k*angle.phi())
                for p in (.5,1,4):
                    log_root = (math.log(lp(actual,p))+float(exact_angle.phi()))/k
                    bound = abs(1/p-.5)*math.log(2)/k
                    self.assertLessEqual(abs(log_root-float(angle.phi())),bound+1e-12)
                logs.append(float(exact_angle.phi()/k))
            self.assertEqual(complex_argument(angle+Angle(F(1),F(0))).phi(),angle.phi())
            observations.append({'angle_pi':str(angle.pi),'angle_one':str(angle.one),
                                 'log_root_over_radius':logs})
        OBSERVATIONS['complex_pathological_restriction'] = observations

    def test_03_06_theorem_3_3_quaternion_negative_homogeneity_counterexample(self):
        theta = Angle(F(0),F(1))
        q = angle_vector(theta,4)
        negative_angle = quaternion_argument(theta+Angle(F(1),F(0)))
        self.assertEqual(quaternion_argument(theta).phi(),1)
        self.assertEqual(negative_angle.phi(),-1)
        self.assertAlmostEqual(math.atan2(np.linalg.norm(q[1:]),-q[0]),negative_angle.value())
        self.assertNotEqual(negative_angle.phi(),quaternion_argument(theta).phi())
        OBSERVATIONS['quaternion_homogeneity_counterexample_log_h'] = {'q':1,'minus_q':-1}

    def test_03_07_theorem_3_3_folded_power_root_counterexample(self):
        theta = Angle(F(0),F(1))
        q = angle_vector(theta,4)
        observations = []
        for k in (1,2,3,4,5,6,7,8,16,32,64):
            arg = quaternion_argument(theta.times(k))
            actual = power(q,k)
            observed_arg = math.atan2(np.linalg.norm(actual[1:]),actual[0])
            self.assertAlmostEqual(observed_arg,arg.value(),places=12)
            observations.append({'k':k,'phi_arg_qk':str(arg.phi()),'log_root':float(arg.phi()/k)})
        self.assertEqual(quaternion_argument(theta.times(4)).phi(),-4)
        self.assertEqual(quaternion_argument(theta.times(7)).phi(),7)
        OBSERVATIONS['quaternion_folded_power_roots'] = observations

    def test_03_08_dense_angle_families_and_large_log_values(self):
        rows = []
        for target in (.2,1.,2.,5.):
            for denominator in (10,100,1000):
                for prescribed_phi in (0,1):
                    numerator = round(denominator*(target-prescribed_phi)/math.pi)
                    angle = Angle(F(numerator,denominator),F(prescribed_phi))
                    self.assertEqual(angle.phi(),prescribed_phi)
                    self.assertLessEqual(abs(angle.value()-target),math.pi/(2*denominator)+1e-12)
            # Prescribed arbitrarily large log values at angles near the same target.
            for prescribed_phi in (-1000,1000):
                numerator = round(10000*(target-prescribed_phi)/math.pi)
                angle = Angle(F(numerator,10000),F(prescribed_phi))
                self.assertEqual(angle.phi(),prescribed_phi)
                self.assertLess(abs(angle.value()-target),.00016)
                rows.append({'target':target,'angle_error':abs(angle.value()-target),
                             'phi':prescribed_phi})
        OBSERVATIONS['dense_angle_and_unbounded_log_samples'] = rows

    def test_04_01_lp_stability_bounds_and_p_gt_2_instability(self):
        observations = []
        for dim in (2,8,32):
            radius,samples = unit_power_samples(vectors(dim,1)[0],(1,2,4,8,16,32))
            original = scale(vectors(dim,1)[0],1/radius)
            for p in (.5,1,1.5,2):
                sigma = dim**(1/p-.5)
                for k,x in samples.items():
                    self.assertLessEqual(lp(x,p),sigma*lp(original,p)**k+1e-9)
        a = [1,1]
        for p in (3,4,math.inf):
            logs = []
            for k in (4,8,16,32,64):
                ratio_log = math.log(lp(power(a,k),p))-k*math.log(lp(a,p))
                prediction = k*(.5-(0 if p==math.inf else 1/p))*math.log(2)
                self.assertAlmostEqual(ratio_log,prediction,places=11)
                logs.append(ratio_log)
            observations.append({'p':str(p),'k':[4,8,16,32,64],'log_stability_ratio':logs})
        OBSERVATIONS['lp_instability'] = observations

    def test_04_02_weighted_sup_stability_criterion(self):
        grid = list(product(range(-3,4),repeat=2))
        cases = ((F(1),F(1)),(F(2),F(2)),(F(5,3),F(5,4)),(F(3),F(1)))
        for u,v in cases:
            criterion = u*u*v*v >= u*u+v*v
            witness = [v,u]
            weighted = max(u*abs(witness[0]),v*abs(witness[1]))
            self.assertEqual(weighted**2>=norm2(witness),criterion)
            if criterion:
                for a in grid:
                    self.assertGreaterEqual(max(u*abs(a[0]),v*abs(a[1]))**2,norm2(a))
                # Uniform sigma=max(u,v) works, by Euclidean power norm identity.
                for a in ([1,1],[2,-1],[0,1]):
                    for k in range(1,13):
                        ak = power(a,k)
                        f = lambda x:max(u*abs(x[0]),v*abs(x[1]))
                        self.assertLessEqual(f(ak)**2,max(u,v)**2*f(a)**(2*k))

    def test_04_03_weighted_l1_stability_criterion(self):
        for u,v in ((F(1),F(1)),(F(2),F(1)),(F(1,2),F(2)),(F(2),F(1,2))):
            f = lambda a:u*abs(a[0])+v*abs(a[1])
            criterion = u>=1 and v>=1
            basis_pass = all(f(e(j,2))>=1 for j in (0,1))
            self.assertEqual(criterion,basis_pass)
            if criterion:
                for a in product(range(-3,4),repeat=2):
                    self.assertGreaterEqual(f(a)**2,norm2(a))
                for a in ([1,1],[2,-1],[0,1]):
                    for k in range(1,13):
                        self.assertLessEqual(f(power(a,k))**2,max(u,v)**2*f(a)**(2*k))

    def test_04_04_equivalent_discontinuous_bump_is_stable(self):
        for dim in (2,8,16):
            for a in vectors(dim)+[e(1,dim),e(0,dim)]:
                for k in range(1,17):
                    self.assertLessEqual(bumped(power(a,k)),3*bumped(a)**k*(1+1e-12))
                self.assertGreaterEqual(bumped(a),math.sqrt(norm2(a)))

    def test_04_05_corollary_4_2_missing_continuity_complex_witness(self):
        # Real-complex submodulus restriction: log g(q)=-1 < log radius=0.
        # For any Hamel extension with these prescribed values, g is globally
        # strongly stable by additivity and oriented complex arguments.
        theta = Angle(F(1),F(-1))
        self.assertEqual(complex_argument(theta).phi(),-1)
        for k in range(1,33):
            self.assertEqual(complex_argument(theta.times(k)).phi(),-k)
        OBSERVATIONS['corollary_4_2_missing_continuity'] = {'log_g_unit':-1,'log_radius':0,
                                                       'scope':'restriction with Hamel extension argument'}

    def test_04_06_theorem_4_3_quaternion_stability_counterexample_family(self):
        rows = []
        for m in (1,2,10,100):
            integer = math.floor(2*m*math.pi)-4
            theta = Angle(F(m),F(-integer,2))
            self.assertTrue(2 <= theta.value() < 2.5)
            once = quaternion_argument(theta)
            twice = quaternion_argument(theta.times(2))
            log_ratio = twice.phi()-2*once.phi()
            self.assertEqual(log_ratio,2*integer)
            q = angle_vector(theta,4)
            actual = power(q,2)
            observed = math.atan2(np.linalg.norm(actual[1:]),actual[0])
            self.assertAlmostEqual(observed,twice.value(),places=11)
            rows.append({'m':m,'M':integer,'theta':theta.value(),'k':2,'log_h_q2_over_h_q_squared':int(log_ratio)})
        OBSERVATIONS['quaternion_unbounded_stability_family'] = rows

    def test_04_07_strong_weighted_l1_criterion_and_boundary_typo(self):
        grid = list(product(range(-2,3),repeat=2))
        for u,v in ((1,1),(4,2),(2,2),(2,1),(F(1,2),2),(2,F(1,2))):
            f = lambda a:u*abs(a[0])+v*abs(a[1])
            strong = v*v>=u>=1
            if strong:
                for a,b in product(grid,repeat=2):
                    self.assertLessEqual(f(mul(a,b)),f(a)*f(b))
            elif u < 1:
                self.assertGreater(f(power(e(0,2),2)),f(e(0,2))**2)
            else:
                self.assertGreater(f(power(e(1,2),2)),f(e(1,2))**2)
        # Printed 'stable but not strongly' uses >= at a boundary where both hold.
        self.assertTrue(4>=2**2>=1)
        self.assertEqual(2**2,4)
        f = lambda a:4*abs(a[0])+2*abs(a[1])
        self.assertEqual(f(power(e(1,2),2)),f(e(1,2))**2)
        # A genuinely stable/non-strong example; sigma=4 is enough, sigma=1 fails.
        g = lambda a:4*abs(a[0])+abs(a[1])
        self.assertEqual(g(power(e(1,2),2)),4)
        self.assertEqual(g(e(1,2))**2,1)
        OBSERVATIONS['weighted_l1_boundary'] = {'u':4,'v':2,'strong':True,'printed_not_strong':True,
                                              'correct_strict_condition':'u > v^2 >= 1'}

    def test_05_01_power_equation_constants_radius_and_unique_scaled_candidate(self):
        for dim in DIMS:
            for a in vectors(dim)+[[0]*dim]:
                for k in range(1,9):
                    self.assertEqual(norm2(power(a,k)),norm2(a)**k)
                    for constant in (0,1):
                        self.assertEqual(constant,constant**k)
            # c*radius has power identity at 1, k=2 iff c=c²; positivity leaves c=1.
            for c in (F(1,2),F(1),F(2)):
                self.assertEqual(c==c*c,c==1)
        self.assertNotEqual(F(-1),F(-1)**2)

    def test_05_02_nilpotent_obstruction_matrix_reference_only(self):
        # Matrix algebra is a reference example OUTSIDE CayleyKernel's algebra.
        n = np.array([[0,1],[0,0]],dtype=int)
        np.testing.assert_array_equal(n@n,np.zeros((2,2),dtype=int))
        self.assertGreater(np.linalg.norm(n),0)
        self.assertEqual(max(abs(np.linalg.eigvals(n))),0)
        # A subnorm must give f(n)>0 and f(0)=0, contradicting f(n²)=f(n)².
        for positive_value in (F(1,10),F(1),F(3)):
            self.assertGreater(positive_value**2,0)

    def test_05_03_theorem_5_2_complex_power_and_quaternion_failure(self):
        theta = Angle(F(0),F(1))
        for k in range(1,65):
            self.assertEqual(complex_argument(theta.times(k)).phi(),k*theta.phi())
        self.assertNotEqual(quaternion_argument(theta.times(4)).phi(),4*theta.phi())
        OBSERVATIONS['theorem_5_2b_log_power_defect_k4'] = -8

    def test_05_04_lemma_5_1_exact_square_roots_generated_targets(self):
        for dim in DIMS[1:]:
            for b in vectors(dim):
                b[0] = abs(b[0])+1
                a = mul(b,b)
                radius = norm2(b)
                denominator = 2*b[0]
                recovered = scale(add(a,scale(e(0,dim),radius)),F(1,denominator))
                self.assertEqual(recovered,b)
                self.assertEqual(mul(recovered,recovered),a)
                self.assertEqual(reference(recovered,recovered),a)
            for scalar in (-9,-4,0,4,9):
                root = paper_sqrt(scale(e(0,dim),scalar))
                np.testing.assert_allclose(mul(root,root),scale(e(0,dim),scalar),atol=1e-14)

    def test_05_05_lemma_5_1_numerical_square_roots_and_cancellation(self):
        largest = 0
        for dim in DIMS[1:]:
            for a in vectors(dim):
                root = paper_sqrt(a)
                error = float(np.max(np.abs(np.array(mul(root,root))-a)))
                largest = max(largest,error)
                np.testing.assert_allclose(mul(root,root),a,rtol=2e-13,atol=2e-13)
                np.testing.assert_allclose(stable_sqrt(a),root,rtol=2e-13,atol=2e-13)
        near = [-1.,1e-12]
        with self.assertRaises(ZeroDivisionError):
            paper_sqrt(near)
        root = stable_sqrt(near)
        np.testing.assert_allclose(mul(root,root),near,rtol=2e-15,atol=1e-30)
        OBSERVATIONS['sqrt_experiments'] = {'ordinary_max_absolute_residual':largest,
                                           'near_negative_real':near,'stable_root':root,
                                           'literal_formula_float64':'division by zero through cancellation'}

    def test_05_06_nonnegativity_square_argument_and_real_exception(self):
        for dim in (2,8,16,32):
            for a in vectors(dim):
                b = stable_sqrt(a)
                self.assertGreaterEqual(math.sqrt(norm2(a)),0)
                self.assertAlmostEqual(math.sqrt(norm2(mul(b,b))),norm2(b),places=10)
        # Identity on A0 is a signed power solution; it takes a negative value.
        for a in (-3,-1,0,2):
            for k in range(1,9):
                self.assertEqual(power([a],k),[a**k])
        self.assertLess(-1,0)
        # Real-part projection does not extend that solution to C.
        self.assertNotEqual(power(e(1,2),2)[0],e(1,2)[0]**2)

    def test_05_07_closure_operations_on_power_solutions(self):
        functions = (
            lambda t:t,
            lambda t:t*t,
            lambda t:max(t,t*t),
            lambda t:min(t,t*t),
            lambda t:t**2*t**3,
            lambda t:F(int(t!=0)),
        )
        for dim in (2,8,16,32):
            for a in vectors(dim,3)+[scale(e(0,dim),F(1,2)),[0]*dim]:
                for k in range(1,9):
                    t = norm2(a)
                    tk = norm2(power(a,k))
                    for f in functions:
                        self.assertEqual(f(tk),f(t)**k)

    def test_05_08_composition_boundary_and_zero_divisor_embedding(self):
        for dim in (1,2,4,8):
            for a,b in zip(vectors(dim),reversed(vectors(dim))):
                self.assertEqual(norm2(mul(a,b)),norm2(a)*norm2(b))
        for dim in (16,32,64):
            a,b = paper_zero_divisors(dim)
            self.assertEqual(mul(a,b),[0]*dim)
            self.assertGreater(norm2(a)*norm2(b),0)
            # Nevertheless, each individual power obeys the norm power identity.
            for k in range(1,9):
                self.assertEqual(norm2(power(a,k)),norm2(a)**k)


class GPUVerification(unittest.TestCase):
    def setUp(self):
        if CK.cp is None:
            self.skipTest('CuPy unavailable; multiply() not executed')
        if not CK.loaded():
            self.skipTest('cuda_kernel.so unavailable; build with make kernel')
        try:
            devices = CK.cp.cuda.runtime.getDeviceCount()
        except CK.cp.cuda.runtime.CUDARuntimeError as exc:
            self.skipTest('CUDA device unavailable: '+str(exc))
        if not devices:
            self.skipTest('No CUDA device')

    def test_01_01_multiply_basis_and_dense_doubling_oracle(self):
        for dim in DIMS:
            pairs = list(product(range(dim),repeat=2))
            a = np.array([e(i,dim) for i,j in pairs]+vectors(dim),dtype=np.float64)
            b = np.array([e(j,dim) for i,j in pairs]+list(reversed(vectors(dim))),dtype=np.float64)
            expected = np.array([reference(x.tolist(),y.tolist()) for x,y in zip(a,b)])
            np.testing.assert_array_equal(replay_product(a,b),expected)

    def test_01_02_05_multiply_powers_norms_and_printed_zero_divisors(self):
        for dim in DIMS:
            a = np.array(vectors(dim,3),dtype=np.float64)
            current = np.tile(e(0,dim),(3,1)).astype(float)
            for k in range(1,9):
                current = replay_product(current,a)
                expected = np.array([quadratic_power(row.tolist(),k) for row in a])
                np.testing.assert_array_equal(current,expected)
                np.testing.assert_allclose(np.sum(current*current,axis=1),np.sum(a*a,axis=1)**k,rtol=2e-13,atol=0)
        a,b = (np.array(x,dtype=np.float64) for x in paper_zero_divisors())
        np.testing.assert_array_equal(replay_product(a,b),np.zeros(16))
        np.testing.assert_array_equal(replay_product(replay_product(a,a),b),-2*b)

    def test_05_multiply_square_roots(self):
        for dim in DIMS[1:]:
            a = np.array(vectors(dim),dtype=np.float64)
            roots = np.array([stable_sqrt(row.tolist()) for row in a])
            np.testing.assert_allclose(replay_product(roots,roots),a,rtol=2e-13,atol=2e-13)


try:
    import pytest
except ImportError:
    pass
else:
    PaperVerification = pytest.mark.cpu(PaperVerification)
    GPUVerification = pytest.mark.cpu(GPUVerification)
