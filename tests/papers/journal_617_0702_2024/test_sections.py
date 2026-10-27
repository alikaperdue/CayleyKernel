"""Tests in paper order; CPU verification is separate from optional hardware tests."""
from collections import Counter
from itertools import combinations,product
import random
import unittest
import numpy as np
import CayleyKernel as CK
from .finite import (e,add,scale,conj,mul,batch_product,basis_table,recursive_sign,parameters,
                     norm_weights,norm,trace,chunks,enumerate_counts,quadratic_histogram,
                     zorn_mul,zorn_norm,split_to_zorn,zorn_to_split,norm_minus_one_quaternion,
                     main_to_split,split_to_main,annihilator,paper_id,paper_nil,paper_zd,
                     paper_units,correct_norm_zero,correct_intersection)

SEED=6172024
OBSERVATIONS={}
CERTIFICATES={}
PRIMES=(3,5,7,11,13,17,19,23,29,31,37)


def random_vectors(p,d,count=8):
    rng=random.Random(SEED+p+d)
    return [tuple(rng.randrange(p) for _ in range(d)) for _ in range(count)]


def polynomial_counts(p,d,split=False):
    weights=norm_weights(d,split)
    imag=quadratic_histogram(weights[1:],p)
    total=quadratic_histogram(weights,p)
    return {'p':p,'dimension':d,'split':split,'idempotents':imag[(-pow(2,-1,p)**2)%p]+2,
            'nilpotents_including_zero':imag[0],'norm_zero_including_zero':total[0]}


class PaperVerification(unittest.TestCase):
    def test_01_01_intro_complex_plus_sign_and_split_quaternion_relations(self):
        for p in (3,5,7):
            self.assertEqual(mul(e(1,2),e(1,2),p),(p-1,0))
            # Printed (ac+bd,ad+bc) instead makes the generator square +1.
            self.assertNotEqual((1,0),mul(e(1,2),e(1,2),p))
            # In an associative anticommuting pair i^2=+1,j^2=-1,
            # (ij)^2=-i^2*j^2=+1, contradicting the printed k^2=-1.
            self.assertNotEqual(1%p,-1%p)
        OBSERVATIONS['intro_sign_qualifications']={
            'complex_formula':'Printed plus sign gives split-complex multiplication, not standard complex multiplication.',
            'split_quaternion':'Printed i^2=1,j^2=k^2=-1,ij=k is incompatible with associative anticommuting quaternion generators in odd characteristic.'}

    def test_01_02_basis_twist_and_parameterized_doubling(self):
        checked=0
        for d in (2,4,8,16,32,64):
            for i,j in product(range(d),repeat=2):
                expected=recursive_sign(i,j,parameters(d))
                self.assertEqual(CK.twist(i,j),expected)
                for a,b in product((-1,1),repeat=2):
                    self.assertEqual(CK.basis(a*(i+1),b*(j+1)),a*b*expected*((i^j)+1))
                checked+=1
        OBSERVATIONS['basis_pairs_against_recursive_doubling']=checked

    def test_01_03_printed_octonion_table_and_split_table_norm_conflict(self):
        # Complete Table1, visually transcribed from enlarged page2. Entries
        # use signed one-based labels, so -1 represents scalar minus one.
        printed=((1,2,3,4,5,6,7,8),(2,-1,4,-3,6,-5,-8,7),
                 (3,-4,-1,2,7,8,-5,-6),(4,3,-2,-1,8,-7,6,-5),
                 (5,-6,-7,-8,-1,2,3,4),(6,5,-8,7,-2,-1,-4,3),
                 (7,8,5,-6,-3,4,-1,-2),(8,-7,6,5,-4,-3,2,-1))
        self.assertEqual(printed,basis_table(8))
        for p in (3,5,7):
            # Selected Table2 cells: e1*e2=e4, e1^2=e2^2=-1, e4^2=+1.
            # Their associated quadratic norms violate composition.
            self.assertNotEqual((-1)%p,1%p) # N(e4) != N(e1)*N(e2)
        CERTIFICATES['literal_table_cells']={
            'table1_full_signed_one_based_labels':printed,
            'table1_all_64_cells_agree_with_CayleyKernel':True,
            'table2':[{'row':1,'column':2,'product':{'4':1}},
                      {'row':1,'column':1,'product':{'0':-1}},
                      {'row':2,'column':2,'product':{'0':-1}},
                      {'row':4,'column':4,'product':{'0':1}}],
            'scope':'Complete Table1 and selected unambiguous Table2 cells; not a full Table2 transcription.'}

    def test_01_04_zorn_product_determinant_and_split_isomorphism(self):
        records=[]
        for p in (3,5,7,11):
            for i,j in product(range(8),repeat=2):
                a,b=e(i,8),e(j,8)
                self.assertEqual(split_to_zorn(mul(a,b,p,True),p),zorn_mul(split_to_zorn(a,p),split_to_zorn(b,p),p))
            for x in random_vectors(p,8):
                self.assertEqual(zorn_to_split(split_to_zorn(x,p),p),x)
                self.assertEqual(norm(x,p,True),zorn_norm(split_to_zorn(x,p),p))
            records.append({'p':p,'split_basis_images_in_zorn':[split_to_zorn(e(i,8),p) for i in range(8)]})
        CERTIFICATES['split_to_zorn_maps']=records

    def test_01_05_standard_and_split_octonions_are_isomorphic_over_odd_fields(self):
        records=[]
        for p in PRIMES:
            u=norm_minus_one_quaternion(p)
            self.assertEqual(norm(u,p),p-1)
            for i,j in product(range(8),repeat=2):
                a,b=e(i,8),e(j,8)
                self.assertEqual(main_to_split(mul(a,b,p),p),mul(main_to_split(a,p),main_to_split(b,p),p,True))
            for x in random_vectors(p,8):
                self.assertEqual(split_to_main(main_to_split(x,p),p),x)
                self.assertEqual(norm(main_to_split(x,p),p,True),norm(x,p))
            records.append({'p':p,'quaternion_u_norm_minus_one':u,
                            'main_basis_images_in_split':[main_to_split(e(i,8),p) for i in range(8)]})
        CERTIFICATES['main_to_split_maps']=records

    def test_01_06_norm_composition_coefficients_for_composition_dimensions(self):
        cases=[]
        for p,d,split in product((3,5),(4,8),(False,True)):
            weights=norm_weights(d,split)
            matrices=[[mul(e(i,d),e(k,d),p,split) for k in range(d)] for i in range(d)]
            def pairing(a,b):return sum(w*x*y for w,x,y in zip(weights,a,b))%p
            for i,j,k,l in product(range(d),repeat=4):
                actual=(pairing(matrices[i][k],matrices[j][l])+pairing(matrices[j][k],matrices[i][l]))%p
                expected=2*weights[i]*weights[k]*int(i==j)*int(k==l)%p
                self.assertEqual(actual,expected)
            cases.append({'p':p,'dimension':d,'split':split,'all_similitude_coefficients_verified':True})
        OBSERVATIONS['composition_polynomial_certificates']=cases

    def test_02_01_full_enumeration_method_and_quadratic_identity(self):
        rows=[]
        for p,d,split in product((3,5),(4,8),(False,True)):
            actual=enumerate_counts(p,d,split)
            self.assertEqual(actual['total'],p**d)
            counts=polynomial_counts(p,d,split)
            self.assertEqual(actual['idempotents'],counts['idempotents'])
            self.assertEqual(actual['square_zero_including_zero'],counts['nilpotents_including_zero'])
            self.assertEqual(actual['norm_zero_including_zero'],counts['norm_zero_including_zero'])
            rows.append({'p':p,'dimension':d,'split':split,**actual})
        OBSERVATIONS['complete_enumerations']=rows

    def test_02_02_quadratic_identity_coefficient_certificate_through_dimension64(self):
        rows=[]
        for d,split in product((4,8,16,32,64),(False,True)):
            for i,j in combinations(range(d),2):
                x=add(e(i,d),e(j,d),3)
                value=add(scale(x,trace(x,3),3),scale(e(0,d),-norm(x,3,split),3),3)
                self.assertEqual(mul(x,x,3,split),value)
            for i in range(d):
                x=e(i,d)
                self.assertEqual(mul(x,x,3,split),add(scale(x,trace(x,3),3),scale(e(0,d),-norm(x,3,split),3),3))
            rows.append({'dimension':d,'split':split,'all_square_coefficients_verified':True})
        OBSERVATIONS['quadratic_identity_coefficients']=rows

    def test_03_01_theorem1_octonion_idempotent_counts(self):
        rows=[]
        for p,split in product((3,5),(False,True)):
            actual=enumerate_counts(p,8,split)['idempotents']
            self.assertEqual(actual,p**6+p**3+2)
            rows.append({'p':p,'split':split,'enumerated':actual,'printed':paper_id(p,8)})
        OBSERVATIONS['theorem1']=rows

    def test_03_02_theorem1_zorn_cases_and_missing_factors(self):
        p=3;counts=Counter();examples={}
        for x in product(range(p),repeat=8):
            if zorn_mul(x,x,p)!=x:continue
            v,w=x[1:4],x[4:7]
            key='I' if not any(v) and not any(w) else 'II' if not any(v) else 'III' if not any(w) else 'IV'
            counts[key]+=1
            if key=='IV' and x[0] in (0,1) and 'IV_endpoint' not in examples:examples['IV_endpoint']=x
        self.assertEqual(dict(counts),{'I':4,'II':52,'III':52,'IV':650})
        self.assertEqual(sum(counts.values()),paper_id(p,8))
        literal_sum=4+2*(p**3-1)+(p**3-1)*(p**3-2)
        self.assertNotEqual(literal_sum,paper_id(p,8))
        CERTIFICATES['theorem1_endpoint_idempotent_zorn']=examples['IV_endpoint']
        OBSERVATIONS['theorem1_case_counts']={'p':p,'actual':dict(counts),'sum':sum(counts.values()),
             'printed_case_II':p**3-1,'printed_case_III':p**3-1,'printed_case_sum':literal_sum}

    def test_03_03_corollary2_idempotents_in_higher_dimensions(self):
        rows=[]
        for p,d,split in product((3,5,7,11),(4,8,16,32,64),(False,True)):
            actual=polynomial_counts(p,d,split)['idempotents']
            self.assertEqual(actual,paper_id(p,d))
            rows.append({'p':p,'dimension':d,'split':split,'count':actual})
        OBSERVATIONS['corollary2_quadratic_counts']=rows

    def test_03_04_figure1_formula_data(self):
        rows=[]
        for p in PRIMES:
            actual=polynomial_counts(p,8)['idempotents']
            self.assertEqual(actual,paper_id(p,8))
            rows.append({'p':p,'paper':paper_id(p,8),'verified_quadratic_count':actual})
        OBSERVATIONS['figure1']=rows

    def test_03_05_theorem3_nilpotents_and_nilpotency_index(self):
        rows=[]
        for p,split in product((3,5),(False,True)):
            actual=enumerate_counts(p,8,split)['square_zero_including_zero']
            self.assertEqual(actual,p**6)
            for x in random_vectors(p,8,16):
                N=norm(x,p,split);t=trace(x,p)
                square=mul(x,x,p,split)
                self.assertEqual(not any(square),N==0 and t==0)
                power=x
                for k in range(1,p+3):
                    self.assertEqual(not any(power),not any(x) or (N==0 and t==0 and k>=2))
                    power=mul(power,x,p,split)
            rows.append({'p':p,'split':split,'nilpotents_including_zero':actual,'nonzero_nilpotents':actual-1})
        OBSERVATIONS['theorem3']=rows

    def test_03_06_ordinary_matrix_nilpotent_probability_is_not_zorn_proof(self):
        rows=[]
        for p in (3,5,7):
            count=0
            for a,b,c,d in product(range(p),repeat=4):
                sq=((a*a+b*c)%p,(a*b+b*d)%p,(c*a+d*c)%p,(c*b+d*d)%p)
                count+=not any(sq)
            self.assertEqual(count,p*p)
            self.assertNotEqual(p**4,p**8)
            rows.append({'p':p,'ordinary_2x2_matrices':p**4,'ordinary_nilpotents':count,
                         'zorn_elements':p**8,'verified_zorn_nilpotents':p**6})
        # CD octonions are nonassociative; ordinary matrix multiplication is associative.
        a,b,c=e(1,8),e(2,8),e(4,8)
        self.assertNotEqual(mul(mul(a,b,3),c,3),mul(a,mul(b,c,3),3))
        OBSERVATIONS['ordinary_matrix_proof_qualification']=rows

    def test_03_07_corollary4_odd_prime_nilpotent_counts_and_characteristic2(self):
        rows=[]
        for p,d,split in product((3,5,7,11),(4,8,16,32,64),(False,True)):
            actual=polynomial_counts(p,d,split)['nilpotents_including_zero']
            self.assertEqual(actual,paper_nil(p,d))
            rows.append({'p':p,'dimension':d,'split':split,'count':actual})
        OBSERVATIONS['corollary4_odd_prime_counts']=rows
        char2=[]
        for d in (4,8):
            actual=enumerate_counts(2,d)['square_zero_including_zero']
            self.assertEqual(actual,2**(d-1));self.assertNotEqual(actual,paper_nil(2,d))
            char2.append({'p':2,'dimension':d,'actual':actual,'printed':paper_nil(2,d)})
        OBSERVATIONS['corollary4_missing_odd_prime_hypothesis']=char2

    def test_03_08_figure2_formula_data(self):
        rows=[]
        for p in PRIMES:
            actual=quadratic_histogram(norm_weights(8)[1:],p)[0]
            self.assertEqual(actual,paper_nil(p,8))
            rows.append({'p':p,'paper':paper_nil(p,8),'verified_quadratic_count':actual})
        OBSERVATIONS['figure2']=rows

    def test_03_09_theorem5_common_nilpotents_exhaustive_coordinates(self):
        rows=[]
        for p in (3,5):
            total=0
            for x in chunks(p,8):
                # Use both actual squaring operations, not the claimed formula.
                a=batch_product(x,x,p);b=batch_product(x,x,p,True)
                common=np.all(a==0,axis=-1)&np.all(b==0,axis=-1)
                total+=int(np.count_nonzero(common))
            self.assertEqual(total,correct_intersection(p))
            self.assertNotEqual(total,p**3+p*p-p)
            rows.append({'p':p,'actual_including_zero':total,'printed':p**3+p*p-p,'missing_factor':p*p})
        x=(0,1,1,1,0,0,0,0)
        self.assertEqual(mul(x,x,3),(0,)*8);self.assertEqual(mul(x,x,3,True),(0,)*8)
        CERTIFICATES['common_nilpotent_nonzero_ternary_block']={'p':3,'x':x,'main_square':(0,)*8,'split_square':(0,)*8}
        OBSERVATIONS['theorem5']=rows

    def test_03_10_theorem5_factorization_and_figure3_data(self):
        rows=[]
        for p in PRIMES:
            ternary=quadratic_histogram((1,1,1),p)[0]
            quaternary=quadratic_histogram((1,1,1,1),p)[0]
            self.assertEqual(ternary,p*p);self.assertEqual(quaternary,p**3+p*p-p)
            self.assertEqual(ternary*quaternary,correct_intersection(p))
            rows.append({'p':p,'paper':p**3+p*p-p,'ternary_factor':ternary,'quaternary_factor':quaternary,
                         'corrected_coordinate_intersection':ternary*quaternary})
        OBSERVATIONS['figure3']=rows

    def test_03_11_zero_divisor_definition_and_norm_zero_witnesses(self):
        cases=[]
        for p,d,split in product((3,5),(4,8),(False,True)):
            samples=0
            for x in product(range(p),repeat=d):
                if not any(x) or norm(x,p,split):continue
                y=conj(x,p)
                self.assertTrue(any(y));self.assertFalse(any(mul(x,y,p,split)))
                self.assertFalse(any(mul(y,x,p,split)))
                # Independent rank verifies a genuine nontrivial multiplication kernel.
                kernel=annihilator(x,p,split=split)
                self.assertGreater(len(kernel),0)
                for v in kernel:self.assertFalse(any(mul(x,v,p,split)))
                cases.append({'p':p,'dimension':d,'split':split,'x':x,'right_annihilator_basis':kernel,
                              'conjugate_partner':y,'norm':0})
                samples+=1
                if samples==3:break
        CERTIFICATES['finite_field_annihilators']=cases
        for p,d in product((3,5),(4,8)):
            zero=(0,)*d
            self.assertFalse(any(zero))
            self.assertEqual(norm(zero,p),0) # excluded by the paper's ZD definition

    def test_03_12_corollary6_printed_sign_and_zero_convention(self):
        rows=[]
        for p,d,split in product((3,5),(4,8),(False,True)):
            actual=enumerate_counts(p,d,split)
            incl=actual['norm_zero_including_zero'];nonzero=actual['nonzero_zero_divisors']
            self.assertEqual(incl,correct_norm_zero(p,d))
            self.assertEqual(nonzero,incl-1)
            self.assertNotEqual(paper_zd(p,d),incl);self.assertNotEqual(paper_zd(p,d),nonzero)
            rows.append({'p':p,'dimension':d,'split':split,'printed':paper_zd(p,d),
                         'actual_including_zero':incl,'actual_nonzero_zero_divisors':nonzero})
        OBSERVATIONS['corollary6']=rows

    def test_03_13_theorem7_unit_counts_and_actual_two_sided_inverses(self):
        rows=[]
        for p,d,split in product((3,5),(4,8),(False,True)):
            actual=enumerate_counts(p,d,split)['units']
            self.assertEqual(actual,p**d-correct_norm_zero(p,d))
            self.assertNotEqual(actual,paper_units(p,d))
            checked=0
            for x in random_vectors(p,d,40):
                N=norm(x,p,split)
                if N:
                    inverse=scale(conj(x,p),pow(N,-1,p),p)
                    self.assertEqual(mul(x,inverse,p,split),e(0,d))
                    self.assertEqual(mul(inverse,x,p,split),e(0,d));checked+=1
            self.assertGreater(checked,0)
            rows.append({'p':p,'dimension':d,'split':split,'actual_units':actual,'printed':paper_units(p,d),
                         'checked_two_sided_inverse_examples':checked})
        OBSERVATIONS['theorem7']=rows

    def test_03_14_higher_dimensions_unit_and_zero_divisor_overlap(self):
        rows=[]
        for p in (3,5,7,11):
            d=16;a=add(e(1,d),e(10,d),p);b=add(e(4,d),scale(e(15,d),-1,p),p)
            inverse=scale(conj(a,p),pow(norm(a,p),-1,p),p)
            self.assertEqual(norm(a,p),2%p)
            self.assertFalse(any(mul(a,b,p)));self.assertFalse(any(mul(b,a,p)))
            self.assertEqual(mul(a,inverse,p),e(0,d));self.assertEqual(mul(inverse,a,p),e(0,d))
            self.assertNotEqual(mul(mul(inverse,a,p),b,p),mul(inverse,mul(a,b,p),p))
            kernel=annihilator(a,p)
            self.assertGreater(len(kernel),0)
            rows.append({'p':p,'a':a,'b':b,'inverse':inverse,'norm_a':norm(a,p),
                         'ab':(0,)*d,'a_times_inverse':e(0,d),'annihilator_basis':kernel})
        CERTIFICATES['sedenion_unit_zero_divisor_overlap']=rows
        OBSERVATIONS['higher_dimensional_unit_partition_failure']='A two-sided-invertible sedenion can also annihilate a nonzero element. Theorem7 cannot assume units are the complement of zero divisors beyond composition dimensions.'

    def test_04_01_finite_field_signs_do_not_define_causal_types(self):
        for p in (3,5,7):
            self.assertEqual(1%p,(1-p)%p)
            self.assertTrue(1>0 and 1-p<0) # same residue, opposite representative signs
        # Over R, unlike F_p, the intended split norm has signed examples.
        weights=norm_weights(8,True)
        values=[sum(w*x*x for w,x in zip(weights,x)) for x in (e(0,8),e(4,8),(1,0,0,0,1,0,0,0))]
        self.assertEqual(values,[1,-1,0])
        OBSERVATIONS['causal_type_scope']='Norm=0 makes sense over F_p; norm<0 and norm>0 do not define ordered-field notions there. The real split norm separately has positive, negative and isotropic examples.'


class GPUVerification(unittest.TestCase):
    def setUp(self):
        if CK.cp is None:self.skipTest('CuPy is unavailable')
        if not CK.loaded():self.skipTest('Build CayleyKernel CUDA library first')
        if CK.cp.cuda.runtime.getDeviceCount()<1:self.skipTest('No CUDA device')

    def test_05_05_half_exact_modp_products(self):
        # With p<=7 and d<=16, |each integer output| <= 16*6^2=576.
        # FP32 accumulation and FP16 output are both exact on this domain;
        # modular reduction after a rounded FP16 result would be invalid.
        for p in (3,5,7):
            for d in (4,8,16):
                vectors=list(random_vectors(p,d,32))
                a=np.asarray(vectors,dtype=np.float16)
                b=np.asarray(vectors[::-1],dtype=np.float16)
                actual=CK.multiply(a,b)
                self.assertEqual(actual.dtype,np.dtype(np.float16))
                expected=np.asarray([mul(x,y,p,reference=True) for x,y in zip(vectors,vectors[::-1])])
                np.testing.assert_array_equal(actual.astype(np.int64)%p,expected)

    def test_05_01_multiply_modp_basis_and_batched_products(self):
        for p,d in product((3,5,7),(4,8,16)):
            pairs=list(product(range(d),repeat=2))
            A=np.eye(d,dtype=np.int64)[[i for i,j in pairs]]*(p-1)
            B=np.eye(d,dtype=np.int64)[[j for i,j in pairs]]
            expected=np.asarray([mul(a,b,p,reference=True) for a,b in zip(A,B)])
            np.testing.assert_array_equal(np.asarray(CK.multiply(A,B))%p,expected)
            rng=np.random.default_rng(SEED+p+d)
            A=rng.integers(0,p,size=(32,d),dtype=np.int64);B=rng.integers(0,p,size=(32,d),dtype=np.int64)
            expected=np.asarray([mul(a,b,p,reference=True) for a,b in zip(A,B)])
            np.testing.assert_array_equal(np.asarray(CK.multiply(A,B))%p,expected)

    def test_05_02_multiply_full_enumeration_mod3_octonion_squares(self):
        idem=nil=0
        for x in chunks(3,8):
            actual=np.asarray(CK.multiply(x,x))%3
            expected=np.asarray([mul(a,a,3,reference=True) for a in x])
            np.testing.assert_array_equal(actual,expected)
            idem+=int(np.count_nonzero(np.all(actual==x,axis=-1)))
            nil+=int(np.count_nonzero(np.all(actual==0,axis=-1)))
        self.assertEqual((idem,nil),(758,729))

    def test_05_03_multiply_split_products_via_verified_field_isomorphism(self):
        # CK has a fixed signature. Convert split inputs to standard inputs,
        # multiply there, reduce modp, and map back; do not fake a signature flag.
        for p in (3,5,7):
            xs=random_vectors(p,8,32);ys=list(reversed(xs))
            A=np.asarray([split_to_main(x,p) for x in xs],dtype=np.int64)
            B=np.asarray([split_to_main(y,p) for y in ys],dtype=np.int64)
            actual=np.asarray(CK.multiply(A,B))%p
            transported=np.asarray([main_to_split(z,p) for z in actual])
            expected=np.asarray([mul(x,y,p,True,True) for x,y in zip(xs,ys)])
            np.testing.assert_array_equal(transported,expected)

    def test_05_04_multiply_sedenion_unit_and_zero_divisor_certificate(self):
        for p in (3,5,7):
            a=add(e(1,16),e(10,16),p);b=add(e(4,16),scale(e(15,16),-1,p),p)
            inverse=scale(conj(a,p),pow(2,-1,p),p)
            A=np.asarray([a,a,inverse],dtype=np.int64)
            B=np.asarray([b,inverse,a],dtype=np.int64)
            expected=np.asarray([(0,)*16,e(0,16),e(0,16)])
            np.testing.assert_array_equal(np.asarray(CK.multiply(A,B))%p,expected)


try:
    import pytest
except ImportError:
    pass
else:
    PaperVerification=pytest.mark.cpu(PaperVerification)
    GPUVerification=pytest.mark.cpu(GPUVerification)
