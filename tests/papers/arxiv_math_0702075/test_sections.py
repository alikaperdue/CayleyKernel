"""Paper-ordered exact tests. B=sqrt(2)*paper braces; no irrational rank guesses."""
from fractions import Fraction as F
from itertools import product
import random
import unittest

import numpy as np
import CayleyKernel as CK
from tests.papers.type_replay import replay_product
from .algebra import (e,add,scale,conj,dot,norm2,mul,oracle_sign,imaginary_unit,
                      pc,perp,h_indices,hermitian,pair,parts,B,split_B,tilde,J,
                      cspan_projection,rows_from_columns,nullspace,span_rank,same_span,
                      ann,ann_dimension,coordinate_intersection,in_D,divide,
                      constraint_columns,predicted_ann_h,d_decomposition,max_ann,
                      maximal_pair,modular_rank)

OBSERVATIONS={}
CERTIFICATES={}
SEED=702075


def samples(n,count=5,restricted=True):
    rng=random.Random(SEED+n)
    result=[]
    for _ in range(count):
        a={i:rng.randint(-2,2) for i in range(1<<n)}
        result.append(perp(a,n) if restricted else {i:x for i,x in a.items() if x})
    return result


def pairs(n):
    a,b=maximal_pair(n)
    out=[(e(1),e(1)),(e(1),e(2)),(e(1),mul(imaginary_unit(n),e(1),n)),
         (a,b),(a,a),(a,e(2))]
    if ann(a,n):
        out.append((a,ann(a,n)[0]))
    out += list(zip(samples(n,2),reversed(samples(n,2))))
    return out


def formula45(a,b,x,y,n):
    first=B(perp(mul(a,x,n),n),perp(mul(b,y,n),n),n)
    second=tilde(pc(add(mul(x,a,n),mul(b,y,n)),n),n)
    third=mul(tilde(pc(add(mul(a,y,n),scale(mul(x,b,n),-1)),n),n),J(n),n+1)
    return first,second,third


def zd4_pool():
    return [pair(e(i),scale(e(j),s),3)
            for i,j,s in product(range(1,8),range(1,8),(-1,1)) if i!=j]


def geometry4(a):
    p,q=parts(a,4)
    c=scale(mul(p,q,3),F(1,norm2(p)))
    quaternion=[e(0),p,q,mul(p,q,3)]
    orth=nullspace([dict(v) for v in quaternion],8)
    minus=[pair(y,scale(mul(c,y,3),-1),3) for y in orth]
    plus=[pair(y,mul(c,y,3),3) for y in orth]
    six=[pair(p,scale(q,-1),3),pair(q,p,3)]+plus
    return quaternion,minus,plus,six


def record_certificate(name,a,n):
    kernel=ann(a,n)
    dimension=1<<n
    columns=[mul(a,e(j),n) for j in range(dimension)]
    rank_mod=modular_rank(columns,dimension)
    kernel_rank_mod=modular_rank(kernel,dimension)
    if any(mul(a,v,n) for v in kernel) or rank_mod!=dimension-len(kernel) or kernel_rank_mod!=len(kernel):
        raise AssertionError('annihilator certificate failed independent validation: '+name)
    CERTIFICATES[name]={'level':n,'dimension':1<<n,'element':a,'norm_squared':norm2(a),
                        'nullity':len(kernel),'annihilator_basis':kernel,
                        'operator_rank_mod_1000003':rank_mod,'kernel_rank_mod_1000003':kernel_rank_mod,
                        'all_kernel_products_exactly_zero':True}


class PaperVerification(unittest.TestCase):
    def test_01_intro_splitting_annihilator_dimension_and_d_locus_examples(self):
        rows=[]
        for n in (3,4,5):
            for a,b in pairs(n):
                ra,rb=ann_dimension(a,n),ann_dimension(b,n)
                actual=ann_dimension(B(a,b,n),n+1)
                d=in_D(a,b,n)
                self.assertIn(actual,(ra+rb,ra+rb+4))
                self.assertEqual(actual,ra+rb+4*int(d))
                rows.append({'source_level':n,'a':a,'b':b,'ann_a':ra,'ann_b':rb,
                             'D_locus':d,'ann_B':actual})
        OBSERVATIONS['intro_dimension_cases']=rows

    def test_02_01_all_basis_signs_and_signed_labels_against_independent_doubling(self):
        for n in range(1,7):
            for i,j in product(range(1<<n),repeat=2):
                s=oracle_sign(i,j,1<<n)
                self.assertEqual(CK.twist(i,j),s)
                for u,v in product((-1,1),repeat=2):
                    self.assertEqual(CK.basis(u*(i+1),v*(j+1)),u*v*s*((i^j)+1))
        for n in (3,4,5):
            for a,b in zip(samples(n,4,False),reversed(samples(n,4,False))):
                self.assertEqual(mul(a,b,n),mul(a,b,n,True))
                self.assertEqual(mul(a,conj(a),n),scale(e(0),norm2(a)))
                self.assertEqual(conj(mul(a,b,n)),mul(conj(b),conj(a),n))

    def test_02_02_complex_action_real_hermitian_and_lemma_2_3(self):
        for n in (3,4,5):
            i=imaginary_unit(n)
            alpha,beta=add(scale(e(0),2),i),add(e(0),scale(i,-3))
            for x,y in zip(samples(n,4,False),reversed(samples(n,4,False))):
                self.assertEqual(mul(alpha,mul(beta,x,n),n),mul(mul(alpha,beta,n),x,n))
                self.assertEqual(mul(x,conj(y),n).get(0,0),dot(x,y))
                self.assertEqual(hermitian(x,y,n),conj(hermitian(y,x,n)))
                self.assertEqual(hermitian(x,x,n),scale(e(0),norm2(x)))
                imaginary={j:v for j,v in y.items() if j}
                self.assertEqual(dot(x,mul(x,imaginary,n)),0)

    def test_02_03_lemmas_2_4_to_2_6_conjugate_linearity_and_orthogonality(self):
        for n in (3,4,5):
            i=imaginary_unit(n);alpha=add(e(0),scale(i,2));beta=add(scale(e(0),3),scale(i,-1))
            for a,x,y in zip(samples(n,4),samples(n,4,False),reversed(samples(n,4,False))):
                self.assertEqual(mul(a,mul(alpha,x,n),n),mul(conj(alpha),mul(a,x,n),n))
                self.assertEqual(hermitian(mul(a,x,n),y,n),scale(conj(hermitian(x,mul(a,y,n),n)),-1))
                self.assertEqual(mul(mul(alpha,a,n),mul(beta,a,n),n),
                                 scale(mul(alpha,conj(beta),n),-norm2(a)))
                self.assertEqual(norm2(mul(alpha,a,n)),norm2(alpha)*norm2(a))
            a,b=e(1),e(2)
            self.assertEqual(hermitian(a,b,n),{})
            self.assertEqual(mul(mul(alpha,a,n),mul(beta,b,n),n),
                             mul(mul(conj(alpha),conj(beta),n),mul(a,b,n),n))

    def test_02_04_h_quaternion_subalgebra_and_perpendicular_coordinates(self):
        for n in range(3,7):
            h=h_indices(n)
            for a,b,c in product(h,repeat=3):
                self.assertEqual(mul(mul(e(a),e(b),n),e(c),n),mul(e(a),mul(e(b),e(c),n),n))
                self.assertTrue(set(mul(e(a),e(b),n))<=set(h))
            self.assertEqual(len(h),4)
            for a,b in ((e(1),e((1<<(n-2))+1)),(e(1),{}),({},e(1))):
                self.assertFalse(set(B(a,b,n-1)) & set(h))

    def test_02_05_exact_annihilator_kernels_modular_rank_image_and_bounds(self):
        rows=[]
        for n in range(3,8):
            a,b=maximal_pair(n)
            cases=[e(1),a,b]+samples(n,1)+[add(a,e(0)),add(a,imaginary_unit(n))]
            for index,x in enumerate(cases):
                kernel=ann(x,n);dim=1<<n
                for v in kernel:
                    self.assertEqual(mul(x,v,n),{})
                    self.assertEqual(mul(x,v,n,True),{})
                    self.assertEqual(pc(v,n),{})
                    self.assertEqual(mul(v,x,n),{})
                columns=[mul(x,e(j),n) for j in range(dim)]
                self.assertEqual(modular_rank(columns,dim),dim-len(kernel))
                self.assertEqual(modular_rank(kernel,dim),len(kernel))
                self.assertEqual(len(kernel)%4,0)
                self.assertLessEqual(len(kernel),max_ann(n))
                for image in columns:
                    self.assertTrue(all(dot(image,v)==0 for v in kernel))
                if kernel:
                    self.assertEqual(pc(x,n),{})
                rows.append({'n':n,'case':index,'nullity':len(kernel),'rank':dim-len(kernel)})
            record_certificate('maximal_source_A%d'%n,a,n)
        OBSERVATIONS['exact_rank_and_bounds']=rows

    def test_02_06_projection_lemmas_with_required_restriction(self):
        for n in (3,4,5):
            alpha=add(e(0),imaginary_unit(n))
            for a,b in zip(samples(n,4),samples(n,4,False)):
                bp=cspan_projection(a,b,n);bpp=add(b,scale(bp,-1))
                self.assertEqual(hermitian(a,bpp,n),{})
                self.assertEqual(pc(mul(a,b,n),n),mul(a,bp,n))
                self.assertEqual(perp(mul(a,b,n),n),mul(a,bpp,n))
                self.assertEqual(pc(mul(b,a,n),n),mul(bp,a,n))
                self.assertEqual(perp(mul(b,a,n),n),mul(bpp,a,n))
                self.assertEqual(pc(mul(alpha,b,n),n),mul(alpha,pc(b,n),n))
                self.assertEqual(pc(mul(b,alpha,n),n),mul(alpha,pc(b,n),n))
            for a,b in pairs(n):
                preimage=nullspace(rows_from_columns([perp(mul(a,e(j),n),n) for j in range(1<<n)],1<<n),1<<n)
                predicted=[a,mul(imaginary_unit(n),a,n)]+ann(a,n)
                self.assertTrue(same_span(preimage,predicted,1<<n))
                self.assertEqual(pc(mul(a,b,n),n),conj(pc(mul(b,a,n),n)))
                self.assertEqual(perp(mul(a,b,n),n),scale(perp(mul(b,a,n),n),-1))

    def test_02_07_lemma_2_15_and_corollary_2_18_printed_generality_counterexample(self):
        a=add(e(0),e(1));b=a;n=3
        self.assertEqual(cspan_projection(a,b,n),a)
        self.assertEqual(mul(a,b,n),scale(e(1),2))
        self.assertEqual(pc(mul(a,b,n),n),{})
        self.assertNotEqual(pc(mul(a,b,n),n),mul(a,cspan_projection(a,b,n),n))
        self.assertEqual(ann(a,n),[])
        OBSERVATIONS['projection_missing_hypothesis']={'a':a,'b':b,'ab':mul(a,b,n),
                                                     'printed_conditions':'a,b arbitrary; a in C_perp suffices for tested uses'}

    def test_02_08_zero_exception_in_theorem_2_11(self):
        for n in (3,4,5):
            self.assertEqual(ann_dimension({},n),1<<n)
            self.assertGreater(ann_dimension({},n),max_ann(n))
        OBSERVATIONS['annihilator_bound_zero_exception']='Ann(0)=A_n; upper bound requires nonzero element'

    def test_03_01_split_complex_scalar_action_and_metric(self):
        for n in (3,4,5,6):
            alpha=add(scale(e(0),2),scale(imaginary_unit(n),-3))
            for a,b in zip(samples(n,3),reversed(samples(n,3))):
                z=B(a,b,n)
                self.assertEqual(split_B(z,n+1),(a,b))
                self.assertEqual(norm2(z),2*(norm2(a)+norm2(b)))
                self.assertEqual(mul(tilde(alpha,n),z,n+1),B(mul(conj(alpha),a,n),mul(alpha,b,n),n))

    def test_04_01_proposition_4_1_orthogonal_special_cases(self):
        for n in (3,4,5):
            a,b=e(1),mul(add(e(0),imaginary_unit(n)),e(1),n)
            x,y=e(2),mul(add(scale(e(0),2),imaginary_unit(n)),e(2),n)
            self.assertTrue(all(not hermitian(u,v,n) for u in (a,b) for v in (x,y)))
            self.assertEqual(mul(B(a,b,n),B(x,y,n),n+1),scale(B(mul(a,x,n),mul(b,y,n),n),2))
            self.assertEqual(mul(B(a,{},n),B({},y,n),n+1),{})

    def test_04_02_lemma_4_3_corollary_4_4_and_J_action(self):
        for n in (3,4,5):
            for a in samples(n,3):
                self.assertEqual(mul(B({},a,n),B(a,{},n),n+1),scale(J(n),2*norm2(a)))
                alpha=add(e(0),imaginary_unit(n));b=mul(alpha,a,n)
                p=tilde(pc(mul(a,b,n),n),n)
                self.assertEqual(mul(B(a,{},n),B(b,{},n),n+1),scale(conj(p),2))
                self.assertEqual(mul(B({},a,n),B({},b,n),n+1),scale(p,2))
                self.assertEqual(mul(B(a,{},n),B({},b,n),n+1),scale(mul(p,J(n),n+1),2))
                self.assertEqual(mul(B({},a,n),B(b,{},n),n+1),scale(mul(conj(p),J(n),n+1),-2))
                self.assertEqual(mul(J(n),B(a,b,n),n+1),B(b,scale(a,-1),n))
                self.assertEqual(mul(B(a,b,n),J(n),n+1),scale(B(b,scale(a,-1),n),-1))

    def test_04_03_theorem_4_5_exhaustive_bilinear_basis_and_orthogonal_terms(self):
        counts={}
        for n in range(3,7):
            u=[e(j) for j in range(1<<n) if j not in (0,1<<(n-1))]
            coordinates=[(x,{}) for x in u]+[({},y) for y in u]
            for (a,b),(x,y) in product(coordinates,repeat=2):
                terms=formula45(a,b,x,y,n)
                self.assertEqual(mul(B(a,b,n),B(x,y,n),n+1),scale(add(*terms),2))
                self.assertTrue(all(dot(t,s)==0 for t,s in ((terms[0],terms[1]),(terms[0],terms[2]),(terms[1],terms[2]))))
            counts[str(n+1)]=len(coordinates)**2
        OBSERVATIONS['theorem_4_5_all_real_bilinear_certificates']=counts

    def test_04_04_inner_products_lemmas_4_9_and_4_10(self):
        for n in (3,4,5):
            data=samples(n,5)
            for a,b,x,y in zip(data,data[1:],reversed(data),data[2:]+data[:2]):
                expected=tilde(add(conj(hermitian(a,x,n)),hermitian(b,y,n)),n)
                self.assertEqual(hermitian(B(a,b,n),B(x,y,n),n+1),scale(expected,2))
                self.assertEqual(dot(B(a,b,n),B(x,y,n)),2*(dot(a,x)+dot(b,y)))

    def test_04_05_iterated_nonquaternion_four_dimensional_subalgebras(self):
        a,b=e(1),e(2);rows=[]
        for n in range(3,8):
            lam=n-2
            self.assertEqual(norm2(a),1);self.assertEqual(norm2(b),1)
            self.assertEqual(hermitian(a,b,n),{})
            ab=mul(a,b,n)
            self.assertEqual(mul(a,ab,n),scale(b,-lam))
            self.assertEqual(mul(b,mul(b,a,n),n),scale(a,-lam))
            basis=[e(0),a,b,ab]
            self.assertEqual(span_rank(basis,1<<n),4)
            for x,y in product(basis,repeat=2):
                self.assertEqual(span_rank(basis+[mul(x,y,n)],1<<n),4)
            self.assertEqual(norm2(ab),lam)
            self.assertEqual(mul(b,ab,n),scale(a,lam))
            self.assertEqual(mul(ab,a,n),scale(b,lam))
            self.assertEqual(mul(ab,ab,n),scale(e(0),-lam))
            rows.append({'level':n,'lambda':lam,'unit_a':a,'unit_b':b})
            a,b=scale(B(a,b,n),F(1,2)),scale(B(b,scale(a,-1),n),F(1,2))
        OBSERVATIONS['four_dimensional_lambda_construction']=rows

    def test_04_06_mutually_annihilating_families_lemma_4_12_corollary_4_13(self):
        xs,ys=[e(1)],[e(2)];rows=[]
        for n in range(3,8):
            expected=1<<(n-3)
            self.assertEqual(len(xs),expected);self.assertEqual(len(ys),expected)
            for family in (xs,ys):
                self.assertEqual(span_rank(family,1<<n),expected)
                for i,x in enumerate(family):
                    self.assertTrue(x)
                    for j,y in enumerate(family):
                        if i!=j:
                            self.assertEqual(mul(x,y,n),{})
            self.assertTrue(all(not hermitian(x,y,n) for x,y in product(xs,ys)))
            rows.append({'level':n,'family_size':expected,'degenerate_algebra_dimension':1+expected})
            xs,ys=([B(x,{},n) for x in xs]+[B({},y,n) for y in ys],
                   [B(y,{},n) for y in ys]+[B({},x,n) for x in xs])
        OBSERVATIONS['mutually_annihilating_families']=rows

    def test_05_01_proposition_5_1_annihilation_equations_positive_and_negative(self):
        positives=negatives=0
        for n in (3,4,5):
            for a,b in pairs(n):
                targets=[(e(1),e(2)),(a,b),({},{}),split_B(ann(B(a,b,n),n+1)[0],n+1)
                         if ann(B(a,b,n),n+1) else (e(2),e(1))]
                for x,y in targets:
                    equations=(perp(mul(a,x,n),n),perp(mul(b,y,n),n),
                               add(mul(x,a,n),mul(b,y,n)),
                               pc(add(mul(a,y,n),scale(mul(x,b,n),-1)),n))
                    actual=not mul(B(a,b,n),B(x,y,n),n+1)
                    self.assertEqual(actual,not any(equations))
                    positives+=actual;negatives+=not actual
        self.assertGreater(positives,0);self.assertGreater(negatives,0)
        OBSERVATIONS['proposition_5_1_cases']={'positive':positives,'negative':negatives}

    def test_05_02_proposition_5_2_entire_parameterized_intersection(self):
        for n in (3,4,5):
            for a,b in pairs(n):
                actual=coordinate_intersection(ann(B(a,b,n),n+1),h_indices(n+1))
                predicted=predicted_ann_h(a,b,n)
                self.assertTrue(same_span(actual,predicted,1<<(n+1)))
                self.assertTrue(all(not mul(B(a,b,n),v,n+1) for v in predicted))

    def test_05_03_theorem_5_10_exhaustive_basis_pairs_and_sedenion_sample_census(self):
        counts={};records=[]
        cases=[(3,e(i),e(j)) for i,j in product((1,2,3,5,6,7),repeat=2)]
        cases += [(4,e(i),e(j)) for i,j in product([k for k in range(16) if k not in (0,8)],repeat=2)]
        fixed=pair(e(1),e(2),3)
        cases += [(4,fixed,b) for b in zd4_pool()]
        for n,a,b in cases:
            actual=ann_dimension(B(a,b,n),n+1)
            ra,rb=ann_dimension(a,n),ann_dimension(b,n)
            d=in_D(a,b,n)
            self.assertEqual(actual,ra+rb+4*int(d))
            key=(n,ra,rb,d,actual)
            counts[str(key)]=counts.get(str(key),0)+1
            records.append({'source_level':n,'a':a,'b':b,'ann_a':ra,'ann_b':rb,'D':d,'ann_B':actual})
        self.assertEqual(len(records),316)
        OBSERVATIONS['annihilator_pair_census']=records
        OBSERVATIONS['annihilator_pair_census_summary']=counts

    def test_05_04_proposition_5_11_entire_annihilators_and_zero_exception(self):
        for n in range(3,7):
            for a in (e(1),maximal_pair(n)[0]):
                orth=nullspace([e(0),imaginary_unit(n),a,mul(imaginary_unit(n),a,n)],1<<n)
                predicted=[B(x,{},n) for x in ann(a,n)]+[B({},y,n) for y in orth]
                actual=ann(B(a,{},n),n+1)
                self.assertEqual(len(actual),ann_dimension(a,n)+(1<<n)-4)
                self.assertTrue(same_span(actual,predicted,1<<(n+1)))
                swapped=[B({},x,n) for x in ann(a,n)]+[B(y,{},n) for y in orth]
                self.assertTrue(same_span(ann(B({},a,n),n+1),swapped,1<<(n+1)))
        self.assertEqual(ann_dimension(B({}, {},3),4),16)
        self.assertNotEqual(ann_dimension(B({}, {},3),4),ann_dimension({},3)+8-4)
        OBSERVATIONS['proposition_5_11_zero_exception']={'actual':16,'printed_dimension_expression':12}

    def test_06_01_d_locus_vanishing_constraints_image_characterization(self):
        proof_projection_example=None
        for n in (3,4,5):
            for a,b in pairs(n):
                d=in_D(a,b,n)
                constraints,_=constraint_columns(a,b,n)
                self.assertEqual(d,all(not c.get(2,0) and not c.get(3,0) for c in constraints))
                self.assertEqual(all(not dot(a,x) for x in ann(b,n)),
                                 all(not hermitian(a,x,n) for x in ann(b,n)))
                if d:
                    self.assertEqual(mul(a,divide(a,b,n),n),b)
                    self.assertEqual(mul(b,divide(b,a,n),n),a)
                    for y in ann(b,n):
                        ay=mul(a,y,n)
                        if ay and not pc(ay,n):
                            proof_projection_example={'n':n,'a':a,'b':b,'y':y,'ay':ay}
        self.assertIsNotNone(proof_projection_example)
        OBSERVATIONS['lemma_6_3_proof_missing_projection']=proof_projection_example

    def test_06_02_lemma_6_4_theorem_6_7_off_d_and_one_zero_exception(self):
        count=0
        for n in (3,4,5):
            for a,b in pairs(n):
                if not in_D(a,b,n):
                    kernel=ann(B(a,b,n),n+1)
                    self.assertEqual(len(kernel),ann_dimension(a,n)+ann_dimension(b,n))
                    self.assertTrue(all(not (set(v)&set(h_indices(n+1))) for v in kernel))
                    count+=1
                    if kernel:
                        record_certificate('off_D_A%d_case%d'%(n+1,count),B(a,b,n),n+1)
        a=e(1)
        self.assertFalse(in_D(a,{},3))
        actual=ann_dimension(B(a,{},3),4)
        self.assertEqual(len(coordinate_intersection(ann(B(a,{},3),4),h_indices(4))),actual)
        printed=ann_dimension(a,3)+ann_dimension({},3)
        self.assertEqual((actual,printed),(4,8))
        OBSERVATIONS['lemma_6_4_missing_nonzero_components']={'actual':actual,'printed':printed,'nonzero_pair_controls':count}

    def test_06_03_definition_6_10_division_and_lemma_6_9_wrong_annihilator(self):
        a=maximal_pair(4)[0];kernel=ann(a,4)
        b=next(e(i) for i in range(1,16) if all(not dot(e(i),v) for v in kernel))
        self.assertEqual(ann(b,4),[])
        self.assertEqual(divide({}, {},3),{})
        with self.assertRaises(ValueError):
            divide(a,kernel[0],4)
        x=divide(a,b,4);other=add(x,kernel[0])
        self.assertNotEqual(x,other)
        self.assertEqual(mul(a,x,4),b);self.assertEqual(mul(a,other,4),b)
        self.assertTrue(all(not dot(x,v) for v in kernel))
        self.assertTrue(all(not dot(x,v) and not dot(other,v) for v in ann(b,4)))
        OBSERVATIONS['lemma_6_9_nonuniqueness']={'a':a,'b':b,'x':x,'other_x':other,
                                                'ann_a':len(kernel),'ann_b':0}

    def test_06_04_lemma_6_11_and_theorem_6_12_full_orthogonal_decomposition(self):
        count=0
        for n in (3,4,5):
            for a,b in pairs(n):
                if not in_D(a,b,n):
                    continue
                quotient=divide(a,b,n)
                self.assertEqual(pc(quotient,n),{})
                self.assertEqual(hermitian(quotient,a,n),{})
                self.assertEqual(hermitian(quotient,b,n),{})
                blocks=d_decomposition(a,b,n)
                actual=ann(B(a,b,n),n+1)
                self.assertTrue(same_span(actual,sum(blocks,[]),1<<(n+1)))
                for k,l in product(range(3),repeat=2):
                    if k!=l:
                        self.assertTrue(all(not hermitian(x,y,n+1) for x,y in product(blocks[k],blocks[l])))
                self.assertEqual(len(actual),ann_dimension(a,n)+ann_dimension(b,n)+4)
                count+=1
                record_certificate('D_decomposition_A%d_case%d'%(n+1,count),B(a,b,n),n+1)
        self.assertGreaterEqual(count,6)
        OBSERVATIONS['full_D_decompositions_checked']=count

    def test_07_01_lemmas_7_1_7_2_and_proposition_7_3(self):
        rows=[];positive=negative=0
        base=[{},e(1),e(2),e(3),e(5),e(6),e(7)]
        for a in (e(1),e(2)):
            A=B(a,{},3)
            for b,c in product(base,repeat=2):
                z=B(b,c,3)
                if not z or not ann(z,4):
                    continue
                expected=not hermitian(a,b,3) and cspan_projection(a,c,3)==c
                self.assertEqual(in_D(A,z,4),expected)
                positive+=expected;negative+=not expected
                if not hermitian(a,b,3) and cspan_projection(a,c,3)==c:
                    self.assertTrue(all(not dot(A,v) for v in ann(z,4)))
                rows.append({'a':a,'b':b,'c':c,'D':expected})
            for z in zd4_pool():
                if not hermitian(A,z,4) and all(not dot(z,v) for v in ann(A,4)):
                    b,c=split_B(z,4)
                    self.assertEqual(B(b,c,3),z)
                    self.assertEqual(hermitian(a,b,3),{})
                    self.assertEqual(cspan_projection(a,c,3),c)
        self.assertGreater(positive,0);self.assertGreater(negative,0)
        OBSERVATIONS['proposition_7_3_cases']={'positive':positive,'negative':negative}
        # Lemma 7.1 allows a degenerate all-zero second argument literally.
        self.assertFalse(all(not dot(B(e(1),{},3),v) for v in ann({},4)))

    def test_07_02_annihilator_eigenspace_geometry_and_missing_norm_factor(self):
        rows=[]
        for a in zd4_pool()[::7]:
            quaternion,minus,plus,six=geometry4(a)
            self.assertTrue(same_span(ann(a,4),minus,16))
            self.assertEqual(span_rank(plus,16),4)
            self.assertEqual(span_rank(six,16),6)
            for x in plus:
                self.assertEqual(mul(a,mul(a,x,4),4),scale(x,-2*norm2(a)))
                self.assertNotEqual(mul(a,mul(a,x,4),4),scale(x,-2))
            product_h=[pair(x,{},3) for x in quaternion]+[pair({},x,3) for x in quaternion]
            self.assertEqual(span_rank(minus+plus+product_h,16),16)
            for block,other in ((minus,plus),(minus,product_h),(plus,product_h)):
                self.assertTrue(all(not dot(x,y) for x,y in product(block,other)))
            rows.append({'a':a,'norm_squared':norm2(a),'actual_positive_eigenvalue':2*norm2(a)})
        OBSERVATIONS['section_7_eigenvalue_scaling']=rows

    def test_07_03_corollary_7_4_six_plane_membership_against_D_definition(self):
        pool=zd4_pool();positive=negative=0
        for a in pool[::14]:
            _,_,_,six=geometry4(a)
            for b in pool:
                expected=span_rank(six+[b],16)==6
                self.assertEqual(in_D(a,b,4),expected)
                positive+=expected;negative+=not expected
        self.assertGreater(positive,0);self.assertGreater(negative,0)
        OBSERVATIONS['corollary_7_4_census']={'positive':positive,'negative':negative,'total':positive+negative}

    def test_07_04_theorem_7_5_local_six_plane_and_bundle_map_checks(self):
        for a in zd4_pool()[::14]:
            _,minus,plus,six=geometry4(a)
            # The second-half sign flip identifies the four-dimensional fibers.
            flipped=[]
            for v in plus:
                p,q=parts(v,4)
                flipped.append(pair(p,scale(q,-1),3))
            self.assertTrue(same_span(flipped,minus,16))
            halves=[parts(v,4) for v in six]
            for (p,q),(r,s) in product(halves,repeat=2):
                self.assertEqual(p.get(0,0),0);self.assertEqual(q.get(0,0),0)
                self.assertEqual(dot(p,r),dot(q,s))
                self.assertEqual(dot(p,s)+dot(r,q),0)
            b=add(*(scale(v,j+1) for j,v in enumerate(six)))
            self.assertEqual(ann_dimension(b,4),4)
            self.assertTrue(in_D(a,b,4))
            self.assertEqual(ann_dimension(B(a,b,4),5),12)
        frame=pair(e(1),e(2),3)
        self.assertEqual(norm2(frame),2) # Stiefel frame needs 1/sqrt(2) for unit A4 vector.
        OBSERVATIONS['topology_scope']={'fiber_rank':6,'unit_fiber_dimension':5,
                                        'unit_frame_squared_length_before_scaling':2,
                                        'global_diffeomorphism_verified':False}

    def test_08_01_theorem_8_1_large_annihilators_in_H_perp(self):
        cases=[]
        for n in range(4,9):
            a,b=maximal_pair(n)
            record_certificate('maximal_source_A%d'%n,a,n)
            cases.append((n,a))
            if n>4:
                x,y=maximal_pair(n-1)
                cases.append((n,B(x,y,n-1)))
        for n,a in cases:
            if ann_dimension(a,n)>=(1<<(n-1)):
                self.assertFalse(set(a)&set(h_indices(n)))

    def test_08_02_proposition_8_2_full_outside_H_annihilators(self):
        rows=[]
        for n in range(4,9):
            m=n-1;a=e(1);i=imaginary_unit(m)
            z=pair(i,a,m)
            orth=nullspace([e(0),i,a,mul(i,a,m)],1<<m)
            predicted=[pair(x,mul(mul(a,i,m),x,m),m) for x in orth]
            actual=ann(z,n)
            for k in range(1<<m):
                self.assertEqual(mul(a,mul(a,e(k),m),m),mul(mul(a,a,m),e(k),m))
            self.assertTrue(all(not mul(z,v,n) for v in predicted))
            self.assertEqual(len(actual),(1<<m)-4)
            self.assertTrue(same_span(actual,predicted,1<<n))
            self.assertTrue(set(z)&set(h_indices(n)))
            rows.append({'level':n,'nullity':len(actual),'half_dimension':1<<m})
        OBSERVATIONS['outside_H_sharp_examples']=rows

    def test_08_03_propositions_8_6_8_7_maximal_branches_and_stable_boundary(self):
        rows=[]
        for n in range(4,8):
            # Boundary c=4*(n-4): for n>=5 use a nonmaximal sharp element.
            a=maximal_pair(n)[0] if n==4 else B(*maximal_pair(n-1),n-1)
            c=4*(n-4)
            self.assertEqual(ann_dimension(a,n),max_ann(n)-c)
            for side,z in (('left',B(a,{},n)),('right',B({},a,n))):
                actual=ann_dimension(z,n+1)
                self.assertEqual(actual,max_ann(n+1)-c)
                self.assertGreaterEqual(actual,(1<<(n+1))-8*(n+1)+24)
                x,y=split_B(z,n+1)
                self.assertEqual((not x)+(not y),1)
                self.assertEqual(norm2(z),2*norm2(a))
                record_certificate('stable_boundary_A%d_%s'%(n+1,side),z,n+1)
                rows.append({'source_level':n,'c':c,'source_nullity':ann_dimension(a,n),
                             'target_nullity':actual,'side':side,'unit_normalization':'divide by sqrt(norm_squared)'})
        OBSERVATIONS['stable_boundary_ray_examples']=rows

    def test_08_04_lemmas_8_8_8_9_maximal_D_pairs_and_sharp_bounds(self):
        rows=[]
        for n in range(3,8):
            a,b=maximal_pair(n)
            self.assertEqual(ann_dimension(a,n),max_ann(n))
            self.assertEqual(ann_dimension(b,n),max_ann(n))
            self.assertTrue(in_D(a,b,n))
            z=B(a,b,n)
            actual=ann_dimension(z,n+1)
            self.assertEqual(actual,(1<<(n+1))-8*n+12)
            x,y=split_B(z,n+1)
            self.assertTrue(x and y)
            if n>=4:
                bound=(1<<(n+1))-8*(n+1)+24
                self.assertEqual(actual,bound-4)
            rows.append({'source_level':n,'component_nullity':max_ann(n),'target_nullity':actual,
                         'deficit_from_max':max_ann(n+1)-actual})
        OBSERVATIONS['sharp_two_component_ray_examples']=rows

    def test_08_05_proposition_8_11_theorem_8_12_threshold_and_nonstable_witnesses(self):
        arithmetic=0;rows=[]
        for n in range(4,13):
            for c in range(0,(1<<n)-4*n+1,4):
                stable=(4*n>=c+16)
                self.assertEqual(stable,c<=4*(n-4))
                self.assertEqual(not stable,n<=F(c,4)+3)
                arithmetic+=1
        for n in range(5,8):
            a,b=maximal_pair(n)
            z=B(a,b,n)
            c=4*(n-3)
            self.assertLessEqual(c,(1<<n)-4*n)
            self.assertEqual(ann_dimension(z,n+1),max_ann(n+1)-c)
            self.assertTrue(all(split_B(z,n+1)))
            rows.append({'source_level':n,'first_nonstable_c':c,'target_nullity':ann_dimension(z,n+1),
                         'both_components_nonzero':True,'unit_normalization':'ray representative'})
        OBSERVATIONS['stability_parameter_checks']={'integer_parameter_cases':arithmetic,'nonstable_ray_witnesses':rows}

    def test_08_06_corollaries_8_13_8_14_dimension_examples_and_unit_scope(self):
        a=maximal_pair(4)[0]
        self.assertEqual(ann_dimension(B(a,{},4),5),16)
        self.assertEqual(ann_dimension(B({},a,4),5),16)
        source=B(*maximal_pair(4),4)
        self.assertEqual(ann_dimension(source,5),12)
        for z in (B(source,{},5),B({},source,5)):
            self.assertEqual(ann_dimension(z,6),40)
        # Multiplying a by arbitrary positive scalars preserves its kernel.
        self.assertTrue(same_span(ann(a,4),ann(scale(a,100),4),16))
        self.assertEqual(norm2(scale(a,100)),10000*norm2(a))
        OBSERVATIONS['corollary_8_13_scope']='Unit-length locus is compact; unrestricted nonzero scalar rays are unbounded.'


class GPUVerification(unittest.TestCase):
    def setUp(self):
        if CK.cp is None:
            self.skipTest('CuPy unavailable; multiply() not executed')
        if not CK.loaded():
            self.skipTest('cuda_kernel.so unavailable; build with make kernel')
        try:
            devices=CK.cp.cuda.runtime.getDeviceCount()
        except CK.cp.cuda.runtime.CUDARuntimeError as exc:
            self.skipTest('CUDA device unavailable: '+str(exc))
        if not devices:
            self.skipTest('No CUDA device')

    def test_02_multiply_basis_and_dense_independent_oracle(self):
        for n in range(1,7):
            dim=1<<n
            pairs_=[(e(i),e(j)) for i,j in product(range(dim),repeat=2)]
            pairs_+=list(zip(samples(n,4,False),reversed(samples(n,4,False))))
            dense=lambda x:[x.get(i,0) for i in range(dim)]
            a=np.array([dense(a) for a,b in pairs_],dtype=float)
            b=np.array([dense(b) for a,b in pairs_],dtype=float)
            expected=np.array([dense(mul(a,b,n,True)) for a,b in pairs_])
            np.testing.assert_array_equal(replay_product(a,b),expected)

    def test_04_multiply_full_splitting_products(self):
        for n in (3,4,5):
            a,b,x,y=samples(n,4)
            u,v=B(a,b,n),B(x,y,n)
            expected=scale(add(*formula45(a,b,x,y,n)),2)
            dim=1<<(n+1);dense=lambda x:np.array([float(x.get(i,0)) for i in range(dim)])
            np.testing.assert_array_equal(replay_product(dense(u),dense(v)),dense(expected))

    def test_05_08_multiply_exact_annihilator_basis_certificates(self):
        for n in range(4,8):
            a=maximal_pair(n)[0];kernel=ann(a,n);dim=1<<n
            dense=lambda x:[float(x.get(i,0)) for i in range(dim)]
            output=replay_product(np.array(dense(a)),np.array([dense(v) for v in kernel]))
            np.testing.assert_allclose(output,np.zeros((len(kernel),dim)),rtol=0,atol=1e-12)


try:
    import pytest
except ImportError:
    pass
else:
    PaperVerification=pytest.mark.cpu(PaperVerification)
    GPUVerification=pytest.mark.cpu(GPUVerification)
