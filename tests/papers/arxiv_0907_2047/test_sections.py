"""Paper-ordered exact finite loop tests and explicitly bounded algebra checks."""
from collections import Counter
from fractions import Fraction as F
from itertools import product
import unittest

import numpy as np
import CayleyKernel as CK
from tests.papers.type_replay import replay_product
from .loops import (REPS, SIGNS, TABLE, ELEMENTS, POSITION, INVERSES,
                    support_mask, indices, labels, independent_sign, signed_product,
                    sparse_mul, sparse_add, sparse_scale, all_spaces, coordinates,
                    invariant, gl_maps, find_isomorphism, verify_isomorphism,
                    classification, norm_composition_defect, zero_divisor, lattice_data)
from .tables import TABLE_IMAGES, TABLE1_ROWS, decode, rows

OBSERVATIONS = {}


def signed_map_element(a,mapping):
    out = {}
    for i,x in a.items():
        label = mapping[i+1]
        out[abs(label)-1] = x*(1 if label>0 else -1)
    return out


def power_label(x,k):
    result = 1
    for _ in range(k):
        result = signed_product(result,x)
    return result


def algebra_inverse(a):
    n = sum(x*x for x in a.values())
    return {i:F(x if i==0 else -x,n) for i,x in a.items()}


class PaperVerification(unittest.TestCase):
    def test_01_01_intro_embeddings_and_standard_sedenion_octonion_composition(self):
        for dimension in (1,2,4,8,16):
            for i,j in product(range(dimension),repeat=2):
                self.assertEqual(CK.basis(i+1,j+1),
                                 independent_sign(i,j,dimension)*((i ^ j)+1))
        standard = support_mask(REPS['S'])
        contained = [name for mask,(name,_,_) in classification().items()
                     if mask.bit_count()==8 and mask & standard == mask]
        self.assertEqual(Counter(contained),{'O':8,'O_quasi':7})

    def test_02_01_notation_all_twists_signed_basis_and_independent_doubling(self):
        for i,j in product(range(32),repeat=2):
            expected = independent_sign(i,j)
            self.assertEqual(CK.twist(i,j),expected)
            for s,t in product((-1,1),repeat=2):
                self.assertEqual(CK.basis(s*(i+1),t*(j+1)),s*t*expected*((i ^ j)+1))
        self.assertEqual(decode('-0'),-1)
        self.assertEqual(decode('0'),1)

    def test_02_02_full_loop_latin_square_identity_inverse_and_nonassociativity(self):
        expected = list(range(64))
        for x in range(64):
            self.assertEqual(sorted(TABLE[x].tolist()),expected)
            self.assertEqual(sorted(TABLE[:,x].tolist()),expected)
            self.assertEqual(TABLE[x,0],x)
            self.assertEqual(TABLE[0,x],x)
            self.assertEqual(TABLE[x,INVERSES[x]],0)
            self.assertEqual(TABLE[INVERSES[x],x],0)
        self.assertFalse(np.array_equal(TABLE,TABLE.T))
        self.assertNotEqual(signed_product(signed_product(2,3),5),signed_product(2,signed_product(3,5)))
        self.assertEqual([ELEMENTS[x] for x in range(64) if x!=0 and TABLE[x,x]==0],[-1])

    def test_02_03_table_1_selected_rows_and_table_2_full_transcription(self):
        for i,raw in TABLE1_ROWS.items():
            self.assertEqual(len(raw.split()),32)
            for j,token in enumerate(raw.split()):
                self.assertEqual(decode(token),CK.basis(i+1,j+1))
        support,raw = TABLE_IMAGES[2]
        for i,row in enumerate(rows(raw)):
            self.assertEqual(len(row),16)
            for j,label in enumerate(row):
                self.assertEqual(label,CK.basis(support[i]+1,support[j]+1))

    def test_03_00_01_complete_proper_nontrivial_subloop_counts(self):
        spaces = all_spaces()
        self.assertEqual(len(spaces),374)
        counts = Counter(2*m.bit_count() for m in spaces if m.bit_count()<32)
        self.assertEqual(counts,{2:1,4:31,8:155,16:155,32:31})
        self.assertEqual(sum(counts.values()),373)
        for mask in spaces:
            u = indices(mask)
            self.assertEqual(u[0],0)
            self.assertTrue(all(mask >> (i ^ j) & 1 for i,j in product(u,repeat=2)))
        OBSERVATIONS['proper_nontrivial_counts_by_loop_order'] = dict(counts)

    def test_03_00_02_all_subloops_normal_by_full_inner_generators(self):
        # T_x=R_x^-1 L_x; L_xy^-1 L_x L_y; R_yx^-1 R_x R_y.
        # Every generator fixes each ABSOLUTE basis index, hence every full
        # signed-support subloop, and fixes the singleton identity separately.
        linv = np.argsort(TABLE,axis=1)
        rinv = np.argsort(TABLE.T,axis=1)
        absolute = np.arange(64) % 32
        checked = 0
        for x in range(64):
            permutation = rinv[x,TABLE[x]]
            np.testing.assert_array_equal(permutation % 32,absolute)
            self.assertEqual(permutation[0],0)
            checked += 1
            for y in range(64):
                left = linv[TABLE[x,y],TABLE[x,TABLE[y]]]
                right = rinv[TABLE[y,x],TABLE[TABLE[:,y],x]]
                for permutation in (left,right):
                    np.testing.assert_array_equal(permutation % 32,absolute)
                    self.assertEqual(permutation[0],0)
                    checked += 1
        OBSERVATIONS['normality_inner_generators_checked'] = checked

    def test_03_01_01_four_sedenion_representatives_and_class_sizes(self):
        classes = classification()
        counts = Counter(name for mask,(name,_,_) in classes.items() if mask.bit_count()==16)
        self.assertEqual(counts,{'S':16,'S_alpha':7,'S_beta':7,'S_gamma':1})
        self.assertEqual(len(gl_maps(4)),20160)
        signatures = [invariant(support_mask(REPS[name])) for name in counts]
        self.assertEqual(len(set(signatures)),4)
        for name in counts:
            representative = support_mask(REPS[name])
            self.assertEqual(representative.bit_count(),16)
            self.assertEqual(classes[representative][0],name)
            for x,y in product(labels(representative),repeat=2):
                self.assertIn(CK.basis(x,y),labels(representative))
        OBSERVATIONS['sedenion_class_counts'] = dict(counts)
        OBSERVATIONS['sedenion_distinguishing_invariants'] = {
            name:invariant(support_mask(REPS[name])) for name in counts}

    def test_03_01_02_explicit_isomorphism_certificates_all_31(self):
        for mask,(name,mapping,_) in classification().items():
            if mask.bit_count()==16:
                self.assertTrue(verify_isomorphism(support_mask(REPS[name]),mask,mapping))

    def test_03_01_03_tables_3_4_5_complete_image_transcriptions(self):
        for number in (3,4,5):
            support,raw = TABLE_IMAGES[number]
            self.assertEqual(len(rows(raw)),16)
            for i,row in enumerate(rows(raw)):
                self.assertEqual(len(row),16)
                for j,label in enumerate(row):
                    self.assertEqual(label,CK.basis(support[i]+1,support[j]+1))

    def test_03_01_04_tables_3_and_4_caption_exchange_regression(self):
        self.assertEqual(TABLE_IMAGES[3][0],REPS['S_beta'])
        self.assertEqual(TABLE_IMAGES[4][0],REPS['S_alpha'])
        self.assertNotEqual(TABLE_IMAGES[3][0],REPS['S_alpha'])
        self.assertNotEqual(TABLE_IMAGES[4][0],REPS['S_beta'])
        OBSERVATIONS['printed_table_caption_discrepancy'] = {
            '3':{'image':'S_beta','caption':'S_alpha'},
            '4':{'image':'S_alpha','caption':'S_beta'}}

    def test_03_02_01_octonion_class_counts_and_all_155_isomorphisms(self):
        counts = Counter()
        for mask,(name,mapping,_) in classification().items():
            if mask.bit_count()==8:
                counts[name] += 1
                self.assertTrue(verify_isomorphism(support_mask(REPS[name]),mask,mapping))
        self.assertEqual(counts,{'O':50,'O_quasi':105})
        self.assertNotEqual(invariant(support_mask(REPS['O'])),invariant(support_mask(REPS['O_quasi'])))
        OBSERVATIONS['octonion_class_counts'] = dict(counts)

    def test_03_02_02_symbolic_composition_all_50_and_zero_divisors_all_105(self):
        witnesses = {}
        for mask,(name,_,_) in classification().items():
            if mask.bit_count()!=8:
                continue
            defect = norm_composition_defect(mask)
            if name=='O':
                self.assertEqual(defect,{})
            else:
                self.assertTrue(defect)
                a,b = zero_divisor(mask)
                self.assertTrue(a and b)
                self.assertEqual(sparse_mul(a,b),{})
                self.assertEqual(sparse_mul(a,b,reference=True),{})
                witnesses[str(mask)] = {'a':a,'b':b}
        self.assertEqual(len(witnesses),105)
        OBSERVATIONS['quasi_octonion_zero_divisors'] = witnesses

    def test_03_03_groups_q8_c4_c2_exhaustive_associativity_and_relations(self):
        counts = Counter()
        for mask,(name,mapping,_) in classification().items():
            if mask.bit_count()>4:
                continue
            counts[name] += 1
            self.assertTrue(verify_isomorphism(support_mask(REPS[name]),mask,mapping))
            values = labels(mask)
            for a,b,c in product(values,repeat=3):
                self.assertEqual(signed_product(signed_product(a,b),c),signed_product(a,signed_product(b,c)))
            if name=='Q8':
                i,j = [x+1 for x in indices(mask)[1:3]]
                self.assertEqual(signed_product(i,j),-signed_product(j,i))
                self.assertEqual(signed_product(i,i),-1)
                self.assertEqual(signed_product(j,j),-1)
            if name=='C4':
                generator = indices(mask)[1]+1
                self.assertEqual(set(power_label(generator,k) for k in range(4)),set(values))
        self.assertEqual(counts,{'Q8':155,'C4':31,'C2':1})

    def test_03_04_nine_classes_and_linear_extension_to_basis_subalgebras(self):
        self.assertEqual(Counter(x[0] for x in classification().values()),
                         {'S':16,'S_alpha':7,'S_beta':7,'S_gamma':1,'O':50,'O_quasi':105,'Q8':155,'C4':31,'C2':1})
        for mask,(name,mapping,_) in classification().items():
            source = REPS[name]
            a = {i:(j%3)-1 for j,i in enumerate(source) if j%3!=1}
            b = {i:2-(j%5) for j,i in enumerate(source) if j%5!=2}
            self.assertEqual(signed_map_element(sparse_mul(a,b),mapping),
                             sparse_mul(signed_map_element(a,mapping),signed_map_element(b,mapping)))

    def test_03_05_01_maximal_subloop_compositions_and_table_6(self):
        all_counts = Counter()
        representative_counts = {}
        for mask,(name,_,_) in classification().items():
            if mask.bit_count()!=16:
                continue
            contained = Counter(kind for sub,(kind,_,_) in classification().items()
                                if sub.bit_count()==8 and sub & mask==sub)
            pair = (contained['O'],contained['O_quasi'])
            self.assertEqual(sum(pair),15)
            all_counts[pair] += 1
            if mask==support_mask(REPS[name]):
                representative_counts[name] = pair
        self.assertEqual(all_counts,{(8,7):17,(2,13):7,(0,15):7})
        self.assertEqual(representative_counts,{'S':(8,7),'S_alpha':(2,13),'S_beta':(0,15),'S_gamma':(8,7)})
        OBSERVATIONS['maximal_octonion_compositions'] = {str(k):v for k,v in all_counts.items()}

    def test_03_05_02_all_31_coordinate_sedenion_algebras_have_zero_divisors(self):
        witnesses = {}
        for mask,(name,_,_) in classification().items():
            if mask.bit_count()==16:
                a,b = zero_divisor(mask)
                self.assertTrue(a and b)
                self.assertEqual(sparse_mul(a,b),{})
                self.assertEqual(sparse_mul(a,b,reference=True),{})
                self.assertNotEqual(sparse_mul(sparse_mul(a,a),b),{})
                self.assertEqual(sparse_mul(a,sparse_mul(a,b)),{})
                witnesses[str(mask)] = {'class':name,'a':a,'b':b}
        self.assertEqual(len(witnesses),31)
        OBSERVATIONS['sedenion_zero_divisors'] = witnesses

    def test_03_06_01_inverse_property_all_signed_pairs(self):
        for x,y in product(range(64),repeat=2):
            self.assertEqual(TABLE[INVERSES[x],TABLE[x,y]],y)
            self.assertEqual(TABLE[TABLE[y,x],INVERSES[x]],y)

    def test_03_06_02_alternative_property_all_signed_pairs(self):
        for x,y in product(range(64),repeat=2):
            self.assertEqual(TABLE[x,TABLE[x,y]],TABLE[TABLE[x,x],y])
            self.assertEqual(TABLE[TABLE[x,y],y],TABLE[x,TABLE[y,y]])

    def test_03_06_03_flexible_law_all_signed_pairs(self):
        for x,y in product(range(64),repeat=2):
            self.assertEqual(TABLE[x,TABLE[y,x]],TABLE[TABLE[x,y],x])

    def test_03_06_04_c_loop_property_all_signed_triples(self):
        z = np.arange(64)
        for x,y in product(range(64),repeat=2):
            np.testing.assert_array_equal(TABLE[x,TABLE[y,TABLE[y,z]]],TABLE[TABLE[TABLE[x,y],y],z])

    def test_03_06_05_power_associativity_all_cyclic_powers(self):
        for x in ELEMENTS:
            self.assertEqual(power_label(x,4),1)
            for a,b in product(range(8),repeat=2):
                self.assertEqual(signed_product(power_label(x,a),power_label(x,b)),power_label(x,a+b))

    def test_03_06_06_weak_inverse_all_signed_pairs(self):
        for x,y in product(range(64),repeat=2):
            self.assertEqual(TABLE[x,INVERSES[TABLE[y,x]]],INVERSES[y])

    def test_03_06_07_anti_automorphic_inverse_all_signed_pairs(self):
        for x,y in product(range(64),repeat=2):
            self.assertEqual(INVERSES[TABLE[x,y]],TABLE[INVERSES[y],INVERSES[x]])

    def test_03_06_08_moufang_positive_and_negative_classes(self):
        for mask,(name,_,signature) in classification().items():
            fails = signature[1]
            if name in ('O','Q8','C4','C2'):
                self.assertEqual(fails,0)
            else:
                self.assertGreater(fails,0)
        witnesses = {}
        for name in ('O_quasi','S','S_alpha','S_beta','S_gamma'):
            values = tuple(i+1 for i in REPS[name])
            for x,y,z in product(values,repeat=3):
                lhs = signed_product(signed_product(x,y),signed_product(z,x))
                rhs = signed_product(x,signed_product(signed_product(y,z),x))
                if lhs!=rhs:
                    witnesses[name] = {'x':x,'y':y,'z':z,'lhs':lhs,'rhs':rhs}
                    break
            self.assertIn(name,witnesses)
        OBSERVATIONS['moufang_counterexamples_signed_one_based_labels'] = witnesses
        # Direct signed checks in ordinary octonions, independent of invariant code.
        for x,y,z in product(labels(support_mask(REPS['O'])),repeat=3):
            self.assertEqual(signed_product(signed_product(x,y),signed_product(z,x)),
                             signed_product(x,signed_product(signed_product(y,z),x)))

    def test_03_06_09_dense_algebra_flexibility_and_power_checks(self):
        rng = np.random.default_rng(9072047)
        for _ in range(16):
            a,b = ({i:int(x) for i,x in enumerate(rng.integers(-2,3,size=32)) if x} for _ in range(2))
            self.assertEqual(sparse_mul(a,sparse_mul(b,a)),sparse_mul(sparse_mul(a,b),a))
            self.assertEqual(sparse_mul(sparse_mul(a,a),a),sparse_mul(a,sparse_mul(a,a)))
            self.assertEqual(sparse_mul(sparse_mul(a,a),sparse_mul(a,a)),sparse_mul(sparse_mul(sparse_mul(a,a),a),a))

    def test_03_06_10_algebra_inverse_alternative_and_c_loop_negative_controls(self):
        mask = support_mask(REPS['S'])
        a,b = zero_divisor(mask)
        self.assertEqual(sparse_mul(algebra_inverse(a),sparse_mul(a,b)),{})
        self.assertNotEqual(b,{})
        self.assertNotEqual(sparse_mul(a,sparse_mul(a,b)),sparse_mul(sparse_mul(a,a),b))
        # C identity with x=1,y=a,z=b reduces to the same alternativity failure.
        self.assertNotEqual(sparse_mul({0:1},sparse_mul(a,sparse_mul(a,b))),
                            sparse_mul(sparse_mul(sparse_mul({0:1},a),a),b))
        # Add a scalar to the zero-divisor partner. All products now inverted
        # are nonzero, but both algebra AAIP and WIP still fail.
        q = sparse_add({0:1},b)
        aq = sparse_mul(a,q)
        qa = sparse_mul(q,a)
        self.assertTrue(aq and qa)
        self.assertNotEqual(algebra_inverse(aq),sparse_mul(algebra_inverse(q),algebra_inverse(a)))
        self.assertNotEqual(sparse_mul(a,algebra_inverse(qa)),algebra_inverse(q))
        OBSERVATIONS['algebra_inverse_counterexample'] = {'a':a,'q':q,'aq':aq,'qa':qa}

    def test_03_06_11_symbolic_quadratic_relation_and_flexibility_all_real_vectors(self):
        # Coefficients of a^2-2*a0*a+N(a)*1, for all 528 quadratic monomials.
        quadratic = Counter()
        for i in range(32):
            for j in range(i,32):
                label = CK.basis(i+1,j+1)
                quadratic[i,j,abs(label)-1] += 1 if label>0 else -1
                if i!=j:
                    label = CK.basis(j+1,i+1)
                    quadratic[i,j,abs(label)-1] += 1 if label>0 else -1
            quadratic[0,i,i] -= 2
            quadratic[i,i,0] += 1
        self.assertFalse({k:c for k,c in quadratic.items() if c})
        # Coefficients of a*(b*a)-(a*b)*a: monomials a_i*a_p*b_j.
        flexibility = Counter()
        for i,p,j in product(range(32),repeat=3):
            coefficient = int(SIGNS[j,p]*SIGNS[i,j ^ p]-SIGNS[i,j]*SIGNS[i ^ j,p])
            key = (min(i,p),max(i,p),j,i ^ j ^ p)
            flexibility[key] += coefficient
        self.assertFalse({k:c for k,c in flexibility.items() if c})
        OBSERVATIONS['symbolic_real_algebra_checks'] = {
            'quadratic_input_monomials':528,'flexibility_ordered_terms':32768,
            'nonzero_residual_coefficients':0}

    def test_03_07_01_full_lattice_covers_and_modular_law_exhaustively(self):
        spaces,meet,join,covers = lattice_data()
        self.assertEqual(len(spaces),375)
        self.assertEqual(len(covers),2078)
        checks = 0
        for a,mask_a in enumerate(spaces):
            for c,mask_c in enumerate(spaces):
                if mask_a & mask_c==mask_a:
                    # All choices of b, checked as a vector operation.
                    lhs = join[a,meet[:,c]]
                    rhs = meet[join[a,:],c]
                    np.testing.assert_array_equal(lhs,rhs)
                    checks += len(spaces)
        OBSERVATIONS['lattice'] = {'nodes':375,'covers':2078,'modular_triples_checked':checks}

    def test_03_07_02_figure_1_block_edges_from_actual_covers(self):
        spaces,_,_,covers = lattice_data()
        def kind(mask):
            return 'trivial' if mask==0 else 'T' if mask.bit_count()==32 else classification()[mask][0]
        edges = Counter((kind(spaces[upper]),kind(spaces[lower])) for lower,upper in covers)
        expected = {('T',x) for x in ('S','S_alpha','S_beta','S_gamma')}
        expected |= {(x,'O_quasi') for x in ('S','S_alpha','S_beta','S_gamma')}
        expected |= {(x,'O') for x in ('S','S_alpha','S_gamma')}
        expected |= {('O','Q8'),('O_quasi','Q8'),('Q8','C4'),('C4','C2'),('C2','trivial')}
        self.assertEqual(set(edges),expected)
        OBSERVATIONS['class_cover_edge_counts'] = {a+' -> '+b:n for (a,b),n in sorted(edges.items())}

    def test_03_07_03_class_block_graph_is_not_itself_a_lattice(self):
        # The paper calls Figure 1 a block-form lattice. Aggregating by class
        # loses the meet operation: S and S_gamma have two incomparable maximal
        # common lower classes, O and O_quasi.
        spaces,_,_,covers = lattice_data()
        def kind(mask):
            return 'trivial' if not mask else 'T' if mask.bit_count()==32 else classification()[mask][0]
        below = {name:{name} for name in tuple(REPS)+('T','trivial')}
        for lower,upper in covers:
            below[kind(spaces[upper])].add(kind(spaces[lower]))
        for _ in range(7):
            for name in below:
                below[name] |= set().union(*(below[x] for x in tuple(below[name])))
        common = below['S'] & below['S_gamma']
        maximal = {x for x in common if not any(x!=y and x in below[y] for y in common)}
        self.assertEqual(maximal,{'O','O_quasi'})
        OBSERVATIONS['class_graph_lattice_caution'] = {'pair':['S','S_gamma'],
                                                      'maximal_common_lower_classes':sorted(maximal)}

    def test_04_01_summary_all_octonion_types_have_same_small_subloop_counts(self):
        for mask,(name,_,_) in classification().items():
            if mask.bit_count()==8:
                counts = Counter(kind for sub,(kind,_,_) in classification().items()
                                 if sub.bit_count()<8 and sub & mask==sub)
                self.assertEqual(counts,{'Q8':7,'C4':7,'C2':1})

    def test_04_02_basis_subalgebras_do_not_enumerate_all_real_subalgebras(self):
        # span_R{1,e1+e2} is a 2-D complex subalgebra outside coordinate supports.
        v = {1:1,2:1}
        self.assertEqual(sparse_mul(v,v),{0:-2})
        a,b = sparse_add({0:2},v),sparse_add({0:-1},sparse_scale(v,3))
        expected = sparse_add({0:-8},sparse_scale(v,5))
        self.assertEqual(sparse_mul(a,b),expected)
        self.assertEqual(len(set(v)|{0}),3) # support span is 3-D, actual subalgebra is 2-D


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

    def test_02_multiply_all_4096_signed_loop_products(self):
        eye = np.eye(32)
        vectors = np.concatenate((eye,-eye))
        a,b = np.repeat(vectors,64,axis=0),np.tile(vectors,(64,1))
        expected = vectors[TABLE.reshape(-1)]
        np.testing.assert_array_equal(replay_product(a,b),expected)

    def test_03_multiply_dense_subalgebra_isomorphisms_all_373(self):
        left=[];right=[];expected=[]
        for mask,(name,mapping,_) in classification().items():
            source = REPS[name]
            a={i:(j%3)-1 for j,i in enumerate(source) if j%3!=1}
            b={i:2-(j%5) for j,i in enumerate(source) if j%5!=2}
            a,b=signed_map_element(a,mapping),signed_map_element(b,mapping)
            dense=lambda x:[x.get(i,0) for i in range(32)]
            left.append(dense(a));right.append(dense(b));expected.append(dense(sparse_mul(a,b,True)))
        np.testing.assert_array_equal(replay_product(np.array(left,dtype=float),np.array(right,dtype=float)),expected)

    def test_03_multiply_all_136_zero_divisor_witnesses(self):
        pairs=[zero_divisor(mask) for mask,(name,_,_) in classification().items() if name in ('O_quasi','S','S_alpha','S_beta','S_gamma')]
        self.assertEqual(len(pairs),136)
        a=np.array([[x.get(i,0) for i in range(32)] for x,y in pairs],dtype=float)
        b=np.array([[y.get(i,0) for i in range(32)] for x,y in pairs],dtype=float)
        np.testing.assert_array_equal(replay_product(a,b),np.zeros((136,32)))


try:
    import pytest
except ImportError:
    pass
else:
    PaperVerification=pytest.mark.cpu(PaperVerification)
    GPUVerification=pytest.mark.cpu(GPUVerification)
