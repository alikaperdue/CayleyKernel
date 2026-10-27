"""Paper-ordered exact tests. GPU tests run whenever CUDA is available."""
from collections import Counter
from fractions import Fraction
from itertools import combinations, product
import random
import unittest

import CayleyKernel as CK
from tests.papers.type_replay import replay_product
from .emanation import (e,add,scale,mul,recursive_sign,assessor,labels,cell,direct_cell,
                        high_bits,paint_cell,recipe_rules,table,counts,trip_count,
                        whorf_count,box_kites,trijoin)

SEED=7040112
OBSERVATIONS={}
CERTIFICATES={}


def exact_pair(r,c,S,N):
    value=cell(r,c,S,N)
    if not value:raise ValueError('blank cell')
    a=assessor(r,S,N);b=assessor(c,S,N,1 if value<0 else -1)
    if mul(a,b,N) or mul(b,a,N):raise AssertionError('not a two-way DMZ')
    return {'N':N,'S':S,'r':r,'c':c,'entry':value,'a':a,'b':b,'ab':{},'ba':{}}


def valid_pairs(S,N):
    ll=labels(S,N)
    return product(ll,repeat=2)


def dot(a,b):
    return sum(x*b.get(i,0) for i,x in a.items())


def check_partition(test,S,N):
    boxes=box_kites(S,N)
    cover=Counter(tuple(sorted((a,b))) for box in boxes for a,b in combinations(box,2) if a^b!=S)
    edges={tuple(sorted((r,c))) for r,c in valid_pairs(S,N) if cell(r,c,S,N)}
    test.assertEqual(set(cover),edges)
    test.assertEqual(set(cover.values()),{1})
    test.assertEqual(24*len(boxes),2*len(edges))
    return boxes


class PaperVerification(unittest.TestCase):
    def test_01_01_xor_signs_and_signed_labels_against_doubling(self):
        checked=0
        for N in range(1,9):
            d=1<<N
            for i,j in product(range(d),repeat=2):
                sign=recursive_sign(i,j,d)
                self.assertEqual(CK.twist(i,j),sign)
                self.assertEqual(CK.basis(i+1,j+1),sign*((i^j)+1))
                if N<=5:
                    for s,t in product((-1,1),repeat=2):
                        self.assertEqual(CK.basis(s*(i+1),t*(j+1)),s*t*sign*((i^j)+1))
                checked+=1
        OBSERVATIONS['basis_pairs_checked']=checked

    def test_01_02_rules_zero_one_two_and_cyclic_positive_trips(self):
        count=0
        def cpo(a,b,c):
            return all(CK.basis(x+1,y+1)==z+1 for x,y,z in ((a,b,c),(b,c,a),(c,a,b)))
        for N in range(3,8):
            G=1<<(N-1)
            for u in range(1,G):self.assertTrue(cpo(u,G,G+u))
            for a,b in combinations(range(1,G),2):
                c=a^b
                if c<b:continue
                if CK.twist(a,b)<0:b,c=c,b
                self.assertTrue(cpo(a,b,c))
                for trip in ((a,G+c,G+b),(G+c,b,G+a),(G+b,G+a,c)):
                    self.assertTrue(cpo(*trip));count+=1
        OBSERVATIONS['rule_two_trip_checks']=count

    def test_01_03_assessor_diagonals_and_unsigned_viziers(self):
        for N in range(4,8):
            G=1<<(N-1)
            for S in range(1,G):
                for r in labels(S,N):
                    R=G+(r^S);c=r^S;C=G+r
                    self.assertEqual((r^c,R^C),(S,S))
                    self.assertEqual((R^c,C^r),(G,G))
                    self.assertEqual((R^r,c^C),(G+S,G+S))
                    if S<9:
                        plus=assessor(r,S,N);minus=assessor(r,S,N,-1)
                        self.assertNotEqual(mul(plus,minus,N),{})
                        self.assertEqual(mul(plus,plus,N),{0:-2})

    def test_01_04_seven_sedenion_box_kites_and_edge_signs(self):
        rows=[]
        for S in range(1,8):
            boxes=check_partition(self,S,4)
            self.assertEqual(len(boxes),1)
            t=table(S,4)
            positive=sum(x>0 for row in t for x in row);negative=sum(x<0 for row in t for x in row)
            self.assertEqual((positive,negative),(12,12))
            triangles=[t for t in combinations(boxes[0],3) if all(cell(a,b,S,4) for a,b in combinations(t,2))]
            sails=[t for t in triangles if t[0]^t[1]^t[2]==0]
            red=sum(all(cell(a,b,S,4)>0 for a,b in combinations(t,2)) for t in triangles)
            self.assertEqual((len(triangles),len(sails),red),(8,4,2))
            rows.append({'S':S,'vertices':boxes[0],'marked_directed_cells':negative,'unmarked_directed_cells':positive,
                         'triangular_faces':8,'xor_sails':sails,'all_negative_edge_faces':2})
        OBSERVATIONS['sedenion_box_kites']=rows

    def test_01_05_emanation_table_construction_and_actual_two_way_products(self):
        checked=0;witnesses=[]
        for N in (4,5,6):
            G=1<<(N-1)
            for S in range(1,G):
                ll=labels(S,N)
                self.assertEqual(len(ll),G-2)
                self.assertEqual(set(ll),set(range(1,G))-{S})
                self.assertTrue(all(a^b==S for a,b in zip(ll,reversed(ll))))
                for r,c in product(ll,repeat=2):
                    actual=direct_cell(r,c,S,N)
                    self.assertEqual(cell(r,c,S,N),actual)
                    self.assertEqual(cell(r,c,S,N),cell(c,r,S,N))
                    if actual:
                        pair=exact_pair(r,c,S,N)
                        # Simultaneous slope reversal supplies the second DMZ.
                        aa=assessor(r,S,N,-1);bb=assessor(c,S,N,-(1 if actual<0 else -1))
                        self.assertFalse(mul(aa,bb,N));self.assertFalse(mul(bb,aa,N))
                        if r<c and len(witnesses)<12:witnesses.append(pair)
                    checked+=1
        CERTIFICATES['primitive_dmz_witnesses']=witnesses
        OBSERVATIONS['cells_checked_with_full_basiss']=checked

    def test_01_06_trip_count_two_step_and_number_hub(self):
        rows=[]
        for N in range(4,8):
            G=1<<(N-1)
            for S in list(range(1,9))+[1<<k for k in range(4,N-1)]:
                if S>=G:continue
                actual=counts(S,N)
                self.assertEqual(actual['filled_directed_cells'],24*trip_count(N-2))
                rows.append(actual)
            S=G//2
            ll=labels(S,N);half=ll[:len(ll)//2]
            for r,c in product(half,repeat=2):
                self.assertEqual(abs(cell(r,c,S,N)),r^c)
        OBSERVATIONS['full_tables_and_number_hubs']=rows

    def test_02_01_theorem_11_central_embedding_preserves_marks_and_blanks(self):
        rows=[]
        for S,N in ((9,5),(15,5),(15,6),(17,6),(24,6),(25,6),(31,6),(48,7),(57,7),(73,8)):
            g=1<<(N-1);checked=0
            for r,c in valid_pairs(S,N):
                self.assertEqual(cell(r,c,S,N),cell(g+r,g+c,S,N+1));checked+=1
            rows.append({'S':S,'old_N':N,'cells_including_blanks_and_marks':checked})
        OBSERVATIONS['theorem_11_central_embedding']=rows

    def test_02_02_theorem_11_label_lines_become_dmz_contents(self):
        rows=[]
        for S,N in ((9,5),(15,5),(25,6),(57,7)):
            g=1<<(N-1)
            for u in labels(S,N):
                self.assertEqual(abs(cell(g,g+u,S,N+1)),u)
                self.assertEqual(abs(cell(g+S,g+u,S,N+1)),u^S)
                self.assertEqual(cell(g+u,g+u,S,N+1),0)
            rows.append({'S':S,'old_N':N,'label_count':len(labels(S,N))})
        OBSERVATIONS['label_content_involution']=rows

    def test_02_03_theorem_12_four_corners_zero_padding(self):
        for S,N in ((9,5),(15,5),(25,6),(31,6),(57,7)):
            for r,c in valid_pairs(S,N):self.assertEqual(cell(r,c,S,N),cell(r,c,S,N+1))
        OBSERVATIONS['four_corners_scope']='All prior labels in five specified transitions, including both edge marks and blanks.'

    def test_02_04_theorem_13_french_window_shutters_and_diagonals(self):
        rows=[]
        for S,N in ((9,5),(15,5),(25,6),(31,6),(48,7),(57,7)):
            g=1<<(N-1);categories=Counter()
            for r,c in valid_pairs(S,N):
                old=cell(r,c,S,N)
                if old:
                    wanted=(1 if old>0 else -1)*(g+abs(old));categories['old_filled']+=1
                elif r==c:wanted=-g;categories['main_diagonal']+=1
                elif r^c==S:wanted=g+S;categories['strut_diagonal']+=1
                else:wanted=0;categories['rule_hidden']+=1
                self.assertEqual(cell(g+r,c,S,N+1),wanted)
                self.assertEqual(cell(r,g+c,S,N+1),wanted)
            rows.append({'S':S,'old_N':N,'categories':dict(categories)})
        OBSERVATIONS['theorem_13_shutters']=rows

    def test_02_05_theorem_13_label_line_extensions_reverse_marks(self):
        for S,N in ((9,5),(15,5),(25,6),(57,7)):
            g=1<<(N-1)
            for u in labels(S,N):
                a,b=cell(g,u,S,N+1),cell(g,g+u,S,N+1)
                self.assertTrue(a and b)
                self.assertNotEqual(a<0,b<0)
                self.assertEqual((abs(a),abs(b)),(g+u,u))

    def test_03_01_whorfian_formula_and_printed_sequence(self):
        printed=[0,3,19,91,395,1643,6699]
        self.assertEqual([whorf_count(N) for N in range(4,11)],printed)
        rows=[]
        # N=4 is a formal sequence term; S>8 is not admissible there.
        for N in range(5,11):
            actual=counts(15,N)
            self.assertEqual(actual['filled_directed_cells'],24*whorf_count(N))
            self.assertEqual(whorf_count(N)+4**(N-4),trip_count(N-2))
            rows.append(actual)
        OBSERVATIONS['whorfian_counts']=rows

    def test_03_02_muntin_and_diagonal_contributions_units(self):
        rows=[]
        for N in range(5,9):
            G=1<<(N-1);S=15;ll=labels(S,N);mu=(1<<(N-4))-1
            nsl={r for r in ll if r%8 in (0,7)}
            self.assertEqual(len(nsl),2*mu)
            omega=sum(bool(cell(r,c,S,N)) for r,c in product(ll,repeat=2) if (r in nsl)!=(c in nsl))
            self.assertEqual(omega,24*mu*(mu+1))
            total=counts(S,N)['filled_directed_cells'];delta=total-omega
            self.assertEqual(delta,24*trip_count(N-3))
            self.assertEqual((omega+delta)//24,whorf_count(N))
            rows.append({'N':N,'mu':mu,'omega_cells':omega,'delta_cells':delta,'box_kites':(omega+delta)//24})
        OBSERVATIONS['omega_delta_cell_vs_box_units']=rows

    def test_03_03_explicit_box_kites_partition_edges(self):
        rows=[]
        for S,N in ((1,5),(8,5),(9,5),(15,6),(15,7),(17,6),(25,6),(31,6),(57,7)):
            boxes=check_partition(self,S,N)
            rows.append({'S':S,'N':N,'box_kites':len(boxes),'every_edge_in_exactly_one_box':True})
            if (S,N)==(57,7):CERTIFICATES['S57_N7_boxes']=boxes
        OBSERVATIONS['explicit_box_partition']=rows

    def test_03_04_theorem_14_singleton_high_bit_and_tandem_shifts(self):
        rows=[]
        for N in range(5,10):
            g=1<<(N-2);G=2*g
            for s in range(1,9):
                S=g+s
                if S>=G:continue
                total=0
                for r,c in valid_pairs(S,N):
                    expected=r!=c and r^c!=S and any(x in (g,s) for x in (r,c,r^c))
                    self.assertEqual(bool(cell(r,c,S,N)),expected);total+=expected
                self.assertEqual(total,6*(G-4))
                rows.append({'N':N,'S':S,'cells':total,'box_kites':total//24})
        OBSERVATIONS['singleton_tandem_shifts']=rows

    def test_03_05_theorem_14_recursive_harmonics(self):
        rows=[]
        for S in (9,15,17,24,33,40):
            start=S.bit_length()+1
            for N in range(start,start+3):
                ll=labels(S,N);bad=[]
                for r,c in product(ll,repeat=2):
                    if paint_cell(r,c,S,N)!=bool(cell(r,c,S,N)):bad.append((r,c))
                self.assertEqual(bad,[])
                rows.append(counts(S,N))
        OBSERVATIONS['singleton_recursive_harmonics']=rows

    def test_03_06_density_limit_and_finite_degree_histograms(self):
        rows=[]
        for N in range(5,15):
            G=1<<(N-1)
            density=Fraction(24*whorf_count(N),(G-2)**2)
            self.assertLess(density,Fraction(5,8))
            rows.append({'N':N,'filled_fraction':str(density),'gap_to_5_over_8':str(Fraction(5,8)-density)})
        self.assertLess(Fraction(rows[-1]['gap_to_5_over_8']),Fraction(1,1000))
        hist=[]
        for N in range(5,9):
            ll=labels(15,N)
            degrees=Counter(sum(bool(cell(r,c,15,N)) for c in ll) for r in ll)
            hist.append({'N':N,'degree_histogram':dict(sorted(degrees.items()))})
        OBSERVATIONS['density_and_degree_histograms']={'densities':rows,'graphs':hist,
             'scope':'Finite graph degrees and a formula asymptotic; no proof of a power-law scale-free network or a fractal dimension.'}

    def test_04_01_chingon_hide_fill_counts(self):
        rows=[]
        for S in range(1,32):
            expected=35 if S<=8 or S==16 else 19 if S<16 else 7 if S<=24 else 23
            actual=counts(S,6)
            self.assertEqual(actual['filled_directed_cells'],24*expected)
            rows.append(actual)
        OBSERVATIONS['chingon_spectrography']=rows

    def test_04_02_theorem_15_restricted_two_bit_off_switch(self):
        rows=[]
        for N,S,L in ((5,1,8),(6,9,16),(7,9,32),(7,17,32)):
            before=off=0
            for r,c in product(range(1,L),repeat=2):
                if r==S or c==S:continue
                if cell(r,c,S,N):
                    before+=1
                    self.assertEqual(cell(r,c,S+L,N),0);off+=1
            self.assertGreater(before,0)
            rows.append({'N':N,'S':S,'new_bit':L,'old_labels_strictly_below_bit':True,'before':before,'turned_off':off})
        OBSERVATIONS['two_bit_off_switch_restricted']=rows

    def test_04_03_theorem_15_unrestricted_wording_counterexample(self):
        self.assertEqual(cell(2,8,1,5),10)
        self.assertEqual(cell(2,8,9,5),10)
        old=exact_pair(2,8,1,5);new=exact_pair(2,8,9,5)
        self.assertFalse(mul(old['a'],old['b'],5));self.assertFalse(mul(new['a'],new['b'],5))
        CERTIFICATES['theorem_15_range_counterexample']={'before':old,'after':new}
        OBSERVATIONS['theorem_15_range_qualification']='R=2,C=8 remains a DMZ when S=1 becomes9 in dimension32. Proof uses addition of bit8 to upper indices, which needs labels below8; XOR toggles that bit for C=8 instead.'

    def test_04_04_hidden_box_octonion_clones(self):
        records=[]
        for low in ((2,4,6),(2,5,7),(3,4,7),(3,5,6)):
            indices=[0,25]+list(low)+[16+(r^9) for r in low]
            self.assertEqual(len(set(indices)),8)
            self.assertTrue(all((i^j) in indices for i,j in product(indices,repeat=2)))
            self.assertTrue(all(not cell(i,j,9,5) for i,j in combinations(low,2)))
            # Construct a signed isomorphism from O by generators low[0],low[1],X.
            a,b=low[:2]
            image=[e(0),e(a),e(b),mul(e(a),e(b),5),e(25)]
            image += [mul(image[j],image[4],5) for j in (1,2,3)]
            self.assertEqual({next(iter(v)) for v in image},set(indices))
            for i,j in product(range(8),repeat=2):
                self.assertEqual(mul(image[i],image[j],5),scale(image[i^j],recursive_sign(i,j,8)))
            matrices=[[mul(e(i),e(k),5) for k in indices] for i in indices]
            for i,j,k,l in product(range(8),repeat=4):
                self.assertEqual(dot(matrices[i][k],matrices[j][l])+dot(matrices[j][k],matrices[i][l]),2*int(i==j)*int(k==l))
            records.append({'S':9,'N':5,'low_trip':low,'indices':indices,'octonion_signed_basis_images':image,
                            'full_norm_composition_coefficients_verified':True})
        CERTIFICATES['hidden_octonion_clones']=records
        OBSERVATIONS['hidden_octonion_clone_count']=len(records)

    def test_04_05_theorem_16_hat_trick_restoration(self):
        rows=[]
        for N,S,L,M in ((6,1,8,16),(7,1,8,32),(8,9,16,64)):
            recovered=0
            for r,c in product(range(1,L),repeat=2):
                if r==S or c==S:continue
                if cell(r,c,S,N):
                    self.assertEqual(cell(r,c,S+L,N),0)
                    self.assertTrue(cell(r,c,S+L+M,N));recovered+=1
            self.assertGreater(recovered,0)
            rows.append({'N':N,'S':S,'L':L,'M':M,'restored_old_directed_cells':recovered})
        OBSERVATIONS['hat_trick_examples']=rows

    def test_04_06_rule_priority_and_paper_examples_25_41_57_73(self):
        rows=[]
        expected_first={25:[9,16],41:[9,32,64,73,96,105],57:[25,32],73:[9,64]}
        for S,N in ((25,6),(41,8),(57,7),(73,8)):
            rules=recipe_rules(S,N)
            self.assertEqual(rules[0]['values'],expected_first[S])
            decisions=Counter();overlap=0
            for r,c in valid_pairs(S,N):
                if r==c or r^c==S:continue
                matches=[i for i,rule in enumerate(rules) if set((r,c,r^c))&set(rule['values'])]
                if len(matches)>1:overlap+=1
                decision=bool((matches[0]+1)%2) if matches else not len(rules)%2
                self.assertEqual(decision,bool(cell(r,c,S,N)))
                decisions['rule_'+str(matches[0]+1) if matches else 'default']+=1
            rows.append({'N':N,'S':S,'rules':rules,'cells_matching_more_than_one_rule':overlap,
                         'first_match_distribution':dict(decisions),**counts(S,N)})
        self.assertEqual(counts(57,7)['edge_count_divided_by_12'],63)
        OBSERVATIONS['paper_rule_examples']=rows

    def test_05_01_proposition_inner_skybox_and_613_bit_preparation(self):
        for K in range(9,4097):
            if K&(K-1)==0:continue
            N=K.bit_length()+1
            self.assertLess(1<<(N-2),K);self.assertLess(K,1<<(N-1))
        self.assertEqual(bin(613)[2:],'1001100101')
        self.assertEqual(high_bits(613),[9,6,5])
        self.assertEqual((613).bit_length()+1,11)
        OBSERVATIONS['S613_preparation']={'S':613,'inner_N':11,'dimension':2048,'binary':bin(613)[2:],'high_bit_exponents':high_bits(613)}

    def test_05_02_cookbook_literal_preparation_exhaustive_finite_audit(self):
        rows=[];witnesses=[]
        for N in range(4,9):
            G=1<<(N-1);bad=[];checked=0
            for S in range(1,G):
                mismatches=0;first=None
                for r,c in valid_pairs(S,N):
                    printed=paint_cell(r,c,S,N);actual=bool(cell(r,c,S,N));checked+=1
                    if printed!=actual:
                        mismatches+=1
                        if first is None:first={'N':N,'S':S,'r':r,'c':c,'recipe_filled':printed,'actual_entry':cell(r,c,S,N)}
                if mismatches:
                    bad.append({'S':S,'mismatches':mismatches});witnesses.append(first)
                    self.assertEqual(S%16,0) # observed failure class, not an assumed exclusion
            rows.append({'N':N,'cells_checked':checked,'mismatching_strut_constants':bad})
        self.assertTrue(any(x['S']==48 for x in witnesses))
        OBSERVATIONS['literal_cookbook_audit']=rows
        CERTIFICATES['literal_recipe_mismatch_witnesses']=witnesses
        self.assertTrue(paint_cell(1,2,48,7))
        a=assessor(1,48,7)
        cases=[]
        for slope in (-1,1):
            b=assessor(2,48,7,slope);value=mul(a,b,7)
            self.assertTrue(value)
            cases.append({'a':a,'b':b,'ab':value})
        CERTIFICATES['S48_literal_recipe_nonzero_products']=cases
        self.assertEqual(counts(47,7)['edge_count_divided_by_12'],79)
        self.assertEqual(counts(48,7)['edge_count_divided_by_12'],15)
        OBSERVATIONS['multiple_of_16_band_boundary']={'S47':counts(47,7),'S48':counts(48,7)}

    def test_05_03_proposed_trailing_bit_correction_exhaustive_finite_audit(self):
        rows=[]
        for N in range(4,9):
            G=1<<(N-1);checked=0
            for S in range(1,G):
                for r,c in valid_pairs(S,N):
                    self.assertEqual(paint_cell(r,c,S,N,True),bool(cell(r,c,S,N)));checked+=1
            rows.append({'N':N,'strut_constants':G-1,'cells_checked':checked,'mismatches':0})
        OBSERVATIONS['proposed_corrected_preparation_audit']=rows

    def test_05_04_recipe_613_seeded_large_bit_queries(self):
        rng=random.Random(SEED);rows=[]
        for S,N,number in ((613,11,5000),(613,12,1000),(48,10,1000),(112,10,1000),((1<<200)+(1<<99)+5,203,1000)):
            G=1<<(N-1);checked=0
            while checked<number:
                r,c=rng.randrange(1,G),rng.randrange(1,G)
                if r==S or c==S:continue
                self.assertEqual(paint_cell(r,c,S,N,True),bool(cell(r,c,S,N)))
                if checked<10:
                    self.assertEqual(cell(r,c,S,N),direct_cell(r,c,S,N))
                    for t in (-1,1):
                        a,b=assessor(r,S,N),assessor(c,S,N,t)
                        self.assertEqual(mul(a,b,N),mul(a,b,N,True))
                checked+=1
            rows.append({'N':N,'S':S,'queried_cells':checked,'full_basis_controls':10})
        OBSERVATIONS['large_bit_query_samples']=rows

    def test_05_05_literal_mutual_orthogonality_and_repeated_lines(self):
        # One row's line can participate in several cell pairs. Distinct
        # directed cells are not a family of distinct orthogonal algebra lines.
        S=1;N=4;r=2
        partners=[c for c in labels(S,N) if cell(r,c,S,N)]
        self.assertEqual(len(partners),4)
        row_line=assessor(r,S,N)
        self.assertEqual(dot(row_line,row_line),2)
        self.assertEqual(sum(bool(x) for row in table(S,N) for x in row),24)
        OBSERVATIONS['orthogonality_interpretation']={'S':S,'N':N,'row':r,'partner_labels':partners,
             'repeated_row_line':row_line,'ambient_dimension':16,'filled_directed_cells':24,
             'scope':'A cell is a zero-product relationship between two lines, not a fresh independent algebra direction per cell. Infinite orthogonality needs a separately specified enlarged space.'}

    def test_05_06_triadic_relation_join_example(self):
        R={(1,2,3),(4,2,3)};S={(2,5,6)};T={(3,6,7),(3,6,8)}
        result=trijoin(R,S,T)
        self.assertEqual(result,{(1,5,7),(1,5,8),(4,5,7),(4,5,8)})
        self.assertEqual(trijoin(R,set(),T),set())
        self.assertTrue(result<=trijoin(R|{(9,2,3)},S,T))
        self.assertTrue(all(len(t)==3 for t in result))
        OBSERVATIONS['trijoin_example']={'R':sorted(R),'S':sorted(S),'T':sorted(T),'result':sorted(result),
            'scope':'Implements the explicit finite relation definition on p29, not a database architecture or a dynamical chaos model.'}


class GPUVerification(unittest.TestCase):
    def setUp(self):
        if CK.cp is None:self.skipTest('CuPy is unavailable')
        if not CK.loaded():self.skipTest('Build CayleyKernel CUDA library first')
        if CK.cp.cuda.runtime.getDeviceCount()<1:self.skipTest('No CUDA device')

    def test_06_01_multiply_all_basis_pairs(self):
        import numpy as np
        for N in range(1,8):
            d=1<<N;pairs=list(product(range(d),repeat=2))
            A=np.eye(d,dtype=np.float64)[[i for i,j in pairs]]
            B=np.eye(d,dtype=np.float64)[[j for i,j in pairs]]
            expected=np.zeros_like(A)
            for k,(i,j) in enumerate(pairs):expected[k,i^j]=recursive_sign(i,j,d)
            np.testing.assert_array_equal(replay_product(A,B),expected)

    def test_06_02_multiply_complete_emanation_tables(self):
        import numpy as np
        for S,N in ((1,4),(15,5),(25,6),(57,7)):
            d=1<<N;aa=[];bb=[];wanted=[];reverse_wanted=[]
            for r,c in valid_pairs(S,N):
                for slope in (-1,1):
                    a=assessor(r,S,N);b=assessor(c,S,N,slope)
                    aa.append([a.get(i,0) for i in range(d)])
                    bb.append([b.get(i,0) for i in range(d)])
                    z=mul(a,b,N,True);wanted.append([z.get(i,0) for i in range(d)])
                    z=mul(b,a,N,True);reverse_wanted.append([z.get(i,0) for i in range(d)])
            A=np.asarray(aa,dtype=np.float64);B=np.asarray(bb,dtype=np.float64)
            # Check all products, including nonzero/diagonal/hidden products.
            np.testing.assert_array_equal(replay_product(A,B),np.asarray(wanted))
            np.testing.assert_array_equal(replay_product(B,A),np.asarray(reverse_wanted))

    def test_06_03_multiply_zero_divisor_scaling_and_recipe_counterexample(self):
        import numpy as np
        a=assessor(2,1,5);b=assessor(8,1,5,-1)
        A=np.asarray([a.get(i,0) for i in range(32)],dtype=np.float64)
        B=np.asarray([b.get(i,0) for i in range(32)],dtype=np.float64)
        np.testing.assert_array_equal(replay_product(3*A,-7*B),np.zeros(32))
        # Actual nonzero product in a cell the literal S48 recipe fills.
        found=next((r,c) for r,c in valid_pairs(48,7) if paint_cell(r,c,48,7) and not cell(r,c,48,7))
        for s in (-1,1):
            aa=assessor(found[0],48,7);bb=assessor(found[1],48,7,s)
            expected=mul(aa,bb,7,True);self.assertTrue(expected)
            A=np.asarray([aa.get(i,0) for i in range(128)],dtype=np.float64)
            B=np.asarray([bb.get(i,0) for i in range(128)],dtype=np.float64)
            np.testing.assert_array_equal(replay_product(A,B),np.asarray([expected.get(i,0) for i in range(128)]))


try:
    import pytest
except ImportError:
    pass
else:
    PaperVerification=pytest.mark.cpu(PaperVerification)
    GPUVerification=pytest.mark.cpu(GPUVerification)
