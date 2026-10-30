# SPDX-License-Identifier: Apache-2.0
"""Independent follow-up suite: XOR subalgebras, unit centralizers, explicit irreps."""
from collections import Counter
from itertools import product,combinations
import random
import unittest
try:
    import numpy as np
    import sympy as s
except ModuleNotFoundError as exc:
    raise unittest.SkipTest('follow-up needs NumPy and SymPy: '+str(exc))
import CayleyKernel as CK
from .extension_algebra import (
    xor_span,rref_subspaces,canonical_basis,gaussian_count,tables,defects,
    table_product,failure_witness,survey,sampled_bases,basis_profile,graph_centralizer,
    sparse_unit_checks,signed_isomorphism,gpu_bases,
)
from .algebra import oracle,operator,e,flat,span_rank,comm,matrix_units,restrict
from .particle_reconstruction import (
    model,adj,clean,restricted_module,restriction_matrix,commutant_dimension,
    sparse_matrix,EXPECTED,EXPECTED_CHARGES,
)
try:
    import pytest
    pytestmark=pytest.mark.cpu
except ImportError:
    pass

CONFIG={'max_exhaustive_level':9,'high_samples':256,'max_unit_level':10,'gpu_max_level':12}
EVIDENCE={}


class Extensions(unittest.TestCase):
    def test_01_enumeration_and_exact_composition_criterion(self):
        for n in range(3,7):
            bs=list(rref_subspaces(n));self.assertEqual(len(bs),gaussian_count(n))
            self.assertEqual(len({tuple(sorted(xor_span(b))) for b in bs}),len(bs))
        for b in rref_subspaces(4):
            canonical=canonical_basis(b)
            for alternative in combinations(xor_span(b)[1:],3):
                reduced=canonical_basis(alternative)
                if len(reduced)==3:self.assertEqual(reduced,canonical)
        with self.assertRaises(ValueError):list(sampled_bases(4,16,0))
        report=survey(4,rref_subspaces(4))
        for key,rep in report['representatives'].items():
            sign=np.array(rep['signs']);xs=s.symbols('x:8');ys=s.symbols('y:8')
            out=table_product(sign,xs,ys)
            polynomial=s.expand(sum(v*v for v in out)-sum(x*x for x in xs)*sum(y*y for y in ys))
            self.assertEqual(polynomial==0,int(key)==0)
            if int(key):
                w=rep['norm_counterexample'];self.assertIsNotNone(w)
                self.assertNotEqual(w['product_norm_sq'],w['expected_norm_sq'])
            # Signed permutation and skew properties required by the Hurwitz criterion.
            for i in range(1,8):
                li=s.zeros(8)
                for j in range(8):li[i^j,j]=int(sign[i,j])
                self.assertEqual(li.T,-li);self.assertEqual(li.T*li,s.eye(8))
        EVIDENCE['criterion']={'positive_and_negative_symbolic_certification':True,
                               'meaning':'Hurwitz criterion for all real coefficients in each classified span'}

    def test_02_exhaustive_xor_classification(self):
        rows=[]
        # Regression totals are supplemented by independent enumeration and polynomial tests.
        expected={3:1,4:8,5:50,6:310,7:2046,8:14478,9:107950}
        for n in range(3,CONFIG['max_exhaustive_level']+1):
            row=survey(n,rref_subspaces(n),retain=True)
            self.assertEqual(row['total'],gaussian_count(n))
            if n in expected:self.assertEqual(row['composition'],expected[n])
            self.assertEqual(set(row['defective_generator_pair_histogram'])-{0,12},set())
            rows.append(row)
        EVIDENCE['exhaustive_xor_subspaces']=rows

    def test_03_high_bit_xor_classification_and_oracle(self):
        rows=[]
        for n in (12,20,32,64,128,256,1024):
            bs=list(sampled_bases(n,CONFIG['high_samples'],2026141078))
            row=survey(n,bs)
            self.assertEqual(row['total'],CONFIG['high_samples'])
            # Independent doubling for every product of the first eight high-bit spans.
            for b,table in zip(bs[:8],tables(bs[:8])):
                es=xor_span(b)
                for i,j in product(range(8),repeat=2):
                    self.assertEqual(int(table[i,j]),oracle(es[i],es[j],n))
            self.assertGreater(row['composition'],0)
            self.assertGreater(row['noncomposition'],0)
            rows.append(row)
        for n in (4,8,12,16,20):
            batch=64 if n<=12 else 8 if n==16 else 2
            bases=gpu_bases(n,batch)
            self.assertEqual(len(bases),batch)
            self.assertEqual(set(map(int,defects(tables(bases[:2])))),{0,12})
        EVIDENCE['sampled_xor_subspaces']=rows

    def test_04_signed_isometry_classes_and_counterexamples(self):
        small=survey(6,rref_subspaces(6))
        targets={int(k):np.array(v['signs']) for k,v in small['representatives'].items()}
        patterns=set(small['sign_table_histogram'])
        for row in EVIDENCE.get('exhaustive_xor_subspaces',[])+EVIDENCE.get('sampled_xor_subspaces',[]):
            patterns.update(row['sign_table_histogram'])
        records=[]
        for key in sorted(patterns):
            source=np.frombuffer(bytes.fromhex(key),dtype=np.int8).reshape(8,8)
            d=int(defects(source)[0]);self.assertIn(d,targets)
            certificate=signed_isomorphism(source,targets[d]);self.assertIsNotNone(certificate)
            p,sign=certificate['permutation'],certificate['signs']
            for i,j in product(range(8),repeat=2):
                self.assertEqual(int(source[i,j])*sign[i^j],sign[i]*sign[j]*int(targets[d][p[i],p[j]]))
            if d:self.assertEqual(failure_witness(source)['product_norm_sq'],0)
            records.append({'table_hex':key,'defective_pairs':d,'isometry':certificate})
        EVIDENCE['observed_table_classes']={'patterns':len(patterns),'certificates':records,
            'targets':{str(k):v.tolist() for k,v in targets.items()},
            'scope':'All observed tables; no claim that sampling exhausts all higher-level subspaces.'}

    def test_05_all_basis_units_complete_columns(self):
        rows=[]
        for n in range(3,CONFIG['max_unit_level']+1):
            d=1<<n
            for u in range(1,d):
                p=basis_profile(d,u)
                self.assertTrue(p['left_square_minus_identity']);self.assertTrue(p['right_square_minus_identity'])
                self.assertTrue(p['commute']);self.assertEqual(p['projector_ranks'],[2,d-2])
                rows.append({'level':n,**p})
        # Larger ambient spaces, selected units but every column for each selected unit.
        for n in (11,12):
            d=1<<n
            units=sorted({1,2,3,5,7,d//4,d//2-1,d//2,d//2+1,d-1})
            for u in units:
                p=basis_profile(d,u)
                self.assertTrue(p['commute']);self.assertEqual(p['projector_ranks'],[2,d-2])
                self.assertTrue(p['left_square_minus_identity'] and p['right_square_minus_identity'])
                rows.append({'level':n,**p})
        EVIDENCE['unit_profiles']=rows

    def test_06_independent_matrix_entry_centralizer_solve(self):
        rows=[]
        for d in (8,16,32):
            for u in range(1,d):
                actual=graph_centralizer(d,u);skew=graph_centralizer(d,u,True)
                p=basis_profile(d,u)
                self.assertEqual(actual,p['joint_centralizer_real_dimension'])
                self.assertEqual(skew,p['skew_joint_centralizer_real_dimension'])
                rows.append({'dimension':d,'unit':u,'full':actual,'skew':skew})
        # General rational matrix-equation elimination, separate from the signed graph solver.
        for u in (1,3,8,15):
            l=operator(e(u),16);r=operator(e(u),16,True)
            bs=restrict(matrix_units(16),lambda x:s.Matrix.vstack(flat(comm(x,l)),flat(comm(x,r))))
            self.assertEqual(len(bs),graph_centralizer(16,u))
        EVIDENCE['independent_centralizer_solves']=rows

    def test_07_high_bit_nondoubling_units(self):
        rng=random.Random(14107807);rows=[]
        for n in (20,32,64,128,256,1024):
            units=[1,3,5,(1<<(n-1))+3,(1<<n)-1]+[rng.getrandbits(n)|1 for _ in range(11)]
            for u in units:
                columns=[0,u,1,u^1]+[rng.getrandbits(n) for _ in range(64)]
                sparse_unit_checks(n,u,columns)
            rows.append({'level':n,'units':len(units),'columns_per_unit':68,
                         'complete_column_scan':False})
        EVIDENCE['high_bit_unit_samples']=rows

    def test_08_reference_literal_color_generator_discrepancy(self):
        m=model();raw=m['printed_color'];fixed=m['color']
        self.assertEqual(raw[2].trace(),-2*s.I)
        self.assertEqual(span_rank(raw+[comm(a,b) for a,b in product(raw,repeat=2)]),9)
        self.assertEqual([g.trace() for g in fixed],[0]*8)
        self.assertEqual(span_rank(fixed+[comm(a,b) for a,b in product(fixed,repeat=2)]),8)
        self.assertEqual(raw[:2]+raw[3:],fixed[:2]+fixed[3:])
        EVIDENCE['reference_discrepancy']={
            'source':'arXiv:2505.07923v4, page 6, Eq.26, i Lambda_3',
            'literal_trace':'-2*I','literal_span_with_commutators_dimension':9,
            'minimal_repair':'Change the sign of its second term from -i to +i.',
            'repaired_trace':'0','repaired_closed_span_dimension':8,
            'interpretation':'A passing regression reproduces a printed discrepancy, not verification of the literal formula.'}

    def test_09_reference_generators_match_original_paper(self):
        m=model();l=m['left'];p=m['projectors'][1]
        q=lambda i,j:l[i]*l[j]
        original=[q(3,4)-q(1,5),q(1,4)+q(3,5),q(1,3)-q(4,5),
                  -q(2,5)-q(4,6),q(2,4)-q(5,6),-q(1,6)-q(2,3),
                  -q(1,2)-q(3,6),-q(1,3)-q(4,5)+2*q(2,6)]
        self.assertEqual(span_rank(m['color']+[g*p for g in original]),8)
        for g in m['color']+m['weak']:
            self.assertEqual(adj(g),-g);self.assertEqual(g.trace(),0)
            for proj in m['projectors']:self.assertEqual(comm(g,proj),s.zeros(8))
        self.assertEqual(clean(-sum((g*g/4 for g in m['color']),s.zeros(8))),s.Rational(4,3)*p)
        self.assertEqual(clean(-sum((g*g/4 for g in m['weak']),s.zeros(8))),s.Rational(3,4)*m['projectors'][2])
        for a,b in product(m['color'],m['weak']):self.assertEqual(comm(a,b),s.zeros(8))
        # Charge conventions: reference weights are 1/2 minus the simple weights,
        # with the last two singlet labels exchanged (conjugate charge convention).
        simple=s.diag(0,*([s.Rational(1,3)]*3),s.Rational(1,2),s.Rational(1,2),0,1)
        grouped=s.Matrix.hstack(*(v for vs in m['states'] for v in vs))
        self.assertEqual(clean(adj(grouped)*m['Y']*grouped),s.eye(8)/2-simple)

    def test_10_clifford_ladders_vacuum_and_peirce_projectors(self):
        m=model();aa=[s.kronecker_product(s.eye(2),a) for a in m['a']]+[m['beta']]
        for i,a in enumerate(aa):
            for j,b in enumerate(aa):
                self.assertEqual(clean(a*b+b*a),s.zeros(16))
                self.assertEqual(clean(a*adj(b)+adj(b)*a),s.eye(16) if i==j else s.zeros(16))
        self.assertEqual(m['vacuum16'].rank(),1)
        self.assertEqual(m['vacuum16']**2,m['vacuum16'])
        self.assertEqual(adj(m['vacuum16']),m['vacuum16'])
        ps=m['projectors'];self.assertEqual([p.rank() for p in ps],[1,3,2,1,1])
        self.assertEqual(sum(ps,s.zeros(8)),s.eye(8))
        for i,p in enumerate(ps):
            self.assertEqual(adj(p),p)
            for j,q in enumerate(ps):self.assertEqual(p*q,p if i==j else s.zeros(8))
        EVIDENCE['particle_projector_ranks']={'internal_complex':[1,3,2,1,1],'with_spatial_factor':[2,6,4,2,2]}

    def test_11_explicit_particle_ladder_bases(self):
        m=model();counts={}
        for label,basis in m['full_ladder_basis'].items():
            i,j=map(int,label);pi=s.kronecker_product(s.eye(2),m['projectors'][i-1]);pj=s.kronecker_product(s.eye(2),m['projectors'][j-1])
            n=len(m['states'][i-1])*len(m['states'][j-1]);self.assertEqual(len(basis),4*n)
            self.assertEqual(span_rank(basis),4*n)
            for x in basis:
                self.assertEqual(pi*x*pj,x)
                self.assertEqual(pj*adj(x)*pi,adj(x))
            counts[label]={'complex_dimension_one_orientation':4*n,'real_dimension_with_hc':8*n}
        self.assertEqual(sum(v['real_dimension_with_hc'] for v in counts.values()),192)
        self.assertEqual(sum(counts[k]['real_dimension_with_hc'] for k in ('12','34','35','45')),64)
        EVIDENCE['explicit_ladder_basis_counts']=counts

    def test_12_certified_internal_irreducible_representations(self):
        m=model();records=[]
        fundamental=restriction_matrix(m['color'][7]/2,m['states'][1])
        cubic_fund=clean((-s.I*fundamental)**3).trace()
        self.assertNotEqual(cubic_fund,0)
        for label,(name,color,weak,y) in EXPECTED.items():
            i,j=map(int,label);cs,ws,yy=restricted_module(i,j)
            d=yy.rows
            self.assertEqual(yy,y*s.eye(d))
            qaction=clean(yy-s.I*ws[2])
            self.assertEqual(qaction.eigenvals(),EXPECTED_CHARGES[label])
            self.assertEqual(commutant_dimension(cs+ws),1)
            self.assertEqual(clean(-sum((g*g for g in cs),s.zeros(d))),
                             (0 if color=='1' else s.Rational(4,3))*s.eye(d))
            self.assertEqual(clean(-sum((g*g for g in ws),s.zeros(d))),
                             (0 if weak==1 else s.Rational(3,4))*s.eye(d))
            cubic=clean((-s.I*cs[7])**3).trace()
            self.assertEqual(s.simplify(cubic-(0 if color=='1' else (1 if color=='3' else -1)*weak*cubic_fund)),0)
            # Directly verify that the actual octonion matrix action induces these matrices.
            bi=m['states'][i-1];bj=m['states'][j-1]
            embedding=s.Matrix.hstack(*(flat(a*adj(b)) for b in bj for a in bi))
            for g,restricted in zip([g/2 for g in m['color']+m['weak']]+[m['Y']],cs+ws+[yy]):
                for k in range(d):
                    x=embedding[:,k].reshape(8,8)
                    self.assertEqual(clean(flat(comm(g,x))-embedding*restricted[:,k]),s.zeros(64,1))
            records.append({'block':label,'paper_assignment':name,'color':color,'weak_dimension':weak,
                            'hypercharge':str(y),'irrep_complex_dimension':d,'spatial_copies':4,
                            'complex_commutant_dimension':1,'color_cartan_cubic_trace':str(cubic),
                            'electric_charge_eigenvalues':{str(k):v for k,v in qaction.eigenvals().items()},
                            'generators':{'color':[sparse_matrix(g) for g in cs],
                                          'weak':[sparse_matrix(g) for g in ws],'Y':sparse_matrix(yy),'Q':sparse_matrix(qaction)}})
        EVIDENCE['particle_irreps']=records
        EVIDENCE['explicit_operators']={
            'coordinate_convention':'Kernel O coordinates; C2 quaternion factor first in tensor products.',
            'projectors':[sparse_matrix(p) for p in m['projectors']],
            'octonion_annihilation':[sparse_matrix(a) for a in m['a']],
            'beta4':sparse_matrix(m['beta']),
            'internal_ladder_bases':{k:[sparse_matrix(x) for x in v] for k,v in m['internal_ladder_basis'].items()},
            'internal_state_vectors':[[sparse_matrix(v) for v in bs] for bs in m['states']]}

    def test_13_diagonal_adjoint_representations_and_missing_top(self):
        m=model();rows=[]
        for index,gs,want in ((1,m['color'],{s.Integer(0):1,s.Integer(3):8}),
                               (2,m['weak'],{s.Integer(0):1,s.Integer(2):3})):
            basis=m['states'][index];n=len(basis)
            reps=[]
            for g in gs:
                a=restriction_matrix(g/2,basis)
                reps.append(s.kronecker_product(s.eye(n),a)-s.kronecker_product(a.T,s.eye(n)))
            casimir=clean(-sum((a*a for a in reps),s.zeros(n*n)))
            self.assertEqual(casimir.eigenvals(),want)
            rows.append({'block':index+1,'casimir_eigenvalues':{str(k):v for k,v in want.items()}})
        ranks=[p.rank() for p in m['projectors']]
        self.assertEqual(sum(4*n*n for n in ranks),64)
        # Only H23 carries a colored weak doublet, and only H25 an up-type singlet.
        self.assertEqual([k for k,v in EXPECTED.items() if v[1]=='3' and v[2]==2],['23'])
        self.assertEqual([k for k,v in EXPECTED.items() if v[1]=='3' and v[3]==s.Rational(2,3)],['25'])
        EVIDENCE['diagonal_adjoint_checks']=rows
        EVIDENCE['particle_scope']={'diagonal_real_dimension':64,'off_diagonal_real_dimension':192,
            'inner_real_dimension':64,'outer_real_dimension':128,
            'missing_in_inner_sector':['(3,2,1/6) top-containing doublet','(3,1,2/3) top singlet'],
            'not_established':['Lorentzian spin statistics','mass spectrum','dynamics','experimental particle identification']}


class ExtensionGPU(unittest.TestCase):
    def setUp(self):
        if CK.cp is None or not CK.loaded():self.skipTest('batched multiply(): CuPy or compiled kernel unavailable')
        try:count=CK.cp.cuda.runtime.getDeviceCount()
        except CK.cp.cuda.runtime.CUDARuntimeError as exc:self.skipTest('batched multiply(): '+str(exc))
        if count<1:self.skipTest('batched multiply(): no CUDA device')

    def test_14_batched_composition_and_noncomposition_products(self):
        cp=CK.cp;rows=[]
        for n in (4,8,12,16,20):
            if n>CONFIG['gpu_max_level']:continue
            batch=64 if n<=12 else 8 if n==16 else 2
            bs=gpu_bases(n,batch);ss=tables(bs);d=1<<n
            x=np.zeros((batch,d));y=np.zeros_like(x);expected=np.zeros_like(x)
            rng=random.Random(n+141078)
            for row,(basis,sign) in enumerate(zip(bs,ss)):
                a=[rng.randrange(-2,3) for _ in range(8)];b=[rng.randrange(-2,3) for _ in range(8)]
                if defects(sign)[0]:
                    witness=failure_witness(sign);a=witness['x'];b=witness['y']
                out=table_product(sign,a,b)
                for k,idx in enumerate(xor_span(basis)):
                    x[row,idx]=a[k];y[row,idx]=b[k];expected[row,idx]=out[k]
            actual=cp.asnumpy(CK.multiply(cp.asarray(x),cp.asarray(y)))
            np.testing.assert_array_equal(actual,expected)
            rows.append({'level':n,'dimension':d,'batch':batch,'dtype':'float64'})
        EVIDENCE['gpu_batches']=rows
