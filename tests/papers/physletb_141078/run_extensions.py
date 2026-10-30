# SPDX-License-Identifier: Apache-2.0
"""Run the separate follow-up and export classification data and exact irrep matrices."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import unittest
from zoneinfo import ZoneInfo
import numpy as np
import sympy as sp
import CayleyKernel as CK
from . import test_extensions as test
from .extension_algebra import canonical_basis,xor_span,tables,defects,failure_witness


class Result(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.records=[]
    def addSuccess(self,t):
        super().addSuccess(t);self.records.append({'test':t.id(),'status':'passed'})
    def addSkip(self,t,reason):
        super().addSkip(t,reason);self.records.append({'test':t.id(),'status':'skipped','reason':reason})
    def addFailure(self,t,err):
        super().addFailure(t,err);self.records.append({'test':t.id(),'status':'failed','error':self._exc_info_to_string(err,t)})
    def addError(self,t,err):
        super().addError(t,err);self.records.append({'test':t.id(),'status':'error','error':self._exc_info_to_string(err,t)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='New directory; existing output directories are refused')
    parser.add_argument('--paper',type=Path,help='Optional original CERN-main.pdf for hashing')
    parser.add_argument('--reference',type=Path,help='Optional arXiv:2505.07923v4 PDF for hashing')
    parser.add_argument('--max-exhaustive-level',type=int,choices=range(3,11),default=9)
    parser.add_argument('--high-samples',type=int,default=256)
    parser.add_argument('--max-unit-level',type=int,choices=range(3,13),default=10)
    parser.add_argument('--gpu-max-level',type=int,choices=(4,8,12,16,20),default=12)
    parser.add_argument('--classify',help='Only classify three independent indices, e.g. 1,2,4; accepts 0x hex')
    args=parser.parse_args()
    if args.classify:
        try:b=tuple(int(x,0) for x in args.classify.split(','))
        except ValueError:parser.error('classify needs three comma-separated integer indices')
        if len(b)!=3 or min(b)<1 or len(canonical_basis(b))!=3:parser.error('indices must be positive and XOR independent')
        table=tables([b])[0];d=int(defects(table)[0])
        print(json.dumps({'basis':b,'elements':xor_span(b),'composition':not d,
                          'defective_pairs':d,'norm_counterexample':failure_witness(table) if d else None},indent=2))
        return
    if args.high_samples<64:parser.error('use at least 64 high samples for this survey suite')
    if args.output and args.output.exists():parser.error('output directory exists; choose a fresh path')
    for p in (args.paper,args.reference):
        if p is not None and not p.is_file():parser.error('input PDF not found: '+str(p))
    test.CONFIG.update(max_exhaustive_level=args.max_exhaustive_level,high_samples=args.high_samples,
                       max_unit_level=args.max_unit_level,gpu_max_level=args.gpu_max_level)
    test.EVIDENCE.clear()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                             for cls in (test.Extensions,test.ExtensionGPU))
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    elapsed=time.monotonic()-start
    root=Path(__file__).resolve().parents[3];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if args.output:
        args.output.mkdir(parents=True)
        csvpath=args.output/'composition_subspaces.csv'
        with csvpath.open('x',newline='') as f:
            writer=csv.writer(f);writer.writerow(['level','rref_basis_1','rref_basis_2','rref_basis_3'])
            for row in test.EVIDENCE.get('exhaustive_xor_subspaces',[]):
                for b in row['passing_rref_bases']:writer.writerow([row['level'],*b])
                del row['passing_rref_bases']
        unitpath=args.output/'basis_units.csv'
        rows=test.EVIDENCE.get('unit_profiles',[])
        with unitpath.open('x',newline='') as f:
            names=['level','unit','rank_P','rank_Q','joint_real_dimension','skew_real_dimension']
            w=csv.writer(f);w.writerow(names)
            for row in rows:w.writerow([row['level'],row['unit'],*row['projector_ranks'],
                row['joint_centralizer_real_dimension'],row['skew_joint_centralizer_real_dimension']])
        if rows:
            test.EVIDENCE['unit_profiles_summary']={
                'complete_column_profiles':len(rows),'csv':'basis_units.csv',
                'levels':{str(n):sum(x['level']==n for x in rows) for n in sorted({x['level'] for x in rows})},
                'all_square_to_minus_identity_and_commute':all(x['left_square_minus_identity'] and x['right_square_minus_identity'] and x['commute'] for x in rows)}
            del test.EVIDENCE['unit_profiles']
        sources=list(Path(__file__).parent.glob('*.py'))+[
            root/'CayleyKernel/__init__.py',root/'CayleyKernel/_cpu.py',root/'CayleyKernel/_algebra.py',
            root/'tests/papers/desktop_batch/common.py',root/'tests/papers/type_replay.py',
            root/'tests/papers/arxiv_2608_28176/algebra.py']
        report={'date':datetime.now(ZoneInfo('America/Vancouver')).isoformat(),
            'original_paper':{'doi':'10.1016/j.physletb.2026.141078','sha256':sha(args.paper) if args.paper else None},
            'reference':{'id':'arXiv:2505.07923v4','url':'https://arxiv.org/abs/2505.07923v4',
                         'sha256':sha(args.reference) if args.reference else None},
            'config':test.CONFIG.copy(),'elapsed_seconds':elapsed,'python':platform.python_version(),
            'numpy':np.__version__,'sympy':sp.__version__,'cupy':getattr(CK.cp,'__version__',None),
            'cpu_backend':CK.cpu_backend(),'kernel_loaded':CK.loaded(),
            'kernel_sha256':sha(Path(CK.library_path())) if CK.loaded() else None,
            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'source_sha256':{str(p.relative_to(root)):sha(p) for p in sources},
            'artifacts_sha256':{p.name:sha(p) for p in (csvpath,unitpath)},
            'tests_run':result.testsRun,'passed':result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
            'skipped':len(result.skipped),'failures':len(result.failures),'errors':len(result.errors),
            'tests':result.records,'experiments':test.EVIDENCE,
            'scope':'Exhaustive finite XOR surveys, sampled high levels, and exact internal complex irreps. '
                    'The literal reference Eq.26 discrepancy is retained separately from its tested sign repair.'}
        with (args.output/'results.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
