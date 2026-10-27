"""Run finite-field CPU verification; --cuda also selects the user GPU tests.

Normal pytest runs GPU tests automatically when CUDA is present. This report
runner defaults to CPU-only, without any CUDA-device probe or multiply call.
"""
import argparse
import csv
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

import CayleyKernel as CK
from . import test_sections
from .figures import line_chart
from .finite import correct_norm_zero, paper_zd, paper_units


class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.passed=[]

    def addSuccess(self,test):
        super().addSuccess(test)
        self.passed.append(test.id())


def exact_json(value):
    if isinstance(value,Fraction):return str(value)
    raise TypeError(type(value).__name__)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper',type=Path,help='Hash the local PDF without changing it')
    parser.add_argument('--output',type=Path,help='Write report, exact data and table renderings here')
    parser.add_argument('--cuda',action='store_true',help='User opt-in: actually execute multiply() tests on a CUDA machine')
    args=parser.parse_args()
    test_sections.OBSERVATIONS.clear();test_sections.CERTIFICATES.clear()
    loader=unittest.defaultTestLoader
    suite=unittest.TestSuite([loader.loadTestsFromTestCase(test_sections.PaperVerification)])
    if args.cuda:
        suite.addTests(loader.loadTestsFromTestCase(test_sections.GPUVerification))
    stream=io.StringIO();start=time.perf_counter()
    result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=RecordedResult).run(suite)
    root=Path(__file__).resolve().parents[3]
    files=sorted(Path(__file__).parent.glob('*.py'))+[root/'CayleyKernel/__init__.py']
    metadata={
        'paper':'617-0702/2024','title':'Graphical and Computational Analysis of Cayley-Dickson Algebra Elements over Z_p',
        'paper_sha256':hashlib.sha256(args.paper.read_bytes()).hexdigest() if args.paper else None,
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'numpy':test_sections.np.__version__,'platform':platform.platform(),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'kernel_loaded':CK.loaded(),'kernel_abi':CK.abi_version(),'cuda_requested':args.cuda,
        'gpu_tests_not_selected':0 if args.cuda else 4,
        'cuda_device_probe_performed':args.cuda and CK.cp is not None and CK.loaded(),
        'elapsed_seconds':time.perf_counter()-start,'tests_run':result.testsRun,'passed':result.passed,
        'skipped':[{'test':t.id(),'reason':reason} for t,reason in result.skipped],
        'failures':[{'test':t.id(),'traceback':trace} for t,trace in result.failures],
        'errors':[{'test':t.id(),'traceback':trace} for t,trace in result.errors],
        'successful':result.wasSuccessful(),'seed':test_sections.SEED,'observations':test_sections.OBSERVATIONS,
        'scope':'Exact modular arithmetic; complete small finite-field enumerations and independent quadratic-form convolutions. Composition-algebra unit/ZD classifications are not extended to sedenions. Printed formulas and corrected counts are kept separate.'}
    sys.stdout.write(stream.getvalue())
    if args.output:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'results.json').write_text(json.dumps(metadata,indent=2,default=exact_json)+'\n')
        (args.output/'test-output.txt').write_text(stream.getvalue())
        if result.wasSuccessful():
            (args.output/'certificates.json').write_text(json.dumps({
                'paper':'617-0702/2024','paper_sha256':metadata['paper_sha256'],'source_sha256':metadata['source_sha256'],
                'index_convention':'Mathematical zero-based indices; basis inputs/outputs are translated signed one-based labels.',
                'cases':test_sections.CERTIFICATES},indent=2,default=exact_json)+'\n')
            obs=test_sections.OBSERVATIONS
            rows=[]
            for a,b,c in zip(obs['figure1'],obs['figure2'],obs['figure3']):
                p=a['p'];zero=correct_norm_zero(p,8)
                rows.append({'p':p,'idempotents_paper':a['paper'],'idempotents_verified':a['verified_quadratic_count'],
                    'nilpotents_paper':b['paper'],'nilpotents_verified':b['verified_quadratic_count'],
                    'intersection_paper':c['paper'],'intersection_corrected':c['corrected_coordinate_intersection'],
                    'zero_divisors_paper':paper_zd(p,8),'norm_zero_including_zero':zero,
                    'nonzero_zero_divisors':zero-1,'units_paper':paper_units(p,8),'units_corrected':p**8-zero})
            with (args.output/'figure-data.csv').open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            line_chart(rows,[('idempotents_verified','Verified and printed counts coincide','#245f9f')],
                       'Figure 1 recomputed: octonion idempotents, including 0 and 1',args.output/'figure1.svg')
            line_chart(rows,[('nilpotents_verified','Verified and printed counts coincide','#187a40')],
                       'Figure 2 recomputed: octonion nilpotents, including 0',args.output/'figure2.svg')
            line_chart(rows,[('intersection_corrected','Corrected common-coordinate count','#245f9f'),
                             ('intersection_paper','Printed formula','#b73535')],
                       'Figure 3 comparison: common nilpotents, including 0',args.output/'figure3-comparison.svg')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
