# SPDX-License-Identifier: Apache-2.0
"""Run in paper order and save provenance, outcomes and compact experiment data."""
import argparse
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
from . import test_sections


class Result(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.records=[]

    def addSuccess(self,test):
        super().addSuccess(test)
        self.records.append({'test':test.id(),'status':'passed'})

    def addSkip(self,test,reason):
        super().addSkip(test,reason)
        self.records.append({'test':test.id(),'status':'skipped','reason':reason})

    def addFailure(self,test,err):
        super().addFailure(test,err)
        self.records.append({'test':test.id(),'status':'failed','error':self._exc_info_to_string(err,test)})

    def addError(self,test,err):
        super().addError(test,err)
        self.records.append({'test':test.id(),'status':'error','error':self._exc_info_to_string(err,test)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper',type=Path,help='Optional original PDF to hash')
    parser.add_argument('--output',type=Path,help='New JSON evidence file; never overwrites')
    args=parser.parse_args()
    if args.output and args.output.exists():parser.error('output exists; choose a new filename')
    if args.paper and not args.paper.is_file():parser.error('paper does not exist')
    test_sections.EVIDENCE.clear()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                             for cls in (test_sections.Sections,test_sections.GPUVerification))
    start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2,resultclass=Result).run(suite)
    elapsed=time.monotonic()-start
    root=Path(__file__).resolve().parents[3]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sources=list(Path(__file__).parent.glob('*.py'))+[
        root/'CayleyKernel/__init__.py',root/'CayleyKernel/_cpu.py',root/'CayleyKernel/_algebra.py',
        root/'tests/papers/desktop_batch/common.py',root/'tests/papers/type_replay.py',
        root/'tests/papers/arxiv_2608_28176/algebra.py']
    if args.output:
        report={
            'date':datetime.now(ZoneInfo('America/Vancouver')).isoformat(),
            'paper_title':'Standard model symmetries and the nested embeddings of R subset C subset H subset O',
            'doi':'10.1016/j.physletb.2026.141078',
            'paper_sha256':sha(args.paper) if args.paper else None,
            'python':platform.python_version(),'numpy':np.__version__,'sympy':sp.__version__,
            'cupy':getattr(CK.cp,'__version__',None),'cpu_backend':CK.cpu_backend(),
            'kernel_loaded':CK.loaded(),'kernel_sha256':sha(Path(CK.library_path())) if CK.loaded() else None,'elapsed_seconds':elapsed,
            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'git_status':subprocess.check_output(['git','status','--short'],cwd=root,text=True).splitlines(),
            'source_sha256':{str(p.relative_to(root)):sha(p) for p in sources},
            'tests_run':result.testsRun,'passed':result.testsRun-len(result.skipped)-len(result.errors)-len(result.failures),
            'skipped':len(result.skipped),'failures':len(result.failures),'errors':len(result.errors),
            'tests':result.records,'experiments':test_sections.EVIDENCE,
            'scope':'Exact operator calculations at fixed dimensions and finite sparse high-index checks. '
                    'High sparse levels are not full dense GPU execution. This does not verify particle physics phenomenology.',
        }
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report,stream,indent=2);stream.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
