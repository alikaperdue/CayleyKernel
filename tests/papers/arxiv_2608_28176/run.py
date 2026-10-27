"""Run only this paper's suite and optionally preserve an auditable report."""
import argparse
import hashlib
import io
import json
import platform
from pathlib import Path
import sys
import time
import unittest

import CayleyKernel as CK
from . import test_sections


class RecordedResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.passed=[]

    def addSuccess(self,test):
        super().addSuccess(test)
        self.passed.append(test.id())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Directory for results.json and test-output.txt')
    parser.add_argument('--paper',type=Path,help='Local PDF to hash for provenance (not modified)')
    args=parser.parse_args()
    paper_hash=hashlib.sha256(args.paper.read_bytes()).hexdigest() if args.paper else None
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_sections)
    start=time.perf_counter()
    result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=RecordedResult).run(suite)
    elapsed=time.perf_counter()-start
    output=stream.getvalue()
    sys.stdout.write(output)
    metadata={
        'paper':'2608.28176v1',
        'paper_sha256':paper_hash,
        'python':sys.version,
        'platform':platform.platform(),
        'kernel_loaded':CK.loaded(),
        'kernel_abi':CK.abi_version(),
        'cupy_importable':CK.cp is not None,
        'elapsed_seconds':elapsed,
        'tests_run':result.testsRun,
        'passed':result.passed,
        'skipped':[{'test':t.id(),'reason':reason} for t,reason in result.skipped],
        'failures':[{'test':t.id(),'traceback':trace} for t,trace in result.failures],
        'errors':[{'test':t.id(),'traceback':trace} for t,trace in result.errors],
        'successful':result.wasSuccessful(),
        'source_sha256':{
            p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(list(Path(__file__).parent.glob('*.py')) +
                            [Path(__file__).resolve().parents[1]/'type_replay.py'])
        },
        'scope':'Finite checks; mixed signatures use independent real references and complex coordinate-embedding GPU replays.',
        'paper_discrepancies':[
            'Corollary 3.7: split example satisfies hypotheses but has four distinct lines, not six.',
            'Lemma 6.14(1): printed y=(c,gamma_n*d) fails; Proposition 6.7 orbit checked separately.',
        ],
    }
    if args.output:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'test-output.txt').write_text(output)
        (args.output/'results.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':
    raise SystemExit(main())
