"""Run paper tests in section order; optionally save provenance and observations."""
import argparse
import hashlib
from fractions import Fraction
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest
from datetime import datetime, timezone

import numpy as np
import CayleyKernel as CK
from . import test_sections


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test.id())


def json_exact(value):
    if isinstance(value, Fraction):
        return value.numerator if value.denominator == 1 else str(value)
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, help='PDF to hash; never modified')
    parser.add_argument('--output', type=Path, help='Write JSON and complete test log here')
    args = parser.parse_args()
    test_sections.OBSERVATIONS.clear()
    test_sections.CERTIFICATES.clear()
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite([
        loader.loadTestsFromTestCase(test_sections.PaperVerification),
        loader.loadTestsFromTestCase(test_sections.GPUVerification),
    ])
    stream = io.StringIO()
    start = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=RecordedResult).run(suite)
    root = Path(__file__).resolve().parents[3]
    files = list(Path(__file__).parent.glob('*.py')) + [root/'CayleyKernel/__init__.py', root/'tests/papers/type_replay.py']
    report = {
        'paper': 'math/0702075v2',
        'paper_sha256': hashlib.sha256(args.paper.read_bytes()).hexdigest() if args.paper else None,
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
        'git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
        'kernel_loaded': CK.loaded(), 'kernel_abi': CK.abi_version(),
        'cupy_importable': CK.cp is not None,
        'elapsed_seconds': time.perf_counter()-start,
        'tests_run': result.testsRun, 'passed': result.passed,
        'skipped': [{'test': t.id(), 'reason': reason} for t, reason in result.skipped],
        'failures': [{'test': t.id(), 'traceback': trace} for t, trace in result.failures],
        'errors': [{'test': t.id(), 'traceback': trace} for t, trace in result.errors],
        'successful': result.wasSuccessful(),
        'observations': test_sections.OBSERVATIONS,
        'scope': 'Exact rational finite annihilators. B=sqrt(2)*paper braces. Full bilinear basis checks certify Theorem 4.5 in dimensions 16,32,64,128. Samples and explicit rays do not establish global diffeomorphisms or all large-n loci.',
        'seed': test_sections.SEED,
        'maximum_dimension_computed': 256,
    }
    sys.stdout.write(stream.getvalue())
    if args.output:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'results.json').write_text(json.dumps(report,indent=2,default=json_exact)+'\n')
        (args.output/'test-output.txt').write_text(stream.getvalue())
        if result.wasSuccessful():
            dataset={
                'paper':'math/0702075v2','paper_sha256':report['paper_sha256'],
                'source_sha256':report['source_sha256'],
                'index_convention':'Sparse dictionaries use mathematical zero-based indices; JSON keys are strings.',
                'coefficient_convention':'Integers are JSON numbers; nonintegral exact fractions are numerator/denominator strings.',
                'brace_convention':'B(a,b)=sqrt(2)*paper {a,b}. Kernels are unchanged by nonzero real scaling.',
                'normalization':'For unit rays divide element by sqrt(norm_squared); no irrational rounding is used for ranks.',
                'cases':test_sections.CERTIFICATES,
            }
            (args.output/'annihilators.json').write_text(json.dumps(dataset,indent=2,default=json_exact)+'\n')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
