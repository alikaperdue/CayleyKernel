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
from .loops import classification, all_spaces, indices, labels, lattice_data


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
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, help='PDF to hash; never modified')
    parser.add_argument('--output', type=Path, help='Write JSON and complete test log here')
    args = parser.parse_args()
    test_sections.OBSERVATIONS.clear()
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
        'paper': '0907.2047v3',
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
        'scope': 'Exhaustive signed basis loop of order 64; coordinate subalgebras, not arbitrary real subalgebras. Symbolic norm/flexibility/quadratic checks have all-real coefficient scope. Direct GPU checks may be skipped.',
        'dense_seed': 9072047,
        'algebra_dimension': 32,
    }
    sys.stdout.write(stream.getvalue())
    if args.output:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'results.json').write_text(json.dumps(report,indent=2,default=json_exact)+'\n')
        (args.output/'test-output.txt').write_text(stream.getvalue())
        if result.wasSuccessful():
            spaces,_,_,covers = lattice_data()
            nodes = []
            for node_id,mask in enumerate(spaces):
                if mask==0:
                    name,mapping,signature = 'trivial',{1:1},None
                elif mask.bit_count()==32:
                    name,mapping,signature = 'T',{x:x for x in labels(mask)},None
                else:
                    name,mapping,signature = classification()[mask]
                nodes.append({
                    'id':node_id,'support_mask':mask,
                    'support':list(indices(mask)) if mask else [],
                    'loop_order':2*mask.bit_count() if mask else 1,
                    'coordinate_algebra_dimension':mask.bit_count() if mask else None,
                    'class':name,'distinguishing_invariant':signature,
                    'representative_isomorphism':sorted(mapping.items()),
                })
            dataset = {'paper':'0907.2047v3','nodes':nodes,'covers':covers,
                       'id_convention':'Our deterministic node IDs are NOT FINITAS IDs. Node 0 denotes singleton {+1}; other supports include both signs.',
                       'isomorphism_convention':'Signed one-based module labels, representative to this node. Class T and trivial use identity maps.',
                       'covers_convention':'[lower_node_id,upper_node_id], immediate inclusions',
                       'source_sha256':report['source_sha256']}
            (args.output/'subloops.json').write_text(json.dumps(dataset,indent=2,default=json_exact)+'\n')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
