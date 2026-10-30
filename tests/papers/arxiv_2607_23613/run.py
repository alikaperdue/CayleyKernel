# SPDX-License-Identifier: Apache-2.0
"""Run the paper suite and optionally save compact reproducible evidence."""
import argparse
import hashlib
import json
import platform
import subprocess
import time
import unittest
from   pathlib  import Path
from   datetime import datetime
from   zoneinfo import ZoneInfo

import numpy as np
import sympy as sp
import CayleyKernel as CK
from . import test_sections


class Result(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.records.append({'test': test.id(), 'status': 'passed'})

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.records.append({'test': test.id(), 'status': 'skipped', 'reason': reason})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.records.append({'test': test.id(), 'status': 'failed', 'error': self._exc_info_to_string(err, test)})

    def addError(self, test, err):
        super().addError(test, err)
        self.records.append({'test': test.id(), 'status': 'error', 'error': self._exc_info_to_string(err, test)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path)
    parser.add_argument('--output', type=Path, help='New JSON file; existing files are never overwritten')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('output already exists; choose a new evidence file')
    if args.paper and not args.paper.is_file():
        parser.error('paper not found')
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_sections))
    repo = Path(__file__).resolve().parents[3]
    sources = list(Path(__file__).parent.glob('*.py')) + [
        repo/'CayleyKernel/__init__.py', repo/'CayleyKernel/_cpu.py',
        repo/'tests/papers/desktop_batch/common.py', repo/'tests/papers/type_replay.py',
        repo/'tests/papers/arxiv_2608_28176/algebra.py']
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if args.output:
        report = {
            'date': datetime.now(ZoneInfo('America/Vancouver')).isoformat(),
            'elapsed_seconds': time.monotonic()-started,
            'python': platform.python_version(), 'numpy': np.__version__, 'sympy': sp.__version__,
            'cpu_backend': CK.cpu_backend(), 'kernel_loaded': CK.loaded(),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
            'paper_sha256': sha(args.paper) if args.paper else None,
            'source_sha256': {str(p.relative_to(repo)): sha(p) for p in sources},
            'tests_run': result.testsRun, 'passed': result.testsRun-len(result.skipped)-len(result.errors)-len(result.failures),
            'skipped': len(result.skipped), 'failures': len(result.failures), 'errors': len(result.errors),
            'tests': result.records, 'experiments': test_sections.EVIDENCE,
            'scope': 'Finite verification; sparse high levels are not dense GPU execution. Numerical roots are not certified complete root sets.',
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
