"""CPU-only checks of baseline validation, gating and opt-in behavior."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from tests import benchmark_support as bs

pytestmark = pytest.mark.cpu


def reference():
    return {
        "schema_version": 1, "protocol": bs.PROTOCOL.copy(),
        "environment": {key: "fixture" for key in bs.ENVIRONMENT_KEYS},
        "cases": {name: {"shape": shape, "dtype": "float32",
                         "repetitions": 1,
                         "samples_seconds": [1.0] * 5, "median_seconds": 1.0}
                  for name, shape in bs.CASE_SHAPES.items()},
    }


def test_baseline_rejects_wrong_device_and_inconsistent_samples():
    report = reference()
    bs.validate_baseline(report, report['environment'])
    for bad in ({}, [], dict(report, protocol={})):
        with pytest.raises(ValueError, match='protocol'):
            bs.validate_baseline(bad, report['environment'])
    environment = dict(report['environment'], pci_bus_id='another GPU')
    with pytest.raises(ValueError, match='environment differs'):
        bs.validate_baseline(report, environment)
    for value in (float('nan'), -1, 0):
        bad = copy.deepcopy(report)
        bad['cases']['single_2pow20']['samples_seconds'][0] = value
        with pytest.raises(ValueError, match='samples'):
            bs.validate_baseline(bad, bad['environment'])
    bad = copy.deepcopy(report)
    bad['cases']['single_2pow20']['median_seconds'] = 99
    with pytest.raises(ValueError, match='median'):
        bs.validate_baseline(bad, bad['environment'])
    bad = copy.deepcopy(report)
    del bad['cases']['single_2pow20']
    with pytest.raises(ValueError, match='all configured'):
        bs.validate_baseline(bad, bad['environment'])


def fake_gpu(monkeypatch):
    monkeypatch.setattr(bs, 'environment_info', lambda cp: reference()['environment'])
    monkeypatch.setattr(bs, 'provenance', lambda: {})
    return SimpleNamespace(
        random=SimpleNamespace(seed=lambda seed: None, rand=lambda *shape, **kw: object()),
        float32='float32',
        cuda=SimpleNamespace(Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None))),
    )


def test_measured_regression_fails_without_changing_baseline(tmp_path, monkeypatch):
    path = tmp_path / 'baseline.json'
    original = json.dumps(reference())
    path.write_text(original)
    session = bs.BenchmarkSession(fake_gpu(monkeypatch), baseline=path)
    # One calibration and five ten-second samples versus a one-second baseline.
    clock = iter(range(0, 120, 10))
    monkeypatch.setattr(bs.time, 'perf_counter', lambda: next(clock))
    with pytest.raises(AssertionError, match='exceeds baseline'):
        session.measure('single_2pow20', lambda x, y: object(), (2**20,))
    assert session.report['cases']['single_2pow20']['repetitions'] == 1
    session.finish()
    assert path.read_text() == original


def test_short_workloads_amortize_sync_overhead_but_detect_slowdown(tmp_path, monkeypatch):
    cp = fake_gpu(monkeypatch)
    clock = {'now': 0.0, 'sync': 5e-6, 'work': 1e-5}

    def synchronize():
        clock['now'] += clock['sync']

    def multiply(x, y):
        clock['now'] += clock['work']
        return object()

    cp.cuda.Stream.null.synchronize = synchronize
    monkeypatch.setattr(bs.time, 'perf_counter', lambda: clock['now'])
    session = bs.BenchmarkSession(cp, record=tmp_path/'unused.json')
    session.measure('batch_1024_dim16', multiply, (1024, 16))
    case = session.report['cases']['batch_1024_dim16']
    assert case['repetitions'] > 1
    assert case['median_seconds'] == pytest.approx(clock['work'], rel=0.01)

    report = reference()
    report['cases']['batch_1024_dim16'] = case
    path = tmp_path/'baseline.json'
    original = json.dumps(report)
    path.write_text(original)
    clock['sync'] *= 10
    comparison = bs.BenchmarkSession(cp, baseline=path)
    comparison.measure('batch_1024_dim16', multiply, (1024, 16))
    clock['work'] *= 2
    with pytest.raises(AssertionError, match='exceeds baseline'):
        comparison.measure('batch_1024_dim16', multiply, (1024, 16))
    assert path.read_text() == original


def test_legacy_timing_protocol_requires_new_recording():
    report = reference()
    report['protocol'] = {
        'warmup': 1, 'trials': 5, 'statistic': 'median',
        'timer': 'synchronized_wall_seconds', 'seed': 2026,
    }
    with pytest.raises(ValueError, match='record a new baseline'):
        bs.validate_baseline(report, report['environment'])


def test_recording_requires_complete_workloads_and_never_overwrites(tmp_path, monkeypatch):
    path = tmp_path / 'reference.json'
    session = bs.BenchmarkSession(fake_gpu(monkeypatch), record=path)
    with pytest.raises(ValueError, match='all configured'):
        session.finish()
    assert not path.exists()
    session.report['cases'] = reference()['cases']
    session.finish()
    assert json.loads(path.read_text())['cases'] == reference()['cases']
    with pytest.raises(ValueError, match='overwrite'):
        bs.BenchmarkSession(fake_gpu(monkeypatch), record=path)


def test_opt_in_requires_flag_and_baseline_mode(tmp_path):
    # Load the repository hooks on CPU-marked synthetic tests, without a GPU.
    path = tmp_path / 'test_opt_in.py'
    path.write_text('import pytest\n@pytest.mark.cpu\n@pytest.mark.benchmark\n'
                    'def test_timing():\n    assert True\n')
    base = [sys.executable, '-m', 'pytest', '-q', '-p', 'tests.conftest',
            '--confcutdir', str(tmp_path), '-p', 'no:cacheprovider', str(path)]
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps(reference()))
    for options, expected, message in (
        ([], 0, '1 skipped'),
        (['-m', 'benchmark'], 0, '1 skipped'),
        (['--run-benchmarks'], 4, 'requires exactly one'),
        (['--run-benchmarks', '--benchmark-record', 'a', '--benchmark-baseline', 'b'],
         4, 'requires exactly one'),
        (['--benchmark-max-regression', 'nan'], 4, 'finite and nonnegative'),
        (['--benchmark-record', str(tmp_path/'out.json')], 4, 'requires --run-benchmarks'),
        (['--run-benchmarks', '--benchmark-baseline', str(tmp_path)],
         4, 'requires an existing JSON file'),
        (['--run-benchmarks', '--benchmark-baseline', str(tmp_path/'missing.json')],
         4, '--benchmark-record PATH.json'),
        (['--run-benchmarks', '--benchmark-baseline', str(baseline)], 0, '1 passed'),
        (['--run-benchmarks', '--benchmark-record', str(tmp_path/'out.json')], 0, '1 passed'),
    ):
        result = subprocess.run(base + options, cwd=bs.ROOT, text=True, capture_output=True)
        assert result.returncode == expected, result.stdout + result.stderr
        assert message in result.stdout + result.stderr
