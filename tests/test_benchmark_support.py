"""CPU-only checks of baseline validation, gating and opt-in behavior."""

import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests import benchmark_support as bs

pytestmark = pytest.mark.cpu


def reference():
    return {
        "schema_version": 1,
        "protocol": bs.PROTOCOL.copy(),
        "environment": {key: "fixture" for key in bs.ENVIRONMENT_KEYS},
        "cases": {
            name: {
                "shape": shape,
                "dtype": "float32",
                "repetitions": 1,
                "samples_seconds": [1.0] * 5,
                "median_seconds": 1.0,
            }
            for name, shape in bs.CASE_SHAPES.items()
        },
    }


def test_baseline_rejects_wrong_device_and_inconsistent_samples():
    report = reference()
    bs.validate_baseline(report, report["environment"])
    for bad in ({}, [], dict(report, protocol={})):
        with pytest.raises(ValueError, match="protocol"):
            bs.validate_baseline(bad, report["environment"])
    environment = dict(report["environment"], pci_bus_id="another GPU")
    with pytest.raises(ValueError, match="environment differs"):
        bs.validate_baseline(report, environment)
    for value in (float("nan"), -1, 0, True):
        bad = copy.deepcopy(report)
        bad["cases"]["single_2pow20"]["samples_seconds"][0] = value
        with pytest.raises(ValueError, match="samples"):
            bs.validate_baseline(bad, bad["environment"])
    bad = copy.deepcopy(report)
    bad["cases"]["single_2pow20"]["median_seconds"] = 99
    with pytest.raises(ValueError, match="median"):
        bs.validate_baseline(bad, bad["environment"])
    bad = copy.deepcopy(report)
    del bad["cases"]["single_2pow20"]
    with pytest.raises(ValueError, match="all configured"):
        bs.validate_baseline(bad, bad["environment"])


def test_topology_reports_device_identity_and_directional_peer_access():
    devices = {
        dev: SimpleNamespace(id=dev, pci_bus_id=f"bus-{dev}", compute_capability="90")
        for dev in (0, 1)
    }
    runtime = SimpleNamespace(
        getDeviceProperties=lambda dev: {"name": f"GPU {dev}".encode()},
        deviceCanAccessPeer=lambda src, dst: int(src == 0 and dst == 1),
    )
    cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda dev=0: devices[dev], runtime=runtime))
    topology = bs.device_topology(cp, [0, 1])
    assert topology["caller_device"] == 0
    assert topology["devices"][1] == {
        "gpu_name": "GPU 1",
        "pci_bus_id": "bus-1",
        "compute_capability": "90",
    }
    assert topology["peer_access"] == {"0->1": True, "1->0": False}


def test_paired_samples_are_synchronized_and_alternate_order(monkeypatch):
    events = []
    clock = 0.0

    def synchronize():
        nonlocal clock
        events.append("sync")
        clock += 1.0

    def run(name, duration):
        nonlocal clock
        events.append(name)
        clock += duration

    monkeypatch.setattr(bs.time, "perf_counter", lambda: clock)
    cases = bs.measure_workloads(
        {"a": lambda: run("a", 2.0), "b": lambda: run("b", 4.0)},
        synchronize,
    )
    trials = bs.PROTOCOL["trials"]
    assert cases["a"]["samples_seconds"] == [3.0] * trials
    assert cases["b"]["samples_seconds"] == [5.0] * trials
    order = ["a", "b", "b", "a", "a", "b", "b", "a", "a", "b"]
    assert events[-3 * len(order) :] == [
        event for name in order for event in ("sync", name, "sync")
    ]


def fake_gpu(monkeypatch):
    monkeypatch.setattr(bs, "environment_info", lambda cp: reference()["environment"])
    monkeypatch.setattr(bs, "provenance", lambda: {})
    return SimpleNamespace(
        random=SimpleNamespace(seed=lambda seed: None, rand=lambda *shape, **kw: object()),
        float32="float32",
        cuda=SimpleNamespace(
            Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None))
        ),
    )


def test_measured_regression_fails_without_changing_baseline(tmp_path, monkeypatch):
    path = tmp_path / "baseline.json"
    original = json.dumps(reference())
    path.write_text(original)
    session = bs.BenchmarkSession(fake_gpu(monkeypatch), baseline=path)
    # One calibration and five ten-second samples versus a one-second baseline.
    clock = iter(range(0, 120, 10))
    monkeypatch.setattr(bs.time, "perf_counter", lambda: next(clock))
    with pytest.raises(AssertionError, match="exceeds baseline"):
        session.measure("single_2pow20", lambda *args, **kwargs: object(), (2**20,))
    assert session.report["cases"]["single_2pow20"]["repetitions"] == 1
    session.finish()
    assert path.read_text() == original


def test_short_workloads_amortize_sync_overhead_but_detect_slowdown(tmp_path, monkeypatch):
    cp = fake_gpu(monkeypatch)
    clock = {"now": 0.0, "sync": 5e-6, "work": 1e-5}

    def synchronize():
        clock["now"] += clock["sync"]

    def multiply(*args, **kwargs):
        clock["now"] += clock["work"]
        return object()

    cp.cuda.Stream.null.synchronize = synchronize
    monkeypatch.setattr(bs.time, "perf_counter", lambda: clock["now"])
    session = bs.BenchmarkSession(cp, record=tmp_path / "unused.json")
    session.measure("batch_1024_dim16", multiply, (1024, 16))
    case = session.report["cases"]["batch_1024_dim16"]
    assert case["repetitions"] > 1
    assert case["median_seconds"] == pytest.approx(clock["work"], rel=0.01)

    report = reference()
    report["cases"]["batch_1024_dim16"] = case
    path = tmp_path / "baseline.json"
    original = json.dumps(report)
    path.write_text(original)
    clock["sync"] *= 10
    comparison = bs.BenchmarkSession(cp, baseline=path)
    comparison.measure("batch_1024_dim16", multiply, (1024, 16))
    clock["work"] *= 2
    with pytest.raises(AssertionError, match="exceeds baseline"):
        comparison.measure("batch_1024_dim16", multiply, (1024, 16))
    assert path.read_text() == original


def test_legacy_timing_protocol_requires_new_recording():
    report = reference()
    report["protocol"] = {
        "warmup": 1,
        "trials": 5,
        "statistic": "median",
        "timer": "synchronized_wall_seconds",
        "seed": 2026,
    }
    with pytest.raises(ValueError, match="record a new baseline"):
        bs.validate_baseline(report, report["environment"])


def test_recording_requires_complete_workloads_and_never_overwrites(tmp_path, monkeypatch):
    path = tmp_path / "reference.json"
    session = bs.BenchmarkSession(fake_gpu(monkeypatch), record=path)
    with pytest.raises(ValueError, match="all configured"):
        session.finish()
    assert not path.exists()
    session.report["cases"] = reference()["cases"]
    session.finish()
    assert json.loads(path.read_text())["cases"] == reference()["cases"]
    with pytest.raises(ValueError, match="overwrite"):
        bs.BenchmarkSession(fake_gpu(monkeypatch), record=path)

    late_path = tmp_path / "concurrent.json"
    session = bs.BenchmarkSession(fake_gpu(monkeypatch), record=late_path)
    session.report["cases"] = reference()["cases"]
    late_path.write_text("another recording")
    with pytest.raises(FileExistsError):
        session.finish()
    assert late_path.read_text() == "another recording"


def test_opt_in_requires_flag_and_baseline_mode(tmp_path):
    # Load the repository hooks on CPU-marked synthetic tests, without a GPU.
    path = tmp_path / "test_opt_in.py"
    path.write_text(
        "import pytest\n"
        "pytestmark = [pytest.mark.cpu, pytest.mark.benchmark]\n"
        "@pytest.mark.benchmark_baseline\n"
        "def test_baseline_timing():\n    assert True\n"
        "def test_paired_timing():\n    assert True\n"
    )
    base = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-c",
        str(bs.ROOT / "pyproject.toml"),
        "-p",
        "tests.conftest",
        "--confcutdir",
        str(tmp_path),
        "-p",
        "no:cacheprovider",
        str(path),
    ]
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(reference()))
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    for options, expected, message in (
        ([], 0, "2 skipped"),
        (["-m", "cpu and not slow"], 0, "2 skipped"),
        (["-m", ""], 0, "2 skipped"),
        (["-m", "benchmark"], 0, "2 skipped"),
        (["--run-benchmarks"], 4, "exactly one"),
        (["--run-benchmarks", "-m", "benchmark and not benchmark_baseline"], 0, "1 passed"),
        (
            ["--run-benchmarks", "--benchmark-baseline", str(baseline), "-k", "paired"],
            4,
            "requires selecting baseline workloads",
        ),
        (
            ["--run-benchmarks", "--benchmark-record", "a", "--benchmark-baseline", "b"],
            4,
            "exactly one",
        ),
        (["--benchmark-max-regression", "nan"], 4, "finite and nonnegative"),
        (["--benchmark-record", str(tmp_path / "out.json")], 4, "requires --run-benchmarks"),
        (
            ["--run-benchmarks", "--benchmark-baseline", str(tmp_path)],
            4,
            "requires an existing JSON file",
        ),
        (
            ["--run-benchmarks", "--benchmark-baseline", str(tmp_path / "missing.json")],
            4,
            "--benchmark-record PATH.json",
        ),
        (["--run-benchmarks", "--benchmark-baseline", str(baseline)], 0, "2 passed"),
        (["--run-benchmarks", "--benchmark-record", str(baseline)], 4, "Refusing to overwrite"),
        (["--run-benchmarks", "--benchmark-record", str(tmp_path / "out.json")], 0, "2 passed"),
    ):
        result = subprocess.run(
            base + options, cwd=bs.ROOT, env=env, text=True, capture_output=True
        )
        assert result.returncode == expected, result.stdout + result.stderr
        assert message in result.stdout + result.stderr
