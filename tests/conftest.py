# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Shared pytest fixtures and guards for the CayleyKernel test suite."""

import os
import sys

import pytest

# Make the repository root importable regardless of where pytest is launched
# from, so that `import CayleyKernel` resolves to the local package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cupy as cp
except Exception:
    cp = None

import CayleyKernel


def pytest_addoption(parser):
    timing = parser.getgroup("GPU performance baselines")
    timing.addoption("--run-benchmarks", action="store_true",
                     help="Enable GPU timing tests (also choose record or baseline).")
    timing.addoption("--benchmark-record", metavar="PATH",
                     help="Record a new local GPU baseline; never overwrite an existing file.")
    timing.addoption("--benchmark-baseline", metavar="PATH",
                     help="Compare timings against an existing GPU baseline.")
    timing.addoption("--benchmark-max-regression", type=float, default=0.20,
                     help="Allowed median slowdown as a fraction of baseline (default: 0.20).")
    group = parser.getgroup("sparse zero divisors")
    group.addoption("--run-sparse-zd", action="store_true",
                    help="Enable CPU-only sparse zero-divisor validation.")
    group.addoption("--zd-levels", default="4,8,32,128,1024,4096",
                    help="Comma-separated doubling levels (n >= 4).")
    group.addoption("--zd-samples", type=int, default=128,
                    help="XOR-compatible candidates per level (default: 128).")
    group.addoption("--zd-seed", type=int, default=2026,
                    help="Seed for reproducible sparse candidates.")
    group.addoption("--zd-repeats", type=int, default=3,
                    help="Timing rounds per method/workload (default: 3).")


def pytest_generate_tests(metafunc):
    if "zd_level" in metafunc.fixturenames:
        try:
            levels = [int(n) for n in metafunc.config.getoption("--zd-levels").split(",")]
        except ValueError:
            raise pytest.UsageError("--zd-levels must contain comma-separated integers >= 4")
        if not levels or any(n < 4 for n in levels):
            raise pytest.UsageError("--zd-levels must contain comma-separated integers >= 4")
        if metafunc.config.getoption("--zd-samples") < 1:
            raise pytest.UsageError("--zd-samples must be positive")
        if metafunc.config.getoption("--zd-repeats") < 1:
            raise pytest.UsageError("--zd-repeats must be positive")
        metafunc.parametrize("zd_level", levels, ids=lambda n: "level-%d" % n)


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-benchmarks"):
        skip = pytest.mark.skip(reason="opt-in: --run-benchmarks plus --benchmark-record or --benchmark-baseline")
        for item in items:
            if item.get_closest_marker("benchmark"):
                item.add_marker(skip)
    if not config.getoption("--run-sparse-zd"):
        skip = pytest.mark.skip(reason="opt-in: use make test-sparse-zd")
        for item in items:
            if item.get_closest_marker("sparse_zd"):
                item.add_marker(skip)


def pytest_configure(config):
    import math
    enabled = config.getoption("--run-benchmarks")
    record = config.getoption("--benchmark-record")
    baseline = config.getoption("--benchmark-baseline")
    if enabled and bool(record) == bool(baseline):
        raise pytest.UsageError("--run-benchmarks requires exactly one of --benchmark-record PATH or --benchmark-baseline PATH")
    if not enabled and (record or baseline):
        raise pytest.UsageError("Benchmark recording/comparison requires --run-benchmarks")
    if baseline:
        from pathlib import Path
        if not Path(baseline).is_file():
            raise pytest.UsageError(
                "--benchmark-baseline requires an existing JSON file, "
                "not a directory or missing path: " + str(baseline)
                + ". To create a baseline, use --run-benchmarks "
                "--benchmark-record PATH.json instead."
            )
    allowance = config.getoption("--benchmark-max-regression")
    if not math.isfinite(allowance) or allowance < 0:
        raise pytest.UsageError("--benchmark-max-regression must be finite and nonnegative")
    # Register custom markers (also declared in pyproject.toml).
    config.addinivalue_line(
        "markers", "slow: heavy 2^20+ identity/batch tests (pytest -m slow)"
    )
    config.addinivalue_line(
        "markers", "benchmark: GPU timing checks (opt-in: --run-benchmarks with a local baseline)"
    )
    config.addinivalue_line("markers", "sparse_zd: opt-in CPU sparse zero-divisor checks")
    config.addinivalue_line("markers", "cpu: CPU-only checks without a CUDA runtime")


@pytest.fixture(autouse=True)
def _seed(request):
    if (request.node.get_closest_marker("sparse_zd")
            or request.node.get_closest_marker("cpu")
            or request.node.get_closest_marker("benchmark")):
        return
    if np is not None:
        np.random.seed(0)
    if cp is not None:
        cp.random.seed(0)


@pytest.fixture(scope="session")
def gpu_benchmark(request):
    from tests.benchmark_support import BenchmarkSession
    if cp is None or not CayleyKernel.loaded():
        pytest.skip("Benchmark requires CuPy and the compiled kernel")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("Benchmark requires a CUDA device")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip("CUDA device unavailable: " + str(exc))
    session = BenchmarkSession(
        cp, baseline=request.config.getoption("--benchmark-baseline"),
        record=request.config.getoption("--benchmark-record"),
        max_regression=request.config.getoption("--benchmark-max-regression"),
    )
    yield session
    session.finish()


@pytest.fixture(autouse=True)
def _require_gpu_kernel(request):
    """Skip the whole suite when the GPU stack is unavailable.

    A fresh clone has no compiled kernel (it is built with `make kernel`).
    Skipping instead of failing keeps `pytest` green in environments
    without nvcc/CUDA, while the message tells developers exactly what
    to run.
    """
    if request.node.get_closest_marker("sparse_zd") or request.node.get_closest_marker("cpu"):
        return
    if cp is None:
        pytest.skip("CuPy/CUDA runtime not available on this machine.")
    if not CayleyKernel.loaded():
        pytest.skip(
            "cuda_kernel.so not found -- build it first:  make kernel"
        )
