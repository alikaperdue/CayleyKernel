# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Shared pytest fixtures and guards for the CayleyKernel test suite."""

import math
from pathlib import Path

import numpy as np
import pytest

import CayleyKernel
from tests.support.gpu_support import GPUUnavailable, check_compiled_kernel, device_count


def pytest_addoption(parser):
    parser.addoption(
        "--require-gpu",
        action="store_true",
        help="Fail selected GPU tests if their device or compiled kernel is unavailable.",
    )
    timing = parser.getgroup("GPU performance")
    timing.addoption(
        "--run-benchmarks",
        action="store_true",
        help="Enable selected GPU timing tests. Baseline workloads also need record or baseline.",
    )
    timing.addoption(
        "--benchmark-record",
        metavar="PATH",
        help="Record a new local GPU baseline; never overwrite an existing file.",
    )
    timing.addoption(
        "--benchmark-baseline",
        metavar="PATH",
        help="Compare timings against an existing GPU baseline.",
    )
    timing.addoption(
        "--benchmark-max-regression",
        type=float,
        default=0.20,
        help="Allowed median slowdown as a fraction of baseline (default: 0.20).",
    )
    group = parser.getgroup("sparse zero divisors")
    group.addoption(
        "--run-sparse-zd",
        action="store_true",
        help="Enable CPU-only sparse zero-divisor validation.",
    )
    group.addoption(
        "--zd-levels",
        default="4,8,32,128,1024,4096",
        help="Comma-separated doubling levels (n >= 4).",
    )
    group.addoption(
        "--zd-samples",
        type=int,
        default=128,
        help="XOR-compatible candidates per level (default: 128).",
    )
    group.addoption(
        "--zd-seed", type=int, default=2026, help="Seed for reproducible sparse candidates."
    )
    group.addoption(
        "--zd-repeats", type=int, default=3, help="Timing rounds per method/workload (default: 3)."
    )


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


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-benchmarks"):
        skip = pytest.mark.skip(reason="opt-in: --run-benchmarks (see docs/testing.md)")
        for item in items:
            if item.get_closest_marker("benchmark"):
                item.add_marker(skip)
    else:
        baseline_workloads = any(item.get_closest_marker("benchmark_baseline") for item in items)
        baseline_mode = config.getoption("--benchmark-record") or config.getoption(
            "--benchmark-baseline"
        )
        if baseline_workloads and not baseline_mode:
            raise pytest.UsageError(
                "Selected baseline workloads require exactly one of "
                "--benchmark-record PATH or --benchmark-baseline PATH"
            )
        if baseline_mode and not baseline_workloads:
            raise pytest.UsageError(
                "Benchmark recording/comparison requires selecting baseline workloads "
                "(see tests/performance/test_benchmark.py)"
            )
    if not config.getoption("--run-sparse-zd"):
        skip = pytest.mark.skip(reason="opt-in: use make test-sparse-zd")
        for item in items:
            if item.get_closest_marker("sparse_zd"):
                item.add_marker(skip)


def pytest_configure(config):
    enabled = config.getoption("--run-benchmarks")
    record = config.getoption("--benchmark-record")
    baseline = config.getoption("--benchmark-baseline")
    if record and baseline:
        raise pytest.UsageError(
            "Choose exactly one of --benchmark-record PATH or --benchmark-baseline PATH"
        )
    if not enabled and (record or baseline):
        raise pytest.UsageError("Benchmark recording/comparison requires --run-benchmarks")
    if baseline and not Path(baseline).is_file():
        raise pytest.UsageError(
            "--benchmark-baseline requires an existing JSON file, "
            "not a directory or missing path: "
            + str(baseline)
            + ". To create a baseline, use --run-benchmarks "
            "--benchmark-record PATH.json instead."
        )
    if record and Path(record).exists():
        raise pytest.UsageError("Refusing to overwrite an existing baseline: " + str(record))
    allowance = config.getoption("--benchmark-max-regression")
    if not math.isfinite(allowance) or allowance < 0:
        raise pytest.UsageError("--benchmark-max-regression must be finite and nonnegative")


def _gpu_unavailable(config, reason):
    if config.getoption("--require-gpu"):
        pytest.fail(f"--require-gpu: {reason}", pytrace=False)
    pytest.skip(str(reason))


@pytest.fixture(scope="session")
def gpu_device_count(pytestconfig):
    try:
        return device_count()
    except GPUUnavailable as exc:
        _gpu_unavailable(pytestconfig, exc)


@pytest.fixture(scope="session")
def compiled_kernel():
    """Host-only ABI checks can use the library without a device."""
    try:
        check_compiled_kernel()
    except GPUUnavailable as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="session")
def gpu_kernel(pytestconfig, gpu_device_count):
    try:
        check_compiled_kernel()
    except GPUUnavailable as exc:
        _gpu_unavailable(pytestconfig, exc)
    return CayleyKernel.cp


@pytest.fixture(scope="module")
def two_gpu_devices(gpu_device_count):
    if gpu_device_count < 2:
        pytest.skip("Multi-GPU tests require at least two visible devices")


@pytest.fixture
def torch_cuda(pytestconfig, gpu_kernel):
    import torch

    if not torch.cuda.is_available():
        _gpu_unavailable(pytestconfig, "PyTorch CUDA is unavailable")


@pytest.fixture(autouse=True)
def _prepare_gpu_test(request):
    if request.node.get_closest_marker("cpu"):
        return
    if request.node.get_closest_marker("gpu_device"):
        request.getfixturevalue("gpu_device_count")
    else:
        request.getfixturevalue("gpu_kernel")
    if not request.node.get_closest_marker("benchmark"):
        np.random.seed(0)
        CayleyKernel.cp.random.seed(0)


@pytest.fixture(scope="session")
def gpu_benchmark(request, gpu_kernel):
    from tests.performance.benchmark_support import BenchmarkSession

    session = BenchmarkSession(
        gpu_kernel,
        baseline=request.config.getoption("--benchmark-baseline"),
        record=request.config.getoption("--benchmark-record"),
        max_regression=request.config.getoption("--benchmark-max-regression"),
    )
    yield session
    if not request.session.testsfailed:
        session.finish()
