# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Opt-in GPU measurements against an explicitly recorded local baseline.

The sign evaluator is O(1); dense multiplication still evaluates dim terms
for each of dim outputs: O(batch * dim**2) total work. At dim=2**20 this is
2**40 terms, so a universal millisecond expectation or five-second gate is
not justified. GPU parallelism does not make the whole product O(1).

Default test runs skip these measurements. Record a baseline explicitly:
    pytest tests/test_benchmark.py --run-benchmarks \
        --benchmark-record benchmark-results/gpu.json -s
Compare on the same GPU/software environment:
    pytest tests/test_benchmark.py --run-benchmarks \
        --benchmark-baseline benchmark-results/gpu.json -s

Each workload has three warmups followed by calibration to at least 100 ms
per sample (capped at 16384 calls). Five synchronized wall-clock samples
report time per call, amortizing synchronization overhead for short workloads.
Long workloads use one call per sample. The median includes Python wrapper,
allocation and release overhead, but input creation is outside the timing.
Old single-call timing baselines must be recorded again under this protocol,
using a new filename to preserve the original. Comparisons permit 20% over
the baseline by default, configurable with --benchmark-max-regression. This is a local
regression policy, not a hardware-independent performance guarantee or a
proof of asymptotic scaling. Baselines are never updated by comparison.
"""

import pytest

cp = pytest.importorskip("cupy")
from CayleyKernel import multiply

pytestmark = pytest.mark.benchmark


def test_single_product_2pow20(gpu_benchmark):
    """Measure a dense float32 product containing 2**40 pair contributions."""
    gpu_benchmark.measure("single_2pow20", multiply, (2**20,))


def test_batched_1024x_dim16(gpu_benchmark):
    """Measure 1024 independent sedenion products in a batched launch."""
    gpu_benchmark.measure("batch_1024_dim16", multiply, (1024, 16))


def test_batch_scaling(gpu_benchmark):
    """Compare each batch size to its own reference, reporting per-item cost."""
    small = gpu_benchmark.measure("batch_16_dim1024", multiply, (16, 1024))
    large = gpu_benchmark.measure("batch_1024_dim1024", multiply, (1024, 1024))
    print(f"\nPer item: batch 16={small / 16:.6g}s; batch 1024={large / 1024:.6g}s")


@pytest.mark.parametrize('batch,dim', [(4096, 8), (16384, 16), (65535, 16), (70000, 4)])
def test_packed_small_vectors(gpu_benchmark, batch, dim):
    """Track shared-memory tiles over mid-range and large batches."""
    gpu_benchmark.measure(f"batch_{batch}_dim{dim}", multiply, (batch, dim))
