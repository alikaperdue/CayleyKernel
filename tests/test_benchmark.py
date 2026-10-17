# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Benchmark gates: timing sanity for the O(1) Cayley-Dickson kernel.

Excluded from the default run (marked ``benchmark`` in pyproject.toml).
Run with:

    pytest -q -m benchmark tests/test_benchmark.py

The gates are deliberately generous (machine-dependent GPU speeds vary a
lot); their purpose is to catch order-of-magnitude regressions -- e.g. a
kernel accidentally falling back to O(dim^2) behaviour -- not to enforce
micro-benchmark stability.
"""

import time

import pytest

cp = pytest.importorskip("cupy")

from CayleyKernel import multiply  # noqa: E402


def _timed(func, *args, warmup=1, trials=3):
    for _ in range(warmup):
        func(*args)
    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()
    for _ in range(trials):
        func(*args)
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - start) / trials


@pytest.mark.benchmark
def test_single_product_2pow20_under_budget():
    """One float32 product at 2^20 must finish well under 5 seconds.

    On any remotely modern GPU this is a few milliseconds; the 5 s gate is
    pure regression armour.
    """
    dim = 2**20
    x = cp.random.rand(dim, dtype=cp.float32)
    y = cp.random.rand(dim, dtype=cp.float32)

    elapsed = _timed(multiply, x, y)
    print(f"\n  single 2^20 float32 product: {elapsed*1000:.3f} ms")
    assert elapsed < 5.0, f"2^20 product too slow: {elapsed:.3f}s"


@pytest.mark.benchmark
def test_batched_1024x_dim16_under_budget():
    """1024 independent sedenion products should be effectively one launch."""
    A = cp.random.rand(1024, 16, dtype=cp.float32)
    B = cp.random.rand(1024, 16, dtype=cp.float32)

    elapsed = _timed(multiply, A, B)
    print(f"\n  batched (1024, 16) float32: {elapsed*1000:.3f} ms")
    assert elapsed < 2.0, f"batched products too slow: {elapsed:.3f}s"


@pytest.mark.benchmark
def test_batch_scaling_sublinear_per_item():
    """Per-item cost should not explode as the batch grows (grid-stride).

    Doubling the batch may add a little launch overhead amortisation noise,
    but the per-item time at 64x the batch must stay within 10x of the
    small-batch per-item time -- O(1)-order behaviour.
    """
    dim = 1024
    A1 = cp.random.rand(16, dim, dtype=cp.float32)
    B1 = cp.random.rand(16, dim, dtype=cp.float32)
    A2 = cp.random.rand(1024, dim, dtype=cp.float32)
    B2 = cp.random.rand(1024, dim, dtype=cp.float32)

    t_small = _timed(multiply, A1, B1) / 16
    t_large = _timed(multiply, A2, B2) / 1024

    print(f"\n  per-item dim=1024: batch=16 -> {t_small*1e6:.2f} us, "
          f"batch=1024 -> {t_large*1e6:.2f} us")
    assert t_large < 10.0 * t_small + 1e-6, (
        f"per-item cost degraded: {t_large:.6g}s vs {t_small:.6g}s"
    )
