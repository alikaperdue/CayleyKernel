# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Paired GPU timings for sparse and dense multiplication."""

import numpy as np
import pytest

import CayleyKernel
from CayleyKernel import cp, sparse_multiply, to_sparse
from CayleyKernel._sparse import _sparse_kernel_handle
from tests.performance.benchmark_support import PROTOCOL, measure_workloads


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "config",
    [
        {"dim": 1024, "K": 8, "batch": 4096, "label": "memory-bound (batch=4096)"},
        {"dim": 4096, "K": 4, "batch": 256, "label": "compute-bound (dim=4096)"},
        {"dim": 8192, "K": 4, "batch": 128, "label": "compute-memory-bound (dim=8192)"},
        {"dim": 16384, "K": 4, "batch": 64, "label": "compute-memory-bound (dim=16384)"},
        {"dim": 32768, "K": 4, "batch": 16, "label": "compute-memory-bound (dim=32768)"},
        {"dim": 131072, "K": 4, "batch": 2, "label": "compute-memory-bound (dim=131072)"},
    ],
    ids=lambda c: c["label"],
)
def test_benchmark(config):
    """Compare products on inputs already in each path's native layout."""
    if _sparse_kernel_handle("float32") is None:
        pytest.skip("Sparse versus dense timing requires the float32 sparse GPU kernel")

    dim = config["dim"]
    K = config["K"]
    batch = config["batch"]
    label = config["label"]

    rng = np.random.default_rng(PROTOCOL["seed"])

    # Build sparse inputs with exactly K nonzeros
    x_dense = np.zeros((batch, dim), dtype=np.float32)
    y_dense = np.zeros((batch, dim), dtype=np.float32)
    for i in range(batch):
        x_idx = rng.choice(dim, K, replace=False)
        y_idx = rng.choice(dim, K, replace=False)
        x_dense[i, x_idx] = rng.standard_normal(K)
        y_dense[i, y_idx] = rng.standard_normal(K)

    x_gpu = cp.asarray(x_dense)
    y_gpu = cp.asarray(y_dense)
    x_sp = to_sparse(x_gpu)
    y_sp = to_sparse(y_gpu)

    print(f"\n{label}: dim={dim}, K={K}, batch={batch}, dtype=float32")
    cases = measure_workloads(
        {
            "dense": lambda: CayleyKernel.multiply(x_gpu, y_gpu),
            "sparse": lambda: sparse_multiply(x_sp, y_sp),
        },
        cp.cuda.Stream.null.synchronize,
    )
    ratio = cases["dense"]["median_seconds"] / cases["sparse"]["median_seconds"]
    print(f"Dense/sparse median ratio: {ratio:.2f}x (informational)")
