# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Tests for the sparse (CSR-style) Cayley-Dickson product.

Deliverable of ``feat/sparse-csr`` (Phases A–C): design document,
Python API (:mod:`CayleyKernel._sparse`), and the dedicated CUDA
kernel (``sparse_mult_<suffix>`` in ``cuda/cuda_kernel.cu``, ABI 5).
See ``docs/sparse-csr-design.md`` for the design and
``tests/test_sparse_zd.py`` for the algebraic evidence.

The sparse API dispatches to the GPU kernel when the library exposes
the Phase C symbols for the operand dtype (float16 / bfloat16 /
float32 / float64 / complex64 / complex128) and otherwise falls back
transparently to a bit-for-bit CPU reference, so the tests remain
meaningful on hosts without a GPU.

Run with::

    pytest -q tests/test_sparse_csr.py
"""

import os
import sys
import time

import numpy as np
import pytest

# Make the repository root importable regardless of where pytest is launched
# from, so that `import CayleyKernel` resolves to the local package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import CayleyKernel
from CayleyKernel import (  # noqa: E402
    from_sparse,
    is_sparse_beneficial,
    sparse_multiply,
    to_sparse,
)

cp = pytest.importorskip("cupy", reason="sparse API tests require CuPy")

# Opt out of the _require_gpu_kernel autouse fixture (most tests are
# CPU-side and do not need cuda_kernel.so).  The one benchmark that does
# needs it guards itself with a pytest.skip.
pytestmark = pytest.mark.cpu


def _dense_ref(a, b):
    """Dense reference product.

    Uses the GPU kernel when available (fast, batched); otherwise falls
    back to the pure-Python ``py_multiply`` row-by-row.
    """
    if CayleyKernel.loaded():
        return CayleyKernel.multiply(a, b)
    # CPU fallback: py_multiply is 1D; loop over the batch axis.
    a_np = cp.asnumpy(a)
    b_np = cp.asnumpy(b)
    if a_np.ndim == 1:
        out = CayleyKernel.py_multiply(a_np, b_np)
        return cp.asarray(out, dtype=a.dtype)
    # Batched: loop rows.
    batch = a_np.shape[0]
    dim = a_np.shape[1]
    out = np.zeros((batch, dim), dtype=np.float64)
    for r in range(batch):
        out[r] = CayleyKernel.py_multiply(a_np[r], b_np[r])
    return cp.asarray(out, dtype=a.dtype)


def _make_dense_batch(batch, dim, K, seed, dtype):
    """Return a CuPy array of shape ``(batch, dim)`` with exactly K
    nonzeros per row (chosen deterministically from ``seed``)."""
    rng = np.random.RandomState(seed)
    rows = np.zeros((batch, dim), dtype=np.float64)
    for r in range(batch):
        idx = rng.choice(dim, size=K, replace=False)
        for i, c in enumerate(idx):
            val = rng.randn() * (1.0 + 0.1 * i)
            if (i % 2) == 1:
                val = -val
            if val == 0:
                val = 1.0
            rows[r, int(c)] = val
    return cp.asarray(rows.astype(dtype))


def _known_zero_divisor_supports(dim):
    """A pair of two-term supports known to produce a zero product.

    Lifted from ``tests/test_sparse_zd.py::test_embedded_known_witness``
    (level 4 / dim 16).  All indices are < 16, so the sign rule and XOR
    are identical for any dim ≥ 16 (high bits are zero on both sides).
    """
    x = np.zeros(dim, dtype=np.float64)
    y = np.zeros(dim, dtype=np.float64)
    x[3] = 1.0
    x[10] = 1.0
    y[6] = 1.0
    y[15] = -1.0
    return x, y


# ---------------------------------------------------------------------------
# Tests: round-trip to_sparse / from_sparse (no dense kernel needed)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [256, 1024])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_roundtrip_dense_to_sparse_and_back(dim, dtype):
    """Round-trip a dense vector (and a batch of K-sparse rows) through
    to_sparse / from_sparse and require exact recovery."""
    rng = np.random.RandomState(0)
    x_single = cp.asarray(rng.randn(dim).astype(dtype))
    sv = to_sparse(x_single)
    back = cp.asnumpy(from_sparse(sv)).reshape(-1)
    np.testing.assert_array_equal(back, cp.asnumpy(x_single).reshape(-1))

    for K in (1, 2, 4, 8):
        x_batch = _make_dense_batch(batch=16, dim=dim, K=K, seed=K, dtype=dtype)
        sv = to_sparse(x_batch)
        back = cp.asnumpy(from_sparse(sv))
        orig = cp.asnumpy(x_batch)
        np.testing.assert_array_equal(back, orig)


# ---------------------------------------------------------------------------
# Tests: zero-divisor cases (no dense kernel needed)
# ---------------------------------------------------------------------------
def test_sparse_product_known_zero_divisor():
    """The embedded witness from tests/test_sparse_zd.py must produce
    an empty support (K == 0) on the sparse path."""
    dim = 16
    x, y = _known_zero_divisor_supports(dim)
    xs = to_sparse(cp.asarray(x, dtype=cp.float64))
    ys = to_sparse(cp.asarray(y, dtype=cp.float64))
    got = sparse_multiply(xs, ys)
    assert got.K == 0
    assert got.support.size == 0
    # Round-tripping the zero product back to dense must also be all-zeros.
    np.testing.assert_array_equal(
        cp.asnumpy(from_sparse(got)), np.zeros((1, dim), dtype=np.float64)
    )


def test_sparse_product_zero_divisor_batch():
    """A batch of the known witness must yield K_out == 0 across the batch."""
    dim = 16
    batch = 4
    x, y = _known_zero_divisor_supports(dim)
    x_b = cp.asarray(np.tile(x, (batch, 1)), dtype=cp.float64)
    y_b = cp.asarray(np.tile(y, (batch, 1)), dtype=cp.float64)
    got = sparse_multiply(to_sparse(x_b), to_sparse(y_b))
    assert got.K == 0
    assert got.support.shape == (batch, 0)


# ---------------------------------------------------------------------------
# Tests: sparse product equals dense product (all supported dtypes)
# ---------------------------------------------------------------------------
# bfloat16 is not exposed by every CuPy build (notably the older 11.x line
# that predates bfloat16 promotion).  Detect and skip it gracefully rather
# than failing collection; the design doc's supported-dtype list still
# includes bfloat16 where it is available.
#
# complex64 / complex128 are part of SPARSE_SUPPORTED_DTYPES (Phase C
# exports sparse_mult_complex64 and sparse_mult_complex128) and the
# Phase B spec requires the parity test to cover every supported
# floating dtype, so they are included here as well.
_SPARSE_DTYPES = [cp.float16, cp.float32, cp.float64, cp.complex64, cp.complex128]
if hasattr(cp, "bfloat16"):
    _SPARSE_DTYPES.append(cp.bfloat16)


@pytest.mark.slow
@pytest.mark.parametrize(
    "config",
    [
        {"dim": 1024, "K": 8, "batch": 4096, "label": "memory-bound (batch=4096)"},
        {"dim": 4096, "K": 4, "batch": 256, "label": "compute-bound (dim=4096)"},
        {"dim": 8192, "K": 4, "batch": 128, "label": "compute-memory-bound (dim=8192)"},
        {"dim": 16384, "K": 4, "batch": 64, "label": "compute-memory-bound (dim=16384)"},
        {"dim": 32768, "K": 4, "batch": 16, "label": "compute-memory-bound (dim=32786)"},
        {"dim": 131072, "K": 4, "batch": 2, "label": "compute-memory-bound (dim=131072)"},
    ],
    ids=lambda c: c["label"],
)
def test_benchmark(config):
    dim = config["dim"]
    K = config["K"]
    batch = config["batch"]
    label = config["label"]

    rng = np.random.default_rng(2026)

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

    # Warmup
    CayleyKernel.multiply(x_gpu, y_gpu)
    sparse_multiply(x_sp, y_sp)
    cp.cuda.Stream.null.synchronize()

    # Dense timing
    n = 5
    t0 = time.perf_counter()
    for _ in range(n):
        CayleyKernel.multiply(x_gpu, y_gpu)
    cp.cuda.Stream.null.synchronize()
    t_dense = (time.perf_counter() - t0) / n

    # Sparse timing
    t0 = time.perf_counter()
    for _ in range(n):
        sparse_multiply(x_sp, y_sp)
    cp.cuda.Stream.null.synchronize()
    t_sparse = (time.perf_counter() - t0) / n

    speedup = t_dense / t_sparse
    print(f"\n[benchmark] {label}")
    print(f"[benchmark]   dim={dim}  K={K}  batch={batch}  dtype=float32")
    print(f"[benchmark]   dense : {t_dense * 1e3:>10.3f} ms")
    print(f"[benchmark]   sparse: {t_sparse * 1e3:>10.3f} ms")
    print(f"[benchmark]   ratio dense/sparse: {speedup:>6.2f}x")
    assert t_dense > 0 and t_sparse > 0


@pytest.mark.parametrize("K", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", _SPARSE_DTYPES)
def test_sparse_multiply_matches_dense(K, dtype):
    """For dim=16 (level 4), the sparse reference must reproduce the
    dense product within dtype-appropriate tolerance.

    The sparse path accumulates in float64; the dense path accumulates
    in the dtype's native precision (float32 for float16/bfloat16,
    own type otherwise).
    """
    dim = 16
    batch = 8
    seed = 1000 + K
    x = _make_dense_batch(batch, dim, K, seed, dtype)
    y = _make_dense_batch(batch, dim, K, seed + 1, dtype)

    xs = to_sparse(x)
    ys = to_sparse(y)
    got = sparse_multiply(xs, ys)
    want = _dense_ref(x, y)

    got_np = cp.asnumpy(from_sparse(got).reshape(-1, dim)).astype(np.float64)
    want_np = cp.asnumpy(want).reshape(-1, dim).astype(np.float64)

    if dtype == cp.float16:
        rtol, atol = 1e-2, 1e-2
    elif dtype == cp.float32 or dtype == cp.complex64:
        rtol, atol = 1e-4, 1e-5
    else:  # float64 / complex128
        rtol, atol = 1e-10, 1e-12
    np.testing.assert_allclose(got_np, want_np, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Tests: is_sparse_beneficial heuristic
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dim, k_a, k_b, expected",
    [
        (256, 2, 2, True),  # 4 < 85.3
        (256, 4, 4, True),  # 16 < 85.3
        (256, 8, 8, True),  # 64 < 85.3
        (256, 16, 16, False),  # 256 > 85.3
        (1024, 8, 8, True),  # 64 < 341.3
        (1024, 16, 16, True),  # 256 < 341.3
        (1024, 32, 32, False),  # 1024 > 341.3
        (4096, 16, 16, True),  # 256 < 1365.3
        (4096, 32, 32, True),  # 1024 < 1365.3
        (4096, 64, 64, False),  # 4096 > 1365.3
    ],
)
def test_is_sparse_beneficial_crossover(dim, k_a, k_b, expected):
    """The heuristic must implement the design-doc crossover
    ``K* ≈ sqrt(dim/α)`` with α = 3.0, i.e. sparse wins when
    ``k_a * k_b < dim / 3``."""
    assert is_sparse_beneficial(dim, k_a, k_b) == expected
