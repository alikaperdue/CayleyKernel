# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Multi-GPU split_batch correctness tests.

These tests are skipped automatically when fewer than two CUDA devices
are visible (``visible_devices() < 2``), so single-GPU CI runs stay green.
"""

import numpy as np
import pytest

import CayleyKernel as ck
from CayleyKernel import _distributed, cp

pytestmark = pytest.mark.usefixtures("two_gpu_devices")


def _rng(batch, dim, dtype, seed=2026):
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "c":
        real = rng.normal(size=(batch, dim))
        imag = rng.normal(size=(batch, dim))
        return (real + 1j * imag).astype(np.dtype(dtype))
    return rng.normal(size=(batch, dim)).astype(np.dtype(dtype))


# ---------------------------------------------------------------------------
# correctness: split_batch(a, b) == multiply(a, b)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [8, 32])
@pytest.mark.parametrize("batch", [1000, 65536])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_split_batch_matches_multiply(dim, batch, dtype):
    a_np = _rng(batch, dim, dtype, seed=batch)
    b_np = _rng(batch, dim, dtype, seed=batch + 1)

    a = cp.asarray(a_np)
    b = cp.asarray(b_np)

    expected = ck.multiply(a, b)
    actual = _distributed.split_batch(a, b)

    np.testing.assert_allclose(
        cp.asnumpy(actual),
        cp.asnumpy(expected),
        rtol=1e-5,
        atol=1e-5,
        err_msg="dim=%d batch=%d dtype=%s" % (dim, batch, dtype),
    )


def test_split_batch_preserves_original_batch_order():
    """Verify that the output row i equals multiply(a[i], b[i]) for all i."""
    dim = 8
    batch = 100
    rng = np.random.default_rng(7)
    a_np = rng.normal(size=(batch, dim)).astype(np.float32)
    b_np = rng.normal(size=(batch, dim)).astype(np.float32)
    a = cp.asarray(a_np)
    b = cp.asarray(b_np)

    out = _distributed.split_batch(a, b)
    out_np = cp.asnumpy(out)

    # Spot-check several rows (including first, last, and middle).
    for i in [0, batch // 2, batch - 1]:
        single = ck.multiply(a[i : i + 1], b[i : i + 1])
        np.testing.assert_allclose(
            out_np[i],
            cp.asnumpy(single)[0],
            rtol=1e-5,
            atol=1e-5,
            err_msg="row %d out of %d" % (i, batch),
        )


def test_visible_devices_returns_sorted_ints():
    devs = _distributed.visible_devices()
    assert isinstance(devs, list)
    assert all(isinstance(d, int) for d in devs)
    assert devs == sorted(devs)


def test_split_batch_rejects_bad_axis():
    a = cp.zeros((10, 8), dtype=cp.float32)
    b = cp.zeros((10, 8), dtype=cp.float32)
    with pytest.raises(ValueError, match="axis"):
        _distributed.split_batch(a, b, axis=1)


def test_split_batch_out_buffer():
    """The out= buffer path reuses the caller's array and preserves shape."""
    dim = 16
    batch = 256
    rng = np.random.default_rng(3)
    a_np = rng.normal(size=(batch, dim)).astype(np.float32)
    b_np = rng.normal(size=(batch, dim)).astype(np.float32)
    a = cp.asarray(a_np)
    b = cp.asarray(b_np)

    expected = ck.multiply(a, b)
    out = cp.zeros((batch, dim), dtype=cp.float32)
    result = _distributed.split_batch(a, b, out=out)
    assert result is out or cp.shares_memory(result, out)
    np.testing.assert_allclose(cp.asnumpy(out), cp.asnumpy(expected), rtol=1e-5, atol=1e-5)
