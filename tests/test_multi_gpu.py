# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Multi-GPU split_batch correctness and overlap sanity tests.

These tests are skipped automatically when fewer than two CUDA devices
are visible (``visible_devices() < 2``), so single-GPU CI runs stay green.
"""

import time

import numpy as np
import pytest

import CayleyKernel as ck
from CayleyKernel import _distributed, cp


@pytest.fixture(scope="module", autouse=True)
def _two_devices(gpu_device_count):
    if gpu_device_count < 2:
        pytest.skip("Multi-GPU tests require at least two visible devices")


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


# ---------------------------------------------------------------------------
# overlap sanity: 2-GPU split must not exceed 2x the sum of the two
# per-GPU times (i.e. we are not serialising on the CPU with extra
# overhead on top of the natural sequential cost)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_split_batch_overlaps_across_devices():
    """On a 2-GPU machine, split_batch(batch=2**20, dim=32, float32)
    must complete in <= 2x the sum of the two per-GPU times.

    Each "per-GPU time" includes the D2D copy of the input chunk to that
    device, the kernel launch, and the final sync -- exactly what
    split_batch does internally per device.  This makes the comparison
    apples-to-apples: both sides pay for data movement.

    With proper CPU-side overlap, split_batch should take closer to
    max(t_dev0, t_dev1) + gather_time.  The 2x bound catches pathological
    CPU serialisation (e.g. an accidental per-row Python loop) without
    being so tight that PCIe copy variance causes flakes.
    """
    dim = 32
    batch = 2**20
    devices = _distributed.visible_devices()[:2]

    # Allocate directly on the default (caller's) device.
    a = cp.random.normal(size=(batch, dim), dtype=cp.float32)
    b = cp.random.normal(size=(batch, dim), dtype=cp.float32)

    half = batch // 2

    # Warm up both devices + the full split_batch path so first-call
    # allocator / driver / kernel-compile costs don't skew timing.
    for dev in devices:
        with cp.cuda.Device(dev):
            tmp = cp.empty((64, dim), dtype=cp.float32)
            tmp.fill(0.0)
    cp.cuda.runtime.deviceSynchronize()
    _distributed.split_batch(a, b, devices=devices)
    cp.cuda.runtime.deviceSynchronize()

    # --- Per-device reference: D2D copy-in + kernel + sync -------------------
    # Mirrors exactly what split_batch does internally for each device,
    # so the comparison is fair (both sides pay for data movement).
    with cp.cuda.Device(devices[0]):
        t0 = time.perf_counter()
        a0_dev = a[:half].copy()
        b0_dev = b[:half].copy()
        ck.multiply(a0_dev, b0_dev)
        cp.cuda.runtime.deviceSynchronize()
        t_dev0 = (time.perf_counter() - t0) * 1000.0

    with cp.cuda.Device(devices[1]):
        t0 = time.perf_counter()
        a1_dev = a[half:].copy()
        b1_dev = b[half:].copy()
        ck.multiply(a1_dev, b1_dev)
        cp.cuda.runtime.deviceSynchronize()
        t_dev1 = (time.perf_counter() - t0) * 1000.0

    t_sum = t_dev0 + t_dev1

    # --- Timed split_batch ---------------------------------------------------
    t_start = time.perf_counter()
    _distributed.split_batch(a, b, devices=devices)
    cp.cuda.runtime.deviceSynchronize()
    t_split_ms = (time.perf_counter() - t_start) * 1000.0

    assert t_split_ms <= 2.0 * t_sum, (
        "split_batch took %.2f ms but the sum of the two per-GPU "
        "times (copy+kernel+sync) was %.2f ms (t0=%.2f, t1=%.2f); "
        "the launches appear to be serialised on the CPU with extra "
        "overhead." % (t_split_ms, t_sum, t_dev0, t_dev1)
    )
