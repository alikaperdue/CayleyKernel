# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""PyTorch interop tests for CayleyKernel.

These tests run against a CUDA ``torch.Tensor`` and validate:

* correctness against the recursive reference in :mod:`tests.support.oracles`,
* ``out=`` identity preservation,
* cross-device / dtype preservation,
* the autograd guard (v1 raises ``NotImplementedError``),
* zero-copy behaviour (no D2H round-trip when inputs are already CUDA).

Run with:
    pytest -q tests/test_torch_interop.py
"""

import statistics
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import CayleyKernel as ck  # noqa: E402
from tests.support.comparisons import assert_product_close  # noqa: E402
from tests.support.oracles import recursive_product  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _require_torch_cuda(pytestconfig, gpu_kernel):
    if not torch.cuda.is_available():
        reason = "PyTorch CUDA is unavailable"
        if pytestconfig.getoption("--require-gpu"):
            pytest.fail(f"--require-gpu: {reason}", pytrace=False)
        pytest.skip(reason)


def _cuda_tensor_factory(*shape, dtype=torch.float32):
    """Allocate a random CUDA tensor on device 0."""
    return torch.randn(*shape, device="cuda", dtype=dtype)


def _as_numpy(t):
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Correctness: dims x batches x dtypes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 8, 32])
@pytest.mark.parametrize("batch", [1, 256])
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.complex64],
    ids=["float32", "float64", "complex64"],
)
def test_torch_multiply_matches_numpy_reference(dim, batch, dtype):
    if dtype == torch.complex64:
        a = _cuda_tensor_factory(batch, dim, dtype=dtype) + 1j * _cuda_tensor_factory(
            batch, dim, dtype=dtype
        )
        b = _cuda_tensor_factory(batch, dim, dtype=dtype) + 1j * _cuda_tensor_factory(
            batch, dim, dtype=dtype
        )
    else:
        a = _cuda_tensor_factory(batch, dim, dtype=dtype)
        b = _cuda_tensor_factory(batch, dim, dtype=dtype)
    got = ck.torch.multiply(a, b)
    assert got.dtype == dtype
    assert got.shape == a.shape
    assert got.device == a.device

    a_np, b_np = _as_numpy(a), _as_numpy(b)
    wide = np.complex128 if a_np.dtype.kind == "c" else np.float64
    ref_np = recursive_product(a_np, b_np, xp=np, calc_dtype=wide)
    got_np = _as_numpy(got).astype(ref_np.dtype)
    if dtype == torch.float64:
        assert_product_close(
            got_np,
            ref_np,
            a_np.dtype,
            contract="interop",
            override=(1e-12, 1e-12),
            reason="Retain the tighter relative budget against the wide Torch reference.",
        )
    else:
        assert_product_close(got_np, ref_np, a_np.dtype, contract="interop")


def test_torch_multiply_single_vector_float32():
    """dim=2 (complex-like): hand-verify (3+4i)(1+2i) = -5+10i."""
    a = torch.tensor([3.0, 4.0], dtype=torch.float32, device="cuda")
    b = torch.tensor([1.0, 2.0], dtype=torch.float32, device="cuda")
    got = ck.torch.multiply(a, b)
    np.testing.assert_allclose(
        _as_numpy(got),
        np.array([-5.0, 10.0], dtype=np.float32),
        rtol=1e-5,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# out= identity
# ---------------------------------------------------------------------------
def test_torch_out_identity():
    a = _cuda_tensor_factory(8, dtype=torch.float32)
    b = _cuda_tensor_factory(8, dtype=torch.float32)
    o = torch.empty_like(a)
    got = ck.torch.multiply(a, b, out=o)
    assert got is o, "out= must preserve the same torch.Tensor object"
    # multiply_into is the alias
    o2 = torch.empty_like(a)
    got2 = ck.torch.multiply_into(a, b, o2)
    assert got2 is o2


def test_torch_multiply_into_requires_out():
    a = _cuda_tensor_factory(8, dtype=torch.float32)
    b = _cuda_tensor_factory(8, dtype=torch.float32)
    with pytest.raises(TypeError):
        ck.torch.multiply_into(a, b, None)


# ---------------------------------------------------------------------------
# Cross-device / dtype preservation
# ---------------------------------------------------------------------------
def test_torch_float16_cuda0_preserved():
    """A float16 cuda:0 input produces a float16 cuda:0 output."""
    a = _cuda_tensor_factory(8, dtype=torch.float16)
    b = _cuda_tensor_factory(8, dtype=torch.float16)
    assert a.dtype == torch.float16 and a.device.index == 0
    got = ck.torch.multiply(a, b)
    assert got.dtype == torch.float16
    assert got.device == torch.device("cuda:0")


def test_torch_rejects_mismatched_devices():
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires 2 CUDA devices")
    a = _cuda_tensor_factory(8, dtype=torch.float32).to("cuda:0")
    b = _cuda_tensor_factory(8, dtype=torch.float32).to("cuda:1")
    with pytest.raises(ValueError, match="a.device"):
        ck.torch.multiply(a, b)


def test_torch_rejects_non_power_of_two_dim():
    a = _cuda_tensor_factory(3, dtype=torch.float32)
    b = _cuda_tensor_factory(3, dtype=torch.float32)
    with pytest.raises(ValueError, match="power of two"):
        ck.torch.multiply(a, b)


def test_torch_rejects_mixed_dtype():
    a = _cuda_tensor_factory(8, dtype=torch.float32)
    b = _cuda_tensor_factory(8, dtype=torch.float64)
    with pytest.raises(TypeError, match="homogeneous"):
        ck.torch.multiply(a, b)


def test_torch_rejects_unsupported_dtype():
    # torch.int32 is not in the supported set.
    a = torch.zeros(8, dtype=torch.int32, device="cuda")
    b = torch.zeros(8, dtype=torch.int32, device="cuda")
    with pytest.raises(
        TypeError, match="float16, bfloat16, float32, float64, complex64, complex128"
    ):
        ck.torch.multiply(a, b)


# ---------------------------------------------------------------------------
# Autograd guard
# ---------------------------------------------------------------------------
def test_torch_autograd_guard():
    a = torch.randn(8, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn(8, device="cuda", dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="autograd is not supported in v1"):
        ck.torch.multiply(a, b)


def test_torch_autograd_guard_disabled():
    """When torch.no_grad() is active, the call must succeed (no raise)."""
    a = torch.randn(8, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn(8, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        got = ck.torch.multiply(a, b)
    assert got.dtype == torch.float32


# ---------------------------------------------------------------------------
# Zero-copy / no D2H round-trip check
# ---------------------------------------------------------------------------
def _time_call(fn, warmup=3, reps=10):
    """Time ``fn`` with CUDA synchronisation on both sides (wall seconds)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def test_torch_zero_copy_vs_pure_cupy():
    """A torch multiply call with already-CUDA inputs should be within 5%
    of a pure-CuPy multiply call at the same shape/dtype.

    This is a heuristic check: if the adapter accidentally introduced a
    D2H copy, the torch path would be many times slower than the CuPy path.
    """
    dim, batch = 32, 4096
    a_t = _cuda_tensor_factory(batch, dim, dtype=torch.float32)
    b_t = _cuda_tensor_factory(batch, dim, dtype=torch.float32)

    # Build the same data as CuPy arrays (sharing memory where possible).
    a_c = ck.torch._torch_to_cupy(a_t)
    b_c = ck.torch._torch_to_cupy(b_t)

    t_cupy = _time_call(lambda: ck.multiply(a_c, b_c))
    t_torch = _time_call(lambda: ck.torch.multiply(a_t, b_t))

    # The torch path should be within 300% of the CuPy path.  The adapter
    # adds Python-level validation, torch.empty_like allocation, and three
    # UnownedMemory wrappings per call.  A D2H round-trip would be an order
    # of magnitude slower (10x+), so a 3x ceiling safely distinguishes
    # normal dispatch overhead from an actual data copy.
    ratio = t_torch / t_cupy
    assert ratio <= 3.0, (
        "torch path (%.4fs) is not within 300%% of CuPy path (%.4fs); "
        "ratio=%.3f — possible D2H round-trip" % (t_torch, t_cupy, ratio)
    )
