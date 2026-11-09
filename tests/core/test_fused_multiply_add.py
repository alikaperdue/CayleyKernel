# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yotta Space Corp
"""Fused multiply-add / multiply-subtract tests.

Covers:
  (a) multiply_add(a, b, c) == multiply(a, b) + c  and
      multiply_subtract(a, b, c) == multiply(a, b) - c
      across dims 2, 4, 8, 16, 32 and dtypes float32, float64, complex64,
      complex128, float16, bfloat16, int32, bool.
  (b) batched shapes (N, dim) and (B, N, dim).
  (c) out= buffer reuse (identity on the returned array).
  (d) stream= smoke test.

All GPU tests are skipped when CuPy or the compiled kernel is unavailable
(see the autouse fixture in conftest.py).
"""

import numpy as np
import pytest

import CayleyKernel as ck
from CayleyKernel import cp
from tests.support.comparisons import assert_product_close
from tests.support.inputs import normal_coefficients
from tests.support.oracles import recursive_product

# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------


def coefficients(shape, dtype, seed=2026):
    """Random coefficients in a range safe for the given dtype."""
    dtype = np.dtype(dtype)
    if dtype.kind in "biu":
        rng = np.random.default_rng(seed)
        lower, upper = (0, 2) if dtype.kind == "b" else (-2, 3)
        return rng.integers(lower, upper, size=shape).astype(dtype)
    divisor = 4 if dtype == np.dtype(np.float16) or dtype.kind == "c" else 1
    return normal_coefficients(shape, dtype, seed, divisor=divisor)


def assert_close(actual, expected, dtype):
    """Dtype-appropriate comparison."""
    actual_np = actual.get() if hasattr(actual, "get") else np.asarray(actual)
    expected_np = np.asarray(expected)
    dt = np.dtype(dtype)
    if dt.kind in "biu":
        np.testing.assert_array_equal(actual_np, expected_np)
    else:
        if dt == np.dtype(np.float16):
            actual_np, expected_np = actual_np.astype(np.float32), expected_np.astype(np.float32)
        assert_product_close(actual_np, expected_np, dtype, contract="fused")


# ---------------------------------------------------------------------------
# (a) Correctness vs. multiply + c  /  multiply - c
# ---------------------------------------------------------------------------
_DTYPE_MATRIX = [
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    np.float16,
    np.int32,
    np.bool_,
]


@pytest.mark.parametrize("dim", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("dtype", _DTYPE_MATRIX)
def test_multiply_add_matches_multiply_plus_c(dim, dtype):
    a = coefficients((dim,), dtype, seed=1)
    b = coefficients((dim,), dtype, seed=2)
    c = coefficients((dim,), dtype, seed=3)

    got = ck.multiply_add(a, b, c)
    if np.dtype(dtype) == np.dtype(np.bool_):
        # Kernel accumulates in int; reference: int32 product + int32 c, then != 0.
        ab = np.asarray(ck.multiply(a.astype(np.int32), b.astype(np.int32)), dtype=np.int32)
        want = (ab + np.asarray(c, dtype=np.int32)) != 0
        np.testing.assert_array_equal(np.asarray(got), want)
    else:
        ab = np.asarray(ck.multiply(a, b))
        want = ab + np.asarray(c)
        assert_close(got, want, dtype)


@pytest.mark.parametrize("dim", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("dtype", _DTYPE_MATRIX)
def test_multiply_subtract_matches_multiply_minus_c(dim, dtype):
    a = coefficients((dim,), dtype, seed=10)
    b = coefficients((dim,), dtype, seed=20)
    c = coefficients((dim,), dtype, seed=30)

    got = ck.multiply_subtract(a, b, c)
    if np.dtype(dtype) == np.dtype(np.bool_):
        ab = np.asarray(ck.multiply(a.astype(np.int32), b.astype(np.int32)), dtype=np.int32)
        want = (ab - np.asarray(c, dtype=np.int32)) != 0
        np.testing.assert_array_equal(np.asarray(got), want)
    else:
        ab = np.asarray(ck.multiply(a, b))
        want = ab - np.asarray(c)
        assert_close(got, want, dtype)


@pytest.mark.parametrize("dim", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("dtype", _DTYPE_MATRIX)
def test_multiply_add_against_recursive_reference(dim, dtype):
    """Direct comparison with the recursive reference (not via multiply)."""
    a = coefficients((dim,), dtype, seed=100)
    b = coefficients((dim,), dtype, seed=200)
    c = coefficients((dim,), dtype, seed=300)

    got = ck.multiply_add(a, b, c)
    if np.dtype(dtype) == np.dtype(np.bool_):
        ab_ref = recursive_product(a, b, xp=np, calc_dtype=np.int32)
        want = (ab_ref.astype(np.int32) + np.asarray(c, dtype=np.int32)) != 0
        np.testing.assert_array_equal(np.asarray(got), want)
    else:
        ab_ref = recursive_product(a, b, xp=np, calc_dtype=None)
        want = ab_ref + np.asarray(c)
        assert_close(got, want, dtype)


# bfloat16 requires CuPy 14+ and ml_dtypes
@pytest.fixture
def bf16_dtype():
    mod = pytest.importorskip("ml_dtypes")
    try:
        return np.dtype(mod.bfloat16)
    except (TypeError, ValueError):
        pytest.skip("bfloat16 dtype not constructible in this NumPy/ml_dtypes version")


@pytest.mark.parametrize("dim", [2, 4, 8, 16, 32])
def test_bfloat16_multiply_add(dim, bf16_dtype):
    if int(cp.__version__.split(".")[0]) < 14:
        pytest.skip("BF16 requires CuPy 14+")
    rng = np.random.default_rng(42)
    a = (rng.integers(-4, 5, size=(dim,)) / 4).astype(bf16_dtype)
    b = (rng.integers(-4, 5, size=(dim,)) / 4).astype(bf16_dtype)
    c = (rng.integers(-4, 5, size=(dim,)) / 4).astype(bf16_dtype)

    got = ck.multiply_add(a, b, c)
    assert got.dtype == bf16_dtype

    ab_ref = recursive_product(a, b, xp=np, calc_dtype=None)
    want = (ab_ref + c).astype(bf16_dtype)
    np.testing.assert_array_equal(
        np.asarray(got).astype(np.float32), np.asarray(want).astype(np.float32)
    )


@pytest.mark.parametrize("dim", [2, 4, 8, 16, 32])
def test_bfloat16_multiply_subtract(dim, bf16_dtype):
    if int(cp.__version__.split(".")[0]) < 14:
        pytest.skip("BF16 requires CuPy 14+")
    rng = np.random.default_rng(99)
    a = (rng.integers(-4, 5, size=(dim,)) / 4).astype(bf16_dtype)
    b = (rng.integers(-4, 5, size=(dim,)) / 4).astype(bf16_dtype)
    c = (rng.integers(-4, 5, size=(dim,)) / 4).astype(bf16_dtype)

    got = ck.multiply_subtract(a, b, c)
    assert got.dtype == bf16_dtype

    ab_ref = recursive_product(a, b, xp=np, calc_dtype=None)
    want = (ab_ref - c).astype(bf16_dtype)
    np.testing.assert_array_equal(
        np.asarray(got).astype(np.float32), np.asarray(want).astype(np.float32)
    )


# ---------------------------------------------------------------------------
# (b) Batched shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(64, 8), (4, 16, 4), (10, 32)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64])
def test_batched_multiply_add(shape, dtype):
    rng = np.random.default_rng(7)
    a = (rng.normal(size=shape) / 4).astype(dtype)
    b = (rng.normal(size=shape) / 4).astype(dtype)
    c = (rng.normal(size=shape) / 4).astype(dtype)

    got = ck.multiply_add(a, b, c)
    assert got.shape == shape

    ab = np.asarray(ck.multiply(a, b))
    want = ab + np.asarray(c)
    assert_close(got, want, dtype)


@pytest.mark.parametrize("shape", [(64, 8), (4, 16, 4)])
def test_batched_multiply_subtract(shape):
    rng = np.random.default_rng(11)
    a = (rng.normal(size=shape) / 4).astype(np.float32)
    b = (rng.normal(size=shape) / 4).astype(np.float32)
    c = (rng.normal(size=shape) / 4).astype(np.float32)

    got = ck.multiply_subtract(a, b, c)
    assert got.shape == shape

    ab = np.asarray(ck.multiply(a, b))
    want = ab - np.asarray(c)
    assert_close(got, want, np.float32)


@pytest.mark.parametrize("dim", [4, 8])
def test_broadcast_stack_vs_vector_multiply_add(dim):
    """One vector broadcast against a stack of products."""
    N = 20
    A = (np.random.default_rng(1).normal(size=(N, dim)) / 4).astype(np.float32)
    v = (np.random.default_rng(2).normal(size=(dim,)) / 4).astype(np.float32)
    c = (np.random.default_rng(3).normal(size=(N, dim)) / 4).astype(np.float32)

    got = ck.multiply_add(A, v, c)
    assert got.shape == (N, dim)

    ab = np.asarray(ck.multiply(A, v))
    want = ab + c
    assert_close(got, want, np.float32)


# ---------------------------------------------------------------------------
# (c) out= buffer reuse (identity)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 8, 32])
def test_multiply_add_out_buffer_identity(dim):
    a = (np.random.default_rng(1).normal(size=(dim,)) / 4).astype(np.float32)
    b = (np.random.default_rng(2).normal(size=(dim,)) / 4).astype(np.float32)
    c = (np.random.default_rng(3).normal(size=(dim,)) / 4).astype(np.float32)

    out = cp.zeros(dim, dtype=cp.float32)
    result = ck.multiply_add(a, b, c, out=out)
    # Identity: the returned array IS the caller's buffer (no copy).
    assert result is out
    # Correctness.
    ab = cp.asnumpy(ck.multiply(a, b))
    want = ab + np.asarray(c)
    assert_close(cp.asnumpy(result), want, np.float32)


@pytest.mark.parametrize("dim", [4, 16])
def test_multiply_subtract_out_buffer_identity(dim):
    a = (np.random.default_rng(4).normal(size=(dim,)) / 4).astype(np.float64)
    b = (np.random.default_rng(5).normal(size=(dim,)) / 4).astype(np.float64)
    c = (np.random.default_rng(6).normal(size=(dim,)) / 4).astype(np.float64)

    out = cp.zeros(dim, dtype=cp.float64)
    result = ck.multiply_subtract(a, b, c, out=out)
    assert result is out

    ab = cp.asnumpy(ck.multiply(a, b))
    want = ab - np.asarray(c)
    assert_close(cp.asnumpy(result), want, np.float64)


def test_out_buffer_reuse_tight_loop():
    """Simulate a tight loop reusing the same buffer (no allocation)."""
    dim = 8
    a = (np.random.default_rng(10).normal(size=(dim,)) / 4).astype(np.float32)
    b = (np.random.default_rng(11).normal(size=(dim,)) / 4).astype(np.float32)
    c = (np.random.default_rng(12).normal(size=(dim,)) / 4).astype(np.float32)

    out = cp.zeros(dim, dtype=cp.float32)
    first = None
    for i in range(5):
        result = ck.multiply_add(a, b, c, out=out)
        assert result is out
        if i == 0:
            first = cp.asnumpy(result).copy()
    np.testing.assert_array_equal(cp.asnumpy(result), first)


# ---------------------------------------------------------------------------
# (d) stream= smoke test
# ---------------------------------------------------------------------------
def test_multiply_add_accepts_nondefault_stream():
    dim = 8
    a = (np.random.default_rng(20).normal(size=(dim,)) / 4).astype(np.float32)
    b = (np.random.default_rng(21).normal(size=(dim,)) / 4).astype(np.float32)
    c = (np.random.default_rng(22).normal(size=(dim,)) / 4).astype(np.float32)

    s = cp.cuda.Stream(non_blocking=True)
    with s:
        got = ck.multiply_add(a, b, c)
    s.synchronize()

    ab = cp.asnumpy(ck.multiply(a, b))
    want = ab + np.asarray(c)
    assert_close(cp.asnumpy(got), want, np.float32)


def test_multiply_subtract_accepts_nondefault_stream():
    dim = 16
    a = (np.random.default_rng(30).normal(size=(dim,)) / 4).astype(np.float64)
    b = (np.random.default_rng(31).normal(size=(dim,)) / 4).astype(np.float64)
    c = (np.random.default_rng(32).normal(size=(dim,)) / 4).astype(np.float64)

    s = cp.cuda.Stream(non_blocking=True)
    with s:
        got = ck.multiply_subtract(a, b, c)
    s.synchronize()

    ab = cp.asnumpy(ck.multiply(a, b))
    want = ab - np.asarray(c)
    assert_close(cp.asnumpy(got), want, np.float64)


# ---------------------------------------------------------------------------
# (e) ABI / binding smoke tests
# ---------------------------------------------------------------------------
@pytest.mark.cpu
def test_abi_version_is_4_or_higher(compiled_kernel):
    assert ck.abi_version() >= 4, (
        "Expected ABI >= 4 for fused multiply-add; got %d. Rebuild with `make kernel`."
        % ck.abi_version()
    )


@pytest.mark.cpu
def test_fma_symbol_bound(compiled_kernel):
    """Verify the ctypes binding for fma_mult_float is configured correctly."""
    fn = ck._select_fma(np.dtype(np.float32))
    assert len(fn.argtypes) == 7
    assert fn.restype is None


@pytest.mark.cpu
def test_fma_symbol_missing_raises(monkeypatch):
    """A stale library without fma_mult_* symbols should raise RuntimeError."""
    monkeypatch.setattr(ck, "_lib", None)
    with pytest.raises(RuntimeError, match="fused fma_mult"):
        ck._select_fma(np.dtype(np.float32))


def test_multiply_add_rejects_wrong_dim():
    with pytest.raises(ValueError, match="power of two"):
        ck.multiply_add(
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )


def test_multiply_subtract_rejects_dim_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        ck.multiply_subtract(
            np.zeros(4, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        )


def test_multiply_add_rejects_bad_out():
    with pytest.raises(TypeError, match="out.dtype"):
        a = np.ones(4, dtype=np.float32)
        ck.multiply_add(a, a, a, out=cp.zeros(4, dtype=cp.float64))


def test_multiply_subtract_rejects_bad_out_shape():
    with pytest.raises(ValueError, match="out.shape"):
        a = np.ones(4, dtype=np.float32)
        ck.multiply_subtract(a, a, a, out=cp.zeros(8, dtype=cp.float32))
