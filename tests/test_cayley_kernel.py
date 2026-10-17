"""Tests for the ``CayleyKernel`` package.

Multi-dimensional (batched) products are verified in two independent ways:

  1. Row-by-row against the recursive reference, and
  2. Row-by-row against the kernel's own 1D (single-vector) path,

which catches any divergence introduced by the batch axis / grid-stride loop.

Run with:
    pytest -q tests/test_cayley_kernel.py
"""

import os
import sys

import numpy as np
import pytest

# Make the repository root importable regardless of where pytest is launched
# from, so that `import CayleyKernel` resolves to the local package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cupy as cp  # noqa: E402
import CayleyKernel  # noqa: E402
from CayleyKernel import multiply  # noqa: E402


# ---------------------------------------------------------------------------
# Reference: well-known recursive Cayley-Dickson doubling product.
# ---------------------------------------------------------------------------
def ref_conj(x):
    """Conjugation: negate every component except the first."""
    xstar = -x
    xstar[0] = x[0]
    return xstar


def ref_mult(x, y):
    """Recursive Cayley-Dickson doubling product (1-D cupy arrays)."""
    n = len(x)
    if n == 1:
        return x * y
    m = n // 2
    a, b = x[:m], x[m:]
    c, d = y[:m], y[m:]
    z = cp.zeros(n, dtype=x.dtype)
    z[:m] = ref_mult(a, c) - ref_mult(ref_conj(d), b)
    z[m:] = ref_mult(d, a) + ref_mult(b, ref_conj(c))
    return z


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    cp.random.seed(0)


# ---------------------------------------------------------------------------
# 1D: kernel vs. recursive reference (single vector, legacy path)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 4, 8, 16])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_1d_float_against_recursive(dim, dtype):
    a = cp.random.randn(dim).astype(dtype)
    b = cp.random.randn(dim).astype(dtype)
    got = multiply(a, b)
    want = ref_mult(a, b)
    assert got.shape == (dim,)
    rtol, atol = (1e-4, 1e-5) if dtype is cp.float32 else (1e-10, 1e-12)
    cp.testing.assert_allclose(got, want, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_1d_int32_against_recursive(dim):
    a = (cp.random.randint(0, 5, size=dim) - 2).astype(cp.int32)
    b = (cp.random.randint(0, 5, size=dim) - 2).astype(cp.int32)
    cp.testing.assert_array_equal(multiply(a, b), ref_mult(a, b))


def test_1d_int64_against_recursive():
    dim = 4
    a = (cp.random.randint(0, 5, size=dim) - 2).astype(cp.int64)
    b = (cp.random.randint(0, 5, size=dim) - 2).astype(cp.int64)
    cp.testing.assert_array_equal(multiply(a, b), ref_mult(a, b))


def test_1d_uint8_against_recursive():
    dim = 4
    a = cp.random.randint(0, 4, size=dim).astype(cp.uint8)
    b = cp.random.randint(0, 4, size=dim).astype(cp.uint8)
    cp.testing.assert_array_equal(multiply(a, b), ref_mult(a, b))


# ---------------------------------------------------------------------------
# 2D batch: kernel's batched path vs. per-vector 1D path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 4, 8])
@pytest.mark.parametrize("batch", [3, 17, 64])
def test_2d_batch_matches_1d(dim, batch):
    A = cp.random.randn(batch, dim).astype(cp.float32)
    B = cp.random.randn(batch, dim).astype(cp.float32)
    got = multiply(A, B)
    assert got.shape == (batch, dim)
    for i in range(batch):
        cp.testing.assert_allclose(
            got[i], multiply(A[i], B[i]), rtol=1e-4, atol=1e-5
        )


@pytest.mark.parametrize("dim", [4, 8])
def test_2d_batch_matches_recursive(dim):
    batch = 5
    A = cp.random.randn(batch, dim).astype(cp.float64)
    B = cp.random.randn(batch, dim).astype(cp.float64)
    got = multiply(A, B)
    for i in range(batch):
        cp.testing.assert_allclose(
            got[i], ref_mult(A[i], B[i]), rtol=1e-10, atol=1e-12
        )


# ---------------------------------------------------------------------------
# 3D batch: kernel's batched path vs. per-vector 1D path
# ---------------------------------------------------------------------------
def test_3d_batch_matches_1d():
    B, N, dim = 2, 5, 8
    A = cp.random.randn(B, N, dim).astype(cp.float32)
    C = cp.random.randn(B, N, dim).astype(cp.float32)
    got = multiply(A, C)
    assert got.shape == (B, N, dim)
    for i in range(B):
        for j in range(N):
            cp.testing.assert_allclose(
                got[i, j], multiply(A[i, j], C[i, j]), rtol=1e-4, atol=1e-5
            )


def test_4d_batch_matches_1d():
    # A genuinely N-D batch: (2, 3, 4, dim).
    dims = (2, 3, 4, 8)
    A = cp.random.randn(*dims).astype(cp.float32)
    B = cp.random.randn(*dims).astype(cp.float32)
    got = multiply(A, B)
    assert got.shape == dims
    flat_a = A.reshape(-1, dims[-1])
    flat_b = B.reshape(-1, dims[-1])
    flat_got = got.reshape(-1, dims[-1])
    for i in range(flat_a.shape[0]):
        cp.testing.assert_allclose(
            flat_got[i], multiply(flat_a[i], flat_b[i]), rtol=1e-4, atol=1e-5
        )


# ---------------------------------------------------------------------------
# Broadcasting: (N, dim) x (dim,) -> (N, dim) and its mirror
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 4, 8])
def test_broadcast_stack_against_vector(dim):
    N = 10
    A = cp.random.randn(N, dim).astype(cp.float32)
    v = cp.random.randn(dim).astype(cp.float32)
    got = multiply(A, v)
    assert got.shape == (N, dim)
    for i in range(N):
        cp.testing.assert_allclose(
            got[i], multiply(A[i], v), rtol=1e-4, atol=1e-5
        )


def test_broadcast_vector_against_stack():
    N, dim = 6, 8
    A = cp.random.randn(N, dim).astype(cp.float32)
    v = cp.random.randn(dim).astype(cp.float32)
    got = multiply(v, A)
    assert got.shape == (N, dim)
    for i in range(N):
        cp.testing.assert_allclose(
            got[i], multiply(v, A[i]), rtol=1e-4, atol=1e-5
        )


def test_broadcast_3d_stack_against_vector():
    B, N, dim = 2, 4, 8
    A = cp.random.randn(B, N, dim).astype(cp.float32)
    v = cp.random.randn(dim).astype(cp.float32)
    got = multiply(A, v)
    assert got.shape == (B, N, dim)
    for i in range(B):
        for j in range(N):
            cp.testing.assert_allclose(
                got[i, j], multiply(A[i, j], v), rtol=1e-4, atol=1e-5
            )


# ---------------------------------------------------------------------------
# Batch size larger than the CUDA grid.y cap (65535) -> grid-stride path
# ---------------------------------------------------------------------------
def test_batch_exceeds_max_grid_y():
    dim = 4
    batch = 70000  # > 65535, forces the grid-stride loop in the kernel
    A = cp.random.randn(batch, dim).astype(cp.float32)
    B = cp.random.randn(batch, dim).astype(cp.float32)
    got = multiply(A, B)
    assert got.shape == (batch, dim)
    for i in (0, 1, 65534, 65535, 65536, batch - 1):
        cp.testing.assert_allclose(
            got[i], multiply(A[i], B[i]), rtol=1e-4, atol=1e-5
        )


# ---------------------------------------------------------------------------
# Container and dtype behaviours
# ---------------------------------------------------------------------------
def test_list_in_list_out():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [5.0, 6.0, 7.0, 8.0]
    got = multiply(a, b)
    assert isinstance(got, list)
    assert len(got) == 4
    assert all(isinstance(v, float) for v in got)


def test_numpy_in_numpy_out():
    a = np.array([1.0, 2.0], dtype=np.float64)
    b = np.array([3.0, 4.0], dtype=np.float64)
    got = multiply(a, b)
    assert isinstance(got, np.ndarray)
    assert got.dtype == np.float64
    assert got.shape == (2,)


def test_cupy_in_cupy_out():
    a = cp.array([1.0, 2.0], dtype=cp.float64)
    b = cp.array([3.0, 4.0], dtype=cp.float64)
    got = multiply(a, b)
    assert isinstance(got, cp.ndarray)
    assert got.dtype == cp.float64
    assert got.shape == (2,)


def test_dtype_promotion():
    a = cp.array([1.0, 2.0], dtype=cp.float32)
    b = cp.array([3.0, 4.0], dtype=cp.float64)
    got = multiply(a, b)
    assert got.dtype == cp.float64


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------
def test_rejects_non_power_of_two_dim():
    with pytest.raises(ValueError):
        multiply(cp.zeros(3, dtype=cp.float32), cp.zeros(3, dtype=cp.float32))


def test_rejects_dim_mismatch():
    with pytest.raises(ValueError):
        multiply(
            cp.zeros(4, dtype=cp.float32),
            cp.zeros(8, dtype=cp.float32),
        )


def test_rejects_scalar_inputs():
    with pytest.raises(ValueError):
        multiply(cp.array(1.0), cp.array(2.0))


def test_rejects_unsupported_dtype():
    with pytest.raises(TypeError):
        multiply(
            cp.zeros(4, dtype=cp.complex64),
            cp.zeros(4, dtype=cp.complex64),
        )


# ---------------------------------------------------------------------------
# Hand-check on a small 2-component product (complex-like behaviour)
# ---------------------------------------------------------------------------
def test_2d_product_small_case():
    a = cp.array([3.0, 4.0], dtype=cp.float64)
    b = cp.array([1.0, 2.0], dtype=cp.float64)
    cp.testing.assert_allclose(
        multiply(a, b), ref_mult(a, b), rtol=1e-12, atol=1e-14
    )
