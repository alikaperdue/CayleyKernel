# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Correctness tests: GPU kernel vs the recursive Cayley-Dickson reference.

Small products are compared against recursive doubling. Larger products
use sparse operands and a reference that expands only their nonzero terms.
The kernel still evaluates the full dense product, so the largest cases
carry the slow marker.

Run with:
    pytest -q tests/test_correctness.py
"""

import numpy as np
import pytest

from CayleyKernel import _select_kernel, cp, multiply, twist
from tests.support.comparisons import assert_product_close
from tests.support.inputs import random_sparse
from tests.support.oracles import doubling_sign, recursive_product, sparse_product


def as_dense(dim, ix, vx, dtype):
    x = np.zeros(dim, dtype=np.float64)
    x[ix] = vx
    return cp.asarray(x.astype(dtype))


# ---------------------------------------------------------------------------
# 1D: kernel vs. recursive reference (dense, small dims)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_1d_float_against_recursive(dim, dtype):
    a = cp.random.randn(dim).astype(dtype)
    b = cp.random.randn(dim).astype(dtype)
    got = multiply(a, b)
    want = recursive_product(a, b, xp=cp, calc_dtype=None)
    assert got.shape == (dim,)
    assert_product_close(got, want, dtype, contract="dense", xp=cp)


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_1d_int32_against_recursive(dim):
    a = (cp.random.randint(0, 5, size=dim) - 2).astype(cp.int32)
    b = (cp.random.randint(0, 5, size=dim) - 2).astype(cp.int32)
    cp.testing.assert_array_equal(multiply(a, b), recursive_product(a, b, xp=cp, calc_dtype=None))


def test_1d_int64_against_recursive():
    dim = 4
    a = (cp.random.randint(0, 5, size=dim) - 2).astype(cp.int64)
    b = (cp.random.randint(0, 5, size=dim) - 2).astype(cp.int64)
    cp.testing.assert_array_equal(multiply(a, b), recursive_product(a, b, xp=cp, calc_dtype=None))


def test_1d_uint8_against_recursive():
    dim = 4
    a = cp.random.randint(0, 4, size=dim).astype(cp.uint8)
    b = cp.random.randint(0, 4, size=dim).astype(cp.uint8)
    cp.testing.assert_array_equal(multiply(a, b), recursive_product(a, b, xp=cp, calc_dtype=None))


def test_bool_matches_int32_nonzero_pattern():
    """bool kernel == (int32 product != 0): the kernel accumulates in int."""
    dim = 8
    a = (cp.random.randint(0, 2, size=dim)).astype(cp.bool_)
    b = (cp.random.randint(0, 2, size=dim)).astype(cp.bool_)
    got = multiply(a, b)
    want = multiply(a.astype(cp.int32), b.astype(cp.int32)) != 0
    cp.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# High dimensions: exact sparse verification (no recursion needed)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dim",
    [
        1024,
        pytest.param(2**14, marks=pytest.mark.slow),
        pytest.param(2**16, marks=pytest.mark.slow),
    ],
)
def test_sparse_high_dim_exact(dim):
    rng = np.random.default_rng(1234)
    ix, vx = random_sparse(dim, 8, rng)
    iy, vy = random_sparse(dim, 8, rng)

    got = multiply(as_dense(dim, ix, vx, np.float64), as_dense(dim, iy, vy, np.float64))
    want = sparse_product(dim, ix, vx, iy, vy)

    cp.testing.assert_array_equal(got, cp.asarray(want))


@pytest.mark.slow
def test_sparse_spot_2pow20():
    """2^20 spot check: one exact sparse product at ~1M dimensions."""
    dim = 2**20
    rng = np.random.default_rng(2020)
    ix, vx = random_sparse(dim, 6, rng)
    iy, vy = random_sparse(dim, 6, rng)

    got = multiply(as_dense(dim, ix, vx, np.float32), as_dense(dim, iy, vy, np.float32))
    want = sparse_product(dim, ix, vx, iy, vy)

    cp.testing.assert_array_equal(got, cp.asarray(want, dtype=cp.float32))


# ---------------------------------------------------------------------------
# Full basis sign table (dim = 16): every e_row * e_col
# ---------------------------------------------------------------------------
@pytest.mark.cpu
def test_basis_sign_oracle_matches_cpu_dim16():
    for row in range(16):
        for col in range(16):
            assert doubling_sign(row, col) == twist(row, col), (row, col)


def test_basis_sign_table_dim16():
    dim = 16
    for row in range(dim):
        for col in range(dim):
            x = cp.zeros(dim, dtype=cp.float64)
            y = cp.zeros(dim, dtype=cp.float64)
            x[row] = 1.0
            y[col] = 1.0
            got = multiply(x, y)

            sign = doubling_sign(row, col)
            want = cp.zeros(dim, dtype=cp.float64)
            want[row ^ col] = float(sign)
            cp.testing.assert_array_equal(got, want)

            # And the recursive reference must agree as well.
            cp.testing.assert_array_equal(got, recursive_product(x, y, xp=cp, calc_dtype=None))


# ---------------------------------------------------------------------------
# Batched paths
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dim", [2, 4, 8])
@pytest.mark.parametrize("batch", [3, 17, 64])
def test_2d_batch_matches_1d(dim, batch):
    A = cp.random.randn(batch, dim).astype(cp.float32)
    B = cp.random.randn(batch, dim).astype(cp.float32)
    got = multiply(A, B)
    assert got.shape == (batch, dim)
    for i in range(batch):
        assert_product_close(got[i], multiply(A[i], B[i]), np.float32, contract="dense", xp=cp)


@pytest.mark.parametrize("dim", [4, 8])
def test_2d_batch_matches_recursive(dim):
    batch = 5
    A = cp.random.randn(batch, dim).astype(cp.float64)
    B = cp.random.randn(batch, dim).astype(cp.float64)
    got = multiply(A, B)
    for i in range(batch):
        assert_product_close(
            got[i],
            recursive_product(A[i], B[i], xp=cp, calc_dtype=None),
            np.float64,
            contract="dense",
            xp=cp,
        )


def test_3d_batch_matches_1d():
    B, N, dim = 2, 5, 8
    A = cp.random.randn(B, N, dim).astype(cp.float32)
    C = cp.random.randn(B, N, dim).astype(cp.float32)
    got = multiply(A, C)
    assert got.shape == (B, N, dim)
    for i in range(B):
        for j in range(N):
            assert_product_close(
                got[i, j], multiply(A[i, j], C[i, j]), np.float32, contract="dense", xp=cp
            )


def test_4d_batch_matches_1d():
    dims = (2, 3, 4, 8)
    A = cp.random.randn(*dims).astype(cp.float32)
    B = cp.random.randn(*dims).astype(cp.float32)
    got = multiply(A, B)
    assert got.shape == dims
    flat_a = A.reshape(-1, dims[-1])
    flat_b = B.reshape(-1, dims[-1])
    flat_got = got.reshape(-1, dims[-1])
    for i in range(flat_a.shape[0]):
        assert_product_close(
            flat_got[i], multiply(flat_a[i], flat_b[i]), np.float32, contract="dense", xp=cp
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
        assert_product_close(got[i], multiply(A[i], v), np.float32, contract="dense", xp=cp)


def test_broadcast_vector_against_stack():
    N, dim = 6, 8
    A = cp.random.randn(N, dim).astype(cp.float32)
    v = cp.random.randn(dim).astype(cp.float32)
    got = multiply(v, A)
    assert got.shape == (N, dim)
    for i in range(N):
        assert_product_close(got[i], multiply(v, A[i]), np.float32, contract="dense", xp=cp)


def test_broadcast_3d_stack_against_vector():
    B, N, dim = 2, 4, 8
    A = cp.random.randn(B, N, dim).astype(cp.float32)
    v = cp.random.randn(dim).astype(cp.float32)
    got = multiply(A, v)
    assert got.shape == (B, N, dim)
    for i in range(B):
        for j in range(N):
            assert_product_close(
                got[i, j], multiply(A[i, j], v), np.float32, contract="dense", xp=cp
            )


# ---------------------------------------------------------------------------
# Batch size larger than the legacy CUDA grid.y cap (65535)
# ---------------------------------------------------------------------------
def test_batch_exceeds_max_grid_y():
    dim = 4
    batch = 70000  # Regression for batches above the old y-axis grid limit.
    A = cp.random.randn(batch, dim).astype(cp.float32)
    B = cp.random.randn(batch, dim).astype(cp.float32)
    got = multiply(A, B)
    assert got.shape == (batch, dim)
    for i in (0, 1, 65534, 65535, 65536, batch - 1):
        assert_product_close(got[i], multiply(A[i], B[i]), np.float32, contract="dense", xp=cp)


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
        # Structured dtypes remain unsupported.
        _select_kernel(np.dtype([("value", np.float32)]))


# ---------------------------------------------------------------------------
# Hand-check on a small 2-component product (complex-like behaviour)
# ---------------------------------------------------------------------------
def test_2d_product_small_case():
    a = cp.array([3.0, 4.0], dtype=cp.float64)
    b = cp.array([1.0, 2.0], dtype=cp.float64)
    # (3 + 4i)(1 + 2i) = -5 + 10i
    got = multiply(a, b)
    cp.testing.assert_allclose(got, cp.array([-5.0, 10.0]), rtol=1e-12, atol=1e-14)
    cp.testing.assert_allclose(
        got, recursive_product(a, b, xp=cp, calc_dtype=None), rtol=1e-12, atol=1e-14
    )
