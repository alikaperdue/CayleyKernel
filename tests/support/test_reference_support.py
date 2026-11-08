# SPDX-License-Identifier: Apache-2.0
"""Validate the shared array oracle without using CayleyKernel products."""

import numpy as np
import pytest

from tests.support.inputs import random_sparse
from tests.support.oracles import (
    doubling_sign,
    doubling_sign_at_level,
    iterative_doubling_sign,
    recursive_product,
    sparse_product,
)

pytestmark = pytest.mark.cpu


def test_exact_products_and_broadcast():
    a = np.array([[[3, 4]], [[0, 1]]], dtype=np.int64)
    b = np.array([[1, 2], [1, 0]], dtype=np.int64)
    got = recursive_product(a, b, xp=np, calc_dtype=None)
    assert got.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(got, [[[-5, 10], [3, 4]], [[-2, 1], [0, 1]]])

    got = recursive_product([1j, 1], [2, 1j], xp=np, calc_dtype=np.complex128)
    np.testing.assert_array_equal(got, [1j, 1])


def test_calculation_dtype_and_final_rounding():
    a = np.array([0, 1], dtype=np.uint8)
    got = recursive_product(a, a, xp=np, calc_dtype=None)
    assert got.dtype == np.dtype(np.uint8)
    np.testing.assert_array_equal(got, [255, 0])
    np.testing.assert_array_equal(recursive_product(a, a, xp=np, calc_dtype=np.int32), [-1, 0])

    a = np.array([True, True])
    got = recursive_product(a, a, xp=np, calc_dtype=np.int32)
    np.testing.assert_array_equal(got != 0, [False, True])

    a = np.array([1, 1 + 2**-10], dtype=np.float16)
    b = np.array([1, 1 - 2**-11], dtype=np.float16)
    narrow = recursive_product(a, b, xp=np, calc_dtype=None)
    wide = recursive_product(a, b, xp=np, calc_dtype=np.float64)
    assert narrow.dtype == np.dtype(np.float16)
    assert wide.dtype == np.dtype(np.float64)
    assert narrow[0] == 0
    assert wide[0] == 2**-21 - 2**-11
    assert wide.astype(np.float16)[0] == 2**-21 - 2**-11

    a = np.array([1, 2**-11], dtype=np.float16)
    b = np.array([1, -1], dtype=np.float16)
    wide = recursive_product(a, b, xp=np, calc_dtype=np.float64)
    assert wide[0] == 1 + 2**-11
    assert wide.astype(np.float16)[0] == 1


def test_dense_oracle_agrees_with_independent_sign_and_sparse_oracles():
    dim = 16
    eye = np.eye(dim, dtype=np.int64)
    expected = np.zeros((dim, dim, dim), dtype=np.int64)
    for i in range(dim):
        for j in range(dim):
            sign = doubling_sign(i, j)
            assert sign == doubling_sign_at_level(i, j, 4) == iterative_doubling_sign(i, j)
            expected[i, j, i ^ j] = sign
    got = recursive_product(eye[:, None], eye[None, :], xp=np, calc_dtype=None)
    np.testing.assert_array_equal(got, expected)

    rng = np.random.default_rng(1234)
    ix, vx = random_sparse(dim, 8, rng)
    iy, vy = random_sparse(dim, 8, rng)
    a, b = np.zeros(dim, dtype=np.int64), np.zeros(dim, dtype=np.int64)
    a[ix], b[iy] = vx, vy
    np.testing.assert_array_equal(
        recursive_product(a, b, xp=np, calc_dtype=None), sparse_product(dim, ix, vx, iy, vy)
    )
