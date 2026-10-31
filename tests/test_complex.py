"""Complex coefficients preserve the existing Cayley–Dickson basis product."""

import ctypes

import numpy as np
import pytest

import CayleyKernel as ck


def reference(a, b):
    """Independent recursive product, with broadcasting on leading axes."""
    if a.shape[-1] == 1:
        return a * b
    m = a.shape[-1] // 2
    x, y = a[..., :m], a[..., m:]
    u, v = b[..., :m], b[..., m:]

    def conj(z):
        result = -z.copy()
        result[..., 0] = z[..., 0]
        return result

    return np.concatenate((reference(x, u) - reference(conj(v), y),
                           reference(v, x) + reference(y, conj(u))), axis=-1)


def coefficients(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)


def assert_close(actual, expected, dtype):
    tol = 2e-5 if np.dtype(dtype) == np.dtype('complex64') else 1e-12
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)


@pytest.mark.cpu
class TestComplexWrapper:
    """Conversion and conjugation need no device; NumPy supplies array ops."""

    @pytest.mark.parametrize('ref_dtype', [None, np.float32, np.complex64])
    def test_complex_list_retains_imaginary_parts(self, monkeypatch, ref_dtype):
        monkeypatch.setattr(ck, 'cp', np)
        values = [1 + 2j, 3 - 4j]
        result = ck._as_cupy(values, ref_dtype)
        assert result.dtype.kind == 'c'
        np.testing.assert_array_equal(result, values)

    @pytest.mark.parametrize('ref_dtype', [None, np.float16, np.float32,
                                         np.float64, np.complex64])
    def test_real_list_follows_floating_precision(self, monkeypatch, ref_dtype):
        monkeypatch.setattr(ck, 'cp', np)
        result = ck._as_cupy([1, 2], ref_dtype)
        expected = np.dtype(np.float32 if ref_dtype is None else ref_dtype)
        assert result.dtype == expected

    @pytest.mark.parametrize('ref_dtype', [np.int64, np.uint32, np.bool_])
    def test_fractional_list_with_integer_operand(self, monkeypatch, ref_dtype):
        monkeypatch.setattr(ck, 'cp', np)
        values = [1.5, 2.5, 3.5]
        result = ck._as_cupy(values, np.dtype(ref_dtype))
        assert result.dtype == np.result_type(ref_dtype, np.float64)
        np.testing.assert_array_equal(result, values)

    @pytest.mark.parametrize('shape', [(4,), (2, 3, 4)])
    def test_conjugation_is_basis_conjugation(self, monkeypatch, shape):
        monkeypatch.setattr(ck, 'cp', np)
        a = coefficients(shape, np.complex128, 1)
        original = a.copy()
        result = ck.conjugate(a)
        np.testing.assert_array_equal(result[..., 0], a[..., 0])
        np.testing.assert_array_equal(result[..., 1:], -a[..., 1:])
        np.testing.assert_array_equal(a, original)
        np.testing.assert_array_equal(ck.conjugate(result), a)

    def test_list_conjugation(self, monkeypatch):
        monkeypatch.setattr(ck, 'cp', np)
        assert ck.conjugate([1 + 2j, 3 - 4j]) == [1 + 2j, -3 + 4j]

    @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
    def test_old_library_reports_missing_complex_kernel(self, monkeypatch, dtype):
        monkeypatch.setattr(ck, '_KERNELS', {})
        with pytest.raises(TypeError, match='Rebuild the kernel'):
            ck._select_kernel(np.dtype(dtype))

    @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
    def test_library_binds_both_complex_entry_points(self, dtype):
        if not ck.loaded():
            pytest.skip('Build the shared library with make kernel first.')
        legacy, batched = ck._select_kernel(np.dtype(dtype))
        assert legacy.argtypes == [ctypes.c_void_p] * 3 + [ctypes.c_int]
        assert batched is not None
        assert batched.argtypes == legacy.argtypes + [ctypes.c_longlong]
        assert legacy.restype is None
        assert batched.restype is None


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('dim', [1, 2, 4, 8, 16, 32])
def test_complex_product_against_recursive(dtype, dim):
    a = coefficients((dim,), dtype, 1)
    b = coefficients((dim,), dtype, 2)
    result = ck.multiply(a, b)
    assert result.dtype == np.dtype(dtype)
    assert_close(result, reference(a, b), dtype)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('layout', ['broadcast', 'strided', 'empty'])
def test_complex_batches(dtype, layout):
    a = coefficients((2, 1, 8), dtype, 3)
    b = coefficients((1, 3, 8), dtype, 4)
    if layout == 'strided':
        a, b = a[..., ::2], b[..., ::2]
    elif layout == 'empty':
        a = a[:0]
    result = ck.multiply(a, b)
    assert result.shape == np.broadcast_shapes(a.shape, b.shape)
    assert result.dtype == np.dtype(dtype)
    assert_close(result, reference(a, b), dtype)


@pytest.mark.parametrize('left,right', [
    (np.float32, np.complex64), (np.complex64, np.float32),
    (np.float64, np.complex64), (np.complex64, np.complex128),
])
def test_complex_dtype_promotion(left, right):
    a = np.array([1, 2, 3, 4], dtype=left)
    b = np.array([4, 3, 2, 1], dtype=right)
    if np.dtype(left).kind == 'c':
        a += 2j
    if np.dtype(right).kind == 'c':
        b -= 3j
    dtype = np.result_type(left, right)
    result = ck.multiply(a, b)
    assert result.dtype == dtype
    assert_close(result, reference(a.astype(dtype), b.astype(dtype)), dtype)


@pytest.mark.parametrize('a,b', [
    ([1 + 2j, 3 - 4j], [2 - 1j, 5 + 3j]),
    ([1, 3], [2 - 1j, 5 + 3j]),
    ([1 + 2j, 3 - 4j], [2, 5]),
])
def test_complex_python_lists(a, b):
    result = ck.multiply(a, b)
    assert isinstance(result, list)
    assert_close(result, reference(np.asarray(a), np.asarray(b)), np.complex128)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_complex_cupy_container_and_bilinearity(dtype):
    cp = ck.cp
    a = cp.asarray(coefficients((8,), dtype, 5))
    b = cp.asarray(coefficients((8,), dtype, 6))
    result = ck.multiply(a, b)
    assert isinstance(result, cp.ndarray)
    assert_close(cp.asnumpy(result), reference(cp.asnumpy(a), cp.asnumpy(b)), dtype)
    # Scalar i commutes with every basis unit and is never conjugated.
    assert_close(cp.asnumpy(ck.multiply(1j * a, b)), cp.asnumpy(1j * result), dtype)
    assert_close(cp.asnumpy(ck.multiply(a, 1j * b)), cp.asnumpy(1j * result), dtype)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_complex_legacy_c_entry_point(dtype):
    cp = ck.cp
    a = cp.asarray(coefficients((8,), dtype, 7))
    b = cp.asarray(coefficients((8,), dtype, 8))
    out = cp.empty_like(a)
    legacy, _ = ck._select_kernel(a.dtype)
    cp.cuda.get_current_stream().synchronize()
    legacy(ctypes.c_void_p(a.data.ptr), ctypes.c_void_p(b.data.ptr),
           ctypes.c_void_p(out.data.ptr), ctypes.c_int(a.size))
    assert_close(cp.asnumpy(out), reference(cp.asnumpy(a), cp.asnumpy(b)), dtype)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
def test_complex_batch_grid_stride(dtype):
    # Cross the grid.y limit without a large algebra or expensive reference.
    a = np.full((65536, 1), 1 + 2j, dtype=dtype)
    b = np.full((1,), 3 - 4j, dtype=dtype)
    assert_close(ck.multiply(a, b), a * b, dtype)
