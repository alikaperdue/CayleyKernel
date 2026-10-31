"""FP16 storage uses FP32 arithmetic before the final FP16 conversion."""

import ctypes

import numpy as np
import pytest

import CayleyKernel as ck


def reference(a, b):
    """Independent recursive doubling reference, evaluated in float64."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
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


def coefficients(shape, seed):
    return (np.random.default_rng(seed).normal(size=shape) / 4).astype(np.float16)


def assert_half(actual, expected):
    assert actual.dtype == np.dtype(np.float16)
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=np.float16),
                               rtol=2e-3, atol=5e-4)


@pytest.mark.cpu
class TestHalfWrapper:
    @pytest.mark.parametrize('values', [[3, 4], [1.5, -2.25]])
    def test_real_list_follows_half_operand(self, monkeypatch, values):
        monkeypatch.setattr(ck, 'cp', np)
        result = ck._as_cupy(values, np.float16)
        assert result.dtype == np.dtype(np.float16)
        np.testing.assert_array_equal(result, values)

    def test_complex_list_with_half_operand_keeps_imaginary_parts(self, monkeypatch):
        monkeypatch.setattr(ck, 'cp', np)
        result = ck._as_cupy([1 + 2j, 3 - 4j], np.float16)
        assert result.dtype.kind == 'c'
        np.testing.assert_array_equal(result, [1 + 2j, 3 - 4j])

    def test_old_library_reports_missing_half_kernel(self, monkeypatch):
        monkeypatch.setattr(ck, '_KERNELS', {})
        with pytest.raises(TypeError, match='Rebuild the kernel'):
            ck._select_kernel(np.dtype(np.float16))

    def test_library_binds_half_entry_points(self):
        if not ck.loaded():
            pytest.skip('Build the shared library with make kernel first.')
        legacy, batched = ck._select_kernel(np.dtype(np.float16))
        assert legacy.argtypes == [ctypes.c_void_p] * 3 + [ctypes.c_int]
        assert batched is not None
        assert batched.argtypes == legacy.argtypes + [ctypes.c_longlong]
        assert legacy.restype is None
        assert batched.restype is None

    @pytest.mark.parametrize('entry', ['legacy', 'batched'])
    def test_empty_c_launch_is_noop(self, entry):
        if not ck.loaded():
            pytest.skip('Build the shared library with make kernel first.')
        legacy, batched = ck._select_kernel(np.dtype(np.float16))
        # The launch helper returns before touching the GPU for empty work.
        out = np.array([7.5], dtype=np.float16)
        ptr = ctypes.c_void_p(out.ctypes.data)
        if entry == 'legacy':
            legacy(ptr, ptr, ptr, ctypes.c_int(0))
        else:
            assert batched is not None
            batched(ptr, ptr, ptr, ctypes.c_int(1), ctypes.c_longlong(0))
        np.testing.assert_array_equal(out, [7.5])


@pytest.mark.parametrize('dim', [1, 2, 4, 8, 16, 32, 64])
def test_half_product_against_recursive(dim):
    a, b = coefficients((dim,), 1), coefficients((dim,), 2)
    assert_half(ck.multiply(a, b), reference(a, b))


def test_half_known_quaternion_product():
    a = np.array([1, 2, 3, 4], dtype=np.float16)
    b = np.array([4, 3, 2, 1], dtype=np.float16)
    np.testing.assert_array_equal(ck.multiply(a, b), [-12, 6, 24, 12])


@pytest.mark.parametrize('layout', ['broadcast', 'strided', 'empty'])
def test_half_batches(layout):
    a = coefficients((2, 1, 8), 3)
    b = coefficients((1, 3, 8), 4)
    if layout == 'strided':
        a, b = a[..., ::2], b[..., ::2]
    elif layout == 'empty':
        a = a[:0]
    result = ck.multiply(a, b)
    assert result.shape == np.broadcast_shapes(a.shape, b.shape)
    assert_half(result, reference(a, b))


@pytest.mark.parametrize('other', [np.float32, np.float64, np.complex64, np.int32])
@pytest.mark.parametrize('half_first', [True, False])
def test_half_dtype_promotion(other, half_first):
    a = np.array([1, 2], dtype=np.float16)
    b = np.array([3, 4], dtype=other)
    if np.dtype(other).kind == 'c':
        b += 1j
    if not half_first:
        a, b = b, a
    result = ck.multiply(a, b)
    assert result.dtype == np.result_type(a.dtype, b.dtype)
    # Explicit two-component product also handles complex promoted values.
    a, b = a.astype(result.dtype), b.astype(result.dtype)
    expected = [a[0] * b[0] - a[1] * b[1],
                a[0] * b[1] + a[1] * b[0]]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('entry', ['legacy', 'batched', 'fallback'])
def test_half_c_entry_points(entry, monkeypatch):
    cp = ck.cp
    a = cp.asarray(coefficients((3, 8), 5))
    b = cp.asarray(coefficients((3, 8), 6))
    out = cp.empty_like(a)
    legacy, batched = ck._select_kernel(a.dtype)
    cp.cuda.get_current_stream().synchronize()
    if entry == 'fallback':
        monkeypatch.setitem(ck._KERNELS, 'float16', (legacy, None))
        out = ck.multiply(a, b)
    elif entry == 'batched':
        assert batched is not None
        batched(a.data.ptr, b.data.ptr, out.data.ptr,
                ctypes.c_int(8), ctypes.c_longlong(3))
    else:
        for i in range(3):
            # This also verifies the ABI uses 2-byte coefficients.
            offset = i * 8 * a.dtype.itemsize
            legacy(a.data.ptr + offset, b.data.ptr + offset, out.data.ptr + offset,
                   ctypes.c_int(8))
    assert isinstance(out, cp.ndarray)
    assert_half(cp.asnumpy(out), reference(cp.asnumpy(a), cp.asnumpy(b)))


def test_half_python_list_with_typed_operand():
    a = np.array([1, 2], dtype=np.float16)
    assert_half(ck.multiply(a, [3, 4]), [-5, 10])


def test_half_float32_accumulation_retains_small_terms():
    a = np.ones(64, dtype=np.float16)
    a[0] = 2048
    b = -np.ones(64, dtype=np.float16)
    b[0] = 1
    result = ck.multiply(a, b)
    # 2048 + 63 = 2111 rounds to 2112 only on output. A half accumulator
    # would discard each +1 after 2048 and incorrectly return 2048.
    assert result[0] == np.float16(2112)


def test_half_products_do_not_overflow_before_cancellation():
    a = np.array([256, 256], dtype=np.float16)
    result = ck.multiply(a, a)
    # Both real terms are 65536: FP32 subtraction gives zero rather than
    # half's inf - inf. The other output genuinely exceeds FP16 range.
    np.testing.assert_array_equal(result, np.array([0, np.inf], dtype=np.float16))


@pytest.mark.parametrize('value', [2**-24, 2**-14, 65504])
def test_half_subnormals_and_maximum_survive_identity(value):
    a = np.array([value], dtype=np.float16)
    np.testing.assert_array_equal(ck.multiply(a, np.ones(1, dtype=np.float16)), a)


def test_half_output_rounds_to_nearest_even():
    # 1 + 2^-11 is halfway between 1 and the next FP16 value.
    a = np.array([1, 2**-11], dtype=np.float16)
    b = np.array([1, -1], dtype=np.float16)
    assert ck.multiply(a, b)[0] == np.float16(1)


def test_half_batch_grid_stride():
    a = np.full((65536, 1), 1.5, dtype=np.float16)
    b = np.full((1,), -2, dtype=np.float16)
    np.testing.assert_array_equal(ck.multiply(a, b), a * b)
