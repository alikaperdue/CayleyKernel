"""BF16 storage/ABI, FP32 arithmetic, and wrapper promotion regressions."""

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

import CayleyKernel as ck

# Mark CPU checks individually so CUDA execution tests are excluded by -m cpu.
# CPU checks remain runnable without CuPy, ml_dtypes, or an NVIDIA driver.


@pytest.fixture
def bf16():
    return pytest.importorskip('ml_dtypes').bfloat16


@pytest.fixture
def gpu(bf16):
    cp = ck.cp
    if cp is None or int(cp.__version__.split('.')[0]) < 14:
        pytest.skip('BF16 GPU tests require CuPy 14+ and ml_dtypes')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('No CUDA device')
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip('CUDA device unavailable: ' + str(exc))
    if not ck.loaded():
        pytest.fail('Build the CUDA library first: make kernel ARCHS="sm_120"')
    # A stale build is a failure, not an optional-dependency skip.
    ck._select_kernel(np.dtype(bf16))
    return cp


def reference(a, b):
    """Independent recursive doubling product with float64 coefficients."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape[-1] == 1:
        return a * b
    m = a.shape[-1] // 2
    x, y, u, v = a[..., :m], a[..., m:], b[..., :m], b[..., m:]

    def conj(z):
        z = -z.copy()
        z[..., 0] *= -1
        return z

    return np.concatenate((reference(x, u) - reference(conj(v), y),
                           reference(v, x) + reference(y, conj(u))), axis=-1)


@pytest.mark.cpu
def test_registry_and_missing_library(monkeypatch):
    assert ck._DTYPE_TO_SUFFIX['bfloat16'] == 'bfloat16'
    assert 'bfloat16' in ck._TYPE_SUFFIXES
    monkeypatch.setattr(ck, '_KERNELS', {})
    with pytest.raises(TypeError, match='Rebuild the kernel'):
        ck._select_kernel(SimpleNamespace(name='bfloat16'))


@pytest.mark.cpu
def test_library_abi_and_empty_launch():
    if not ck.loaded():
        pytest.skip('Build the shared library first: make kernel')
    legacy, batched = ck._select_kernel(SimpleNamespace(name='bfloat16'))
    assert legacy.argtypes == [ctypes.c_void_p] * 3 + [ctypes.c_int]
    assert batched is not None
    assert batched.argtypes == legacy.argtypes + [ctypes.c_longlong]
    assert legacy.restype is None and batched.restype is None
    # No GPU access occurs for empty work, including when pointers are null.
    legacy(None, None, None, 0)
    batched(None, None, None, 1, 0)


@pytest.mark.cpu
@pytest.mark.parametrize('other', [np.float16, np.float32, np.float64,
                                        np.int8, np.int64, np.complex64, np.bool_])
@pytest.mark.parametrize('reverse', [False, True])
def test_mixed_promotion(monkeypatch, bf16, other, reverse):
    monkeypatch.setattr(ck, 'cp', np)
    a, b = np.dtype(bf16), np.dtype(other)
    if reverse:
        a, b = b, a
    assert ck._result_dtype(a, b) == np.result_type(np.float32, other)


@pytest.mark.cpu
@pytest.mark.parametrize('values', [[1, -2], [1.5, -2.25]])
def test_same_dtype_and_lists(monkeypatch, bf16, values):
    monkeypatch.setattr(ck, 'cp', np)
    dtype = np.dtype(bf16)
    assert dtype.itemsize == 2
    assert ck._result_dtype(dtype, dtype) == dtype
    result = ck._as_cupy(values, dtype)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result.astype(np.float32), values)
    result = ck._as_cupy([1 + 2j], dtype)
    assert result.dtype.kind == 'c'
    assert result[0] == 1 + 2j


@pytest.mark.cpu
def test_cupy_dtype_metadata(bf16):
    if ck.cp is None or int(ck.cp.__version__.split('.')[0]) < 14:
        pytest.skip('Requires CuPy 14+; no GPU is needed for dtype metadata')
    dtype = ck.cp.dtype(bf16)
    assert ck._result_dtype(dtype, dtype) == dtype
    for other in (np.float16, np.float64, np.int32, np.complex64):
        assert ck._result_dtype(dtype, ck.cp.dtype(other)) == ck.cp.result_type(np.float32, other)


@pytest.mark.cpu
def test_algebra_helpers(bf16):
    a = np.array([1, 2, 3, 4], dtype=bf16)
    np.testing.assert_array_equal(ck.conjugate(a).astype(np.float32), [1, -2, -3, -4])
    assert ck.norm_sq(a).dtype == np.dtype(np.float32)
    assert ck.norm_sq(a) == 30
    np.testing.assert_allclose(ck.inverse(a), np.array([1, -2, -3, -4]) / 30)
    assert ck.inner(a, a) == 30
    assert ck.exp(a).dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(ck.imag(a).astype(np.float32), [0, 2, 3, 4])


@pytest.mark.parametrize('dim', [1, 2, 4, 8, 16, 32, 64])
def test_product(gpu, bf16, dim):
    rng = np.random.default_rng(42)
    # Dyadic inputs make the FP32 reference exact and isolate final rounding.
    a = (rng.integers(-8, 9, size=(3, dim)) / 8).astype(bf16)
    b = (rng.integers(-8, 9, size=(3, dim)) / 8).astype(bf16)
    actual = ck.multiply(a, b)
    assert actual.dtype == np.dtype(bf16)
    np.testing.assert_array_equal(actual.astype(np.float32),
                                  reference(a, b).astype(bf16).astype(np.float32))


@pytest.mark.parametrize('layout', ['broadcast', 'strided', 'empty'])
def test_batches(gpu, bf16, layout):
    a = np.arange(16, dtype=np.float32).reshape(2, 1, 8).astype(bf16)
    b = np.ones((1, 3, 8), dtype=bf16)
    if layout == 'strided':
        a, b = a[..., ::2], b[..., ::2]
    elif layout == 'empty':
        a = a[:0]
    actual = ck.multiply(gpu.asarray(a), gpu.asarray(b))
    assert actual.dtype == np.dtype(bf16)
    assert actual.shape == np.broadcast_shapes(a.shape, b.shape)
    np.testing.assert_array_equal(gpu.asnumpy(actual).astype(np.float32),
                                  reference(a, b).astype(bf16).astype(np.float32))


@pytest.mark.parametrize('other', [np.float16, np.float32, np.float64, np.int32, np.complex64])
@pytest.mark.parametrize('reverse', [False, True])
def test_gpu_mixed_promotion(gpu, bf16, other, reverse):
    a = np.array([1, 2], dtype=bf16)
    b = np.array([3, 4], dtype=other)
    if np.dtype(other).kind == 'c':
        b += 1j
    if reverse:
        a, b = b, a
    result = ck.multiply(a, b)
    assert result.dtype == np.result_type(np.float32, other)
    a, b = a.astype(result.dtype), b.astype(result.dtype)
    np.testing.assert_array_equal(result, [a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]])


@pytest.mark.parametrize('entry', ['legacy', 'batched', 'fallback'])
def test_c_entry_points(gpu, bf16, entry, monkeypatch):
    a = gpu.asarray(np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=bf16))
    b = gpu.asarray(np.array([4, 3, 2, 1], dtype=bf16))
    b = gpu.ascontiguousarray(gpu.broadcast_to(b, a.shape))
    out = gpu.empty_like(a)
    legacy, batched = ck._select_kernel(a.dtype)
    assert legacy.argtypes == [ctypes.c_void_p] * 3 + [ctypes.c_int]
    assert batched.argtypes == legacy.argtypes + [ctypes.c_longlong]
    gpu.cuda.get_current_stream().synchronize()
    if entry == 'fallback':
        monkeypatch.setitem(ck._KERNELS, 'bfloat16', (legacy, None))
        out = ck.multiply(a, b)
    elif entry == 'batched':
        batched(a.data.ptr, b.data.ptr, out.data.ptr, 4, 2)
    else:
        for i in range(2):
            offset = i * 4 * 2  # Explicitly exercise the 2-byte ABI.
            legacy(a.data.ptr + offset, b.data.ptr + offset, out.data.ptr + offset, 4)
    np.testing.assert_array_equal(gpu.asnumpy(out).astype(np.float32),
                                  reference(gpu.asnumpy(a), gpu.asnumpy(b)))


def test_accumulation_rounding_and_range(gpu, bf16):
    a = np.ones(64, dtype=bf16)
    a[0] = 256
    b = -np.ones(64, dtype=bf16)
    b[0] = 1
    assert float(ck.multiply(a, b)[0]) == 320  # 319 rounds once to BF16.
    a = np.array([1, 2**-8], dtype=bf16)
    assert float(ck.multiply(a, np.array([1, -1], dtype=bf16))[0]) == 1
    # Values outside FP16 range, but safe for FP32 products.
    # ml_dtypes versions can reject Python integers larger than int64.
    a = np.array([2.0**-100, 2.0**100], dtype=bf16)
    result = ck.multiply(a, np.array([1, 0], dtype=bf16))
    np.testing.assert_array_equal(result.astype(np.float32), a.astype(np.float32))
    assert ck.multiply(np.array([1, 2], dtype=bf16), [3, 4]).dtype == np.dtype(bf16)


def test_grid_stride(gpu, bf16):
    a = gpu.asarray(np.full((65536, 1), 1.5, dtype=bf16))
    b = gpu.asarray(np.array([-2], dtype=bf16))
    actual = ck.multiply(a, b)
    assert actual.dtype == np.dtype(bf16)
    np.testing.assert_array_equal(gpu.asnumpy(actual).astype(np.float32), -3)
