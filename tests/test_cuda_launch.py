"""Launch geometry, shared-memory tails and grid-stride coverage."""

from types import SimpleNamespace

import numpy as np
import pytest

import CayleyKernel as ck

BLOCK_SIZES = (32, 64, 128, 256, 512, 1024)


@pytest.mark.cpu
@pytest.mark.parametrize('value', ['', '0', '16', '33', '2048', '-32', '256.0', ' 256'])
def test_invalid_block_size_rejected_before_launch(monkeypatch, value):
    monkeypatch.setenv('CAYLEY_KERNEL_BLOCK_SIZE', value)
    with pytest.raises(ValueError, match='CAYLEY_KERNEL_BLOCK_SIZE'):
        ck._launch((None, None), None, None, None, 4, 1)


@pytest.mark.cpu
@pytest.mark.parametrize('value', [None, *map(str, BLOCK_SIZES)])
def test_block_size_can_change_between_calls(monkeypatch, value):
    if value is None:
        monkeypatch.delenv('CAYLEY_KERNEL_BLOCK_SIZE', raising=False)
    else:
        monkeypatch.setenv('CAYLEY_KERNEL_BLOCK_SIZE', value)
    calls = []
    array = SimpleNamespace(data=SimpleNamespace(ptr=123))
    ck._launch((None, lambda *args: calls.append(args)), array, array, array, 4, 37)
    assert len(calls) == 1
    assert calls[0][-2].value == 4
    assert calls[0][-1].value == 37


def reference(a, b):
    """Independent recursive doubling, vectorized over leading batch axes."""
    if a.shape[-1] == 1:
        return a * b
    mid = a.shape[-1] // 2
    x, y = a[..., :mid], a[..., mid:]
    u, v = b[..., :mid], b[..., mid:]

    def conj(z):
        z = -z.copy()
        z[..., 0] = -z[..., 0]
        return z

    return np.concatenate((reference(x, u) - reference(conj(v), y),
                           reference(v, x) + reference(y, conj(u))), axis=-1)


@pytest.mark.parametrize('threads', BLOCK_SIZES)
@pytest.mark.parametrize('dim', [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize('dtype', [
    'float16', 'float32', 'float64', 'complex64', 'complex128',
    'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool',
])
def test_all_block_sizes_against_recursive(monkeypatch, threads, dim, dtype):
    monkeypatch.setenv('CAYLEY_KERNEL_BLOCK_SIZE', str(threads))
    rng = np.random.default_rng(2026)
    # Small integral coefficients make the reference exact for every dtype.
    lower = 0 if np.dtype(dtype).kind in 'ub' else -2
    a = rng.integers(lower, 3, size=(37, dim)).astype(dtype)
    b = rng.integers(lower, 3, size=(37, dim)).astype(dtype)
    if np.dtype(dtype).kind == 'c':
        a += 1j * rng.integers(-2, 3, size=a.shape)
        b += 1j * rng.integers(-2, 3, size=b.shape)
    acc_dtype = np.complex128 if np.dtype(dtype).kind == 'c' else np.float64
    expected = reference(a.astype(acc_dtype), b.astype(acc_dtype)).astype(dtype)
    np.testing.assert_array_equal(ck.multiply(a, b), expected)


@pytest.mark.parametrize('dim', [1, 4, 16, 32, 1024])
def test_flat_grid_stride_and_partial_last_tile(monkeypatch, dim):
    monkeypatch.setenv('CAYLEY_KERNEL_BLOCK_SIZE', '32')
    # Force a second grid-stride iteration, including a partial final block.
    batch = (65535 * 32) // dim + 3
    a = (np.arange(batch * dim) % 31 - 15).reshape(batch, dim).astype(np.float32)
    b = np.zeros_like(a)
    b[:, 0] = 1
    np.testing.assert_array_equal(ck.multiply(a, b), a)


@pytest.mark.parametrize('batch', [4097, 65535, 65536, 70001])
def test_midrange_and_large_batches(monkeypatch, batch):
    monkeypatch.delenv('CAYLEY_KERNEL_BLOCK_SIZE', raising=False)
    rng = np.random.default_rng(987)
    a = rng.normal(size=(batch, 8)).astype(np.float64)
    b = rng.normal(size=(batch, 8)).astype(np.float64)
    np.testing.assert_allclose(ck.multiply(a, b), reference(a, b), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize('threads', BLOCK_SIZES)
@pytest.mark.parametrize('dim', [1, 16, 32])
def test_bfloat16_launch_sizes(monkeypatch, threads, dim):
    bf16 = pytest.importorskip('ml_dtypes').bfloat16
    if int(ck.cp.__version__.split('.')[0]) < 14:
        pytest.skip('BF16 requires CuPy 14+')
    monkeypatch.setenv('CAYLEY_KERNEL_BLOCK_SIZE', str(threads))
    rng = np.random.default_rng(42)
    a = (rng.integers(-8, 9, size=(37, dim)) / 8).astype(bf16)
    b = (rng.integers(-8, 9, size=(37, dim)) / 8).astype(bf16)
    expected = reference(a.astype(np.float64), b.astype(np.float64)).astype(bf16)
    actual = ck.multiply(a, b)
    np.testing.assert_array_equal(actual.astype(np.float32), expected.astype(np.float32))
