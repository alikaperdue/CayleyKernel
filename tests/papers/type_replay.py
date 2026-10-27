"""Typed device replays of paper products; the oracle never calls the kernel.

The original result is returned so the paper's existing exact assertions keep
checking their original inputs. Additional replays check coefficient storage
and complexification, not extensions of real norm/topology theorems.
"""
import numpy as np
import CayleyKernel as CK


DTYPES = (np.float16, np.complex64, np.complex128)


def reference(a, b):
    """Independent four-product doubling, bilinear over central scalars."""
    if a.shape[-1] == 1:
        return a * b
    h = a.shape[-1] // 2
    x, y = a[..., :h], a[..., h:]
    u, v = b[..., :h], b[..., h:]

    def conjugate(z):
        out = z.copy()
        out[..., 1:] *= -1
        return out

    return np.concatenate((reference(x, u) - reference(conjugate(v), y),
                           reference(v, x) + reference(y, conjugate(u))), axis=-1)


def host(a):
    if CK.cp is not None and isinstance(a, CK.cp.ndarray):
        return CK.cp.asnumpy(a)
    return np.asarray(a)


def bounded(a):
    """Power-of-two rescaling bounds products without losing sparse zeros."""
    a = host(a)
    if np.iscomplexobj(a):
        raise ValueError('typed replay fixtures must start with real coefficients')
    a = a.astype(np.float64)
    largest = float(np.max(np.abs(a))) if a.size else 0.
    if largest > 1:
        a = np.ldexp(a, -int(np.ceil(np.log2(largest))))
    return a


def typed_cases(a, b):
    """References use the values actually representable in the input dtype."""
    a, b = bounded(a), bounded(b)
    for dtype in DTYPES:
        modes = ('real',) if dtype == np.float16 else ('phase', 'mixed')
        for mode in modes:
            if mode == 'phase':
                x, y = a * (1 + 1j), b * (1 - 2j)
            elif mode == 'mixed':
                # Independent imaginary coordinates exercise more than scalar
                # phase covariance, including a non-real scalar coordinate.
                x, y = a + 1j * np.roll(a, 1, axis=-1), b + 1j * np.roll(b, -1, axis=-1)
            else:
                x, y = a, b
            x, y = x.astype(dtype), y.astype(dtype)
            wide = np.float64 if dtype == np.float16 else np.complex128
            expected = reference(x.astype(wide), y.astype(wide)).astype(dtype)
            yield mode, x, y, expected


def replay_product(a, b):
    """Execute the original call plus half/complex replays on the real GPU."""
    original = CK.multiply(a, b)
    for mode, x, y, expected in typed_cases(a, b):
        actual = host(CK.multiply(x, y))
        assert actual.dtype == x.dtype, (mode, actual.dtype, x.dtype)
        if x.dtype == np.dtype(np.float16):
            rtol, atol = 1e-3, float(np.nextafter(np.float16(0), np.float16(1)))
        elif x.dtype == np.dtype(np.complex64):
            rtol, atol = 2e-5, 2e-5
        else:
            rtol, atol = 1e-12, 1e-12
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol,
                                   err_msg=f'{x.dtype} {mode} paper replay')
    return original


def require_gpu(case):
    if CK.cp is None:
        case.skipTest('CuPy unavailable; typed multiply() was not executed')
    if not CK.loaded():
        case.skipTest('compiled kernel unavailable; build with make kernel')
    try:
        count = CK.cp.cuda.runtime.getDeviceCount()
    except CK.cp.cuda.runtime.CUDARuntimeError as exc:
        case.skipTest('CUDA device unavailable: ' + str(exc))
    if not count:
        case.skipTest('no CUDA device')


def signature_fixture(gammas):
    """Transport a real +/- signature into the complexified standard algebra.

    Each +1 doubling generator maps to i times its standard generator; a
    basis monomial gets the product of its generator phases. This is a
    complex coordinate embedding, not a real signature flag on multiply().
    """
    from tests.papers.arxiv_2608_28176.algebra import Algebra, e
    A = Algebra(len(gammas), gammas, reference=True)
    phase = np.array([1j ** sum(bool(i & (1 << k)) for k, g in enumerate(gammas) if g == 1)
                      for i in range(A.dim)])
    eye = np.eye(A.dim)
    a, b = np.repeat(eye, A.dim, axis=0), np.tile(eye, (A.dim, 1))
    rng = np.random.default_rng(20261026)
    a = np.concatenate((a, rng.integers(-2, 3, (8, A.dim))))
    b = np.concatenate((b, rng.integers(-2, 3, (8, A.dim))))
    expected = np.zeros_like(a)
    for row, (x, y) in enumerate(zip(a, b)):
        product = A.mul({i: int(c) for i, c in enumerate(x) if c},
                        {i: int(c) for i, c in enumerate(y) if c})
        for i, c in product.items():
            expected[row, i] = c
    return a * phase, b * phase, expected * phase


def replay_signatures(case, signatures):
    require_gpu(case)
    for gammas in signatures:
        a, b, expected = signature_fixture(gammas)
        for dtype in (np.complex64, np.complex128):
            with case.subTest(gammas=gammas, dtype=np.dtype(dtype).name):
                actual = host(CK.multiply(a.astype(dtype), b.astype(dtype)))
                case.assertEqual(actual.dtype, np.dtype(dtype))
                np.testing.assert_array_equal(actual, expected.astype(dtype))
