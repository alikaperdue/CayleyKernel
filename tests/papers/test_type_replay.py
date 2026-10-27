"""CPU validation of the typed fixtures and their independent expectations."""
from itertools import product

import numpy as np
import pytest

from tests.papers.arxiv_2608_28176.algebra import Algebra
from tests.papers.type_replay import bounded, reference, signature_fixture, typed_cases

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize('dim', [1, 2, 8, 16, 64])
def test_typed_oracle_against_sparse_doubling(dim):
    rng = np.random.default_rng(2026)
    a = rng.integers(-2, 3, (2, 1, dim))
    b = rng.integers(-2, 3, (1, 3, dim))
    A = Algebra(dim.bit_length() - 1, reference=True)
    cases = list(typed_cases(a, b))
    assert len(cases) == 5
    for mode, x, y, expected in cases:
        assert x.dtype == y.dtype == expected.dtype
        x, y = np.broadcast_arrays(x, y)
        assert expected.shape == (2, 3, dim)
        for index in np.ndindex(x.shape[:-1]):
            sparse = A.mul(dict(enumerate(x[index].astype(np.complex128))),
                           dict(enumerate(y[index].astype(np.complex128))))
            dense = np.array([sparse.get(i, 0) for i in range(dim)])
            if expected.dtype.kind != 'c':
                dense = dense.real
            np.testing.assert_array_equal(expected[index], dense.astype(expected.dtype))
        if mode != 'real':
            assert np.any(x.imag) and np.any(y.imag)


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_complex_signature_transport_against_standard_doubling(n):
    for gammas in product((-1, 1), repeat=n):
        a, b, expected = signature_fixture(gammas)
        np.testing.assert_array_equal(reference(a, b), expected)


def test_phase_replay_preserves_zero_divisors_and_nonzero_controls():
    a, b = np.zeros(16), np.zeros(16)
    a[[1, 15]], b[[2, 12]] = [1, -1], [1, 1]
    assert np.any(reference(a, a))
    for mode, x, y, expected in typed_cases(a, b):
        if mode in ('real', 'phase'):
            np.testing.assert_array_equal(expected, 0)


def test_half_protocol_bounds_large_power_fixtures_and_keeps_dyadics():
    a = np.array([2.**200, -2.**199, 0, 2.**198])
    np.testing.assert_array_equal(bounded(a), [1, -.5, 0, .25])
    mode, x, y, expected = next(typed_cases(a, a))
    assert mode == 'real' and x.dtype == np.float16
    assert np.all(np.isfinite(expected))
    # The oracle quantizes operands before computing the expected product.
    a = np.array([1/3, 1/7])
    _, x, y, expected = next(typed_cases(a, a))
    np.testing.assert_array_equal(expected, reference(x.astype(float), y.astype(float)).astype(np.float16))


def test_replay_dispatch_checks_all_types_and_retains_original_result(monkeypatch):
    import CayleyKernel as CK
    from tests.papers.type_replay import replay_product
    calls = []

    def cpu_test_double(a, b):
        # Validate wrapper dispatch only; this is not a device execution test.
        calls.append(np.asarray(a).dtype)
        dtype = np.result_type(a, b)
        wide = np.complex128 if dtype.kind == 'c' else np.float64
        return reference(np.asarray(a, dtype=wide), np.asarray(b, dtype=wide)).astype(dtype)

    monkeypatch.setattr(CK, 'multiply', cpu_test_double)
    a = np.arange(8, dtype=float)
    result = replay_product(a, a)
    np.testing.assert_array_equal(result, reference(a, a))
    assert calls == [np.dtype(t) for t in (np.float64, np.float16, np.complex64,
                                         np.complex64, np.complex128, np.complex128)]


def test_replay_rejects_coefficient_conjugation(monkeypatch):
    import CayleyKernel as CK
    from tests.papers.type_replay import replay_product

    def wrong_product(a, b):
        dtype = np.result_type(a, b)
        wide = np.complex128 if dtype.kind == 'c' else np.float64
        return reference(np.asarray(a, dtype=wide).conjugate(),
                         np.asarray(b, dtype=wide)).astype(dtype)

    monkeypatch.setattr(CK, 'multiply', wrong_product)
    with pytest.raises(AssertionError, match='complex64 phase paper replay'):
        replay_product(np.arange(8, dtype=float), np.ones(8))
