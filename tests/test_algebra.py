# SPDX-License-Identifier: Apache-2.0
"""Public algebra helpers, checked against an independent doubling oracle."""

import numpy as np
import pytest

import CayleyKernel as ck
from tests.test_complex import reference


@pytest.mark.cpu
class TestAlgebra:
    @pytest.mark.parametrize("dim", [1, 2, 4, 8, 16, 32])
    def test_conjugation_norm_and_inner(self, dim):
        rng = np.random.default_rng(dim)
        a = rng.normal(size=(2, 1, dim))
        b = rng.normal(size=(3, dim))
        star = ck.conjugate(a)
        np.testing.assert_array_equal(ck.conjugate(star), a)
        product = reference(a, star)
        np.testing.assert_allclose(product[..., 0], ck.norm_sq(a))
        np.testing.assert_allclose(product[..., 1:], 0, atol=1e-12)
        np.testing.assert_allclose(ck.inner(a, b), reference(a, ck.conjugate(b))[..., 0])
        np.testing.assert_allclose(ck.imag(a)[..., 1:], a[..., 1:])
        np.testing.assert_array_equal(ck.imag(a)[..., 0], 0)
        np.testing.assert_array_equal(ck.real(a), a[..., 0])

    @pytest.mark.parametrize("dim", [1, 2, 4, 8, 16, 32])
    def test_inverse_is_two_sided(self, dim):
        a = np.random.default_rng(dim).normal(size=(3, dim))
        if dim >= 16:
            with pytest.warns(RuntimeWarning, match="zero divisors"):
                inv = ck.inverse(a)
        else:
            inv = ck.inverse(a)
        identity = np.zeros_like(a)
        identity[..., 0] = 1
        np.testing.assert_allclose(reference(a, inv), identity, atol=1e-12)
        np.testing.assert_allclose(reference(inv, a), identity, atol=1e-12)

    def test_zero_divisor_warns_despite_positive_norm(self):
        a = np.zeros(16)
        b = np.zeros(16)
        a[[3, 10]] = 1
        b[6], b[15] = 1, -1
        np.testing.assert_array_equal(reference(a, b), 0)
        assert ck.norm_sq(a) == 2
        with pytest.warns(RuntimeWarning, match="cancellation"):
            ck.inverse(a)

    @pytest.mark.parametrize("a", [[0, 0], [[1, 0], [0, 0]], [1, 1j]])
    def test_zero_norm_rejected(self, a):
        with pytest.raises(ZeroDivisionError, match="zero norm_sq"):
            ck.inverse(a)

    def test_containers_and_no_mutation(self):
        a = [[1, 2], [3, 4]]
        assert ck.conjugate(a) == [[1, -2], [3, -4]]
        assert ck.norm_sq(a) == [5, 25]
        assert ck.inner(a, [2, 3]) == [8, 18]
        assert ck.real(a) == [1, 3]
        assert ck.imag(a) == [[0, 2], [0, 4]]
        assert ck.conjugate((1, 2)) == [1, -2]
        assert ck.conjugate(x=[1, 2]) == [1, -2]
        assert isinstance(ck.inverse(a), list)
        assert isinstance(ck.exp(a), list)
        assert a == [[1, 2], [3, 4]]
        x = np.arange(24.0).reshape(3, 8)[:, ::2]
        original = x.copy()
        for fn in (ck.conjugate, ck.imag, ck.real, ck.norm_sq, ck.inverse, ck.exp):
            fn(x)
        np.testing.assert_array_equal(x, original)

    @pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int8, np.int64, np.float16])
    def test_arithmetic_promotion(self, dtype):
        a = np.array([1, 1], dtype=dtype)
        np.testing.assert_array_equal(ck.conjugate(a), [1, -1])
        np.testing.assert_allclose(ck.inverse(a), [0.5, -0.5])
        assert ck.norm_sq(np.array([100, 100], dtype=np.int8)) == 20000
        assert ck.inner(np.array([100, 100], dtype=np.int8), [100, 100]) == 20000

    def test_complex_coefficients(self):
        a = np.array([1 + 2j, 3 - 4j])
        np.testing.assert_array_equal(ck.conjugate(a), [a[0], -a[1]])
        assert ck.norm_sq(a) == np.sum(a * a)
        assert ck.inner(a, a) == ck.norm_sq(a)
        assert ck.real(a) == a[0]
        np.testing.assert_allclose(reference(a, ck.inverse(a)), [1, 0], atol=1e-12)

    @pytest.mark.parametrize("a", [0, [], [1, 2, 3], np.empty((2, 0))])
    @pytest.mark.parametrize("fn", [ck.conjugate, ck.norm_sq, ck.inverse, ck.exp, ck.real, ck.imag])
    def test_invalid_dimensions(self, a, fn):
        with pytest.raises(ValueError, match="power-of-two"):
            fn(a)

    def test_inner_validation(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            ck.inner([1, 2], [1, 2, 3, 4])
        with pytest.raises(ValueError):
            ck.inner(np.ones((2, 4)), np.ones((3, 4)))

    @pytest.mark.parametrize("a", [["a", "b"], [object(), object()]])
    def test_nonnumeric_rejected(self, a):
        with pytest.raises(TypeError, match="numeric"):
            ck.norm_sq(a)

    def test_ordered_division(self, monkeypatch):
        monkeypatch.setattr(ck, "multiply", reference)
        a = np.array([1., 2., 3., 4.])
        b = np.array([[2., 1., -1., 3.], [3., 2., 1., 0.]])
        right = ck.right_div(a, b)
        left = ck.left_div(a, b)
        np.testing.assert_allclose(reference(right, b), np.broadcast_to(a, b.shape))
        np.testing.assert_allclose(reference(a, left), b, atol=1e-12)
        assert not np.allclose(right, left)

    @pytest.mark.parametrize("fn", [ck.right_div, ck.left_div])
    @pytest.mark.parametrize("array_first", [True, False])
    def test_division_mixed_integer_containers(self, monkeypatch, fn, array_first):
        def product(a, b):
            # In particular, inverse fractions must not inherit an integer
            # reference dtype through multiply's list conversion policy.
            assert a.dtype.kind == b.dtype.kind == "f"
            return reference(a, b)

        monkeypatch.setattr(ck, "multiply", product)
        a, b = np.array([1, 2]), [2, 1]
        if not array_first:
            a, b = a.tolist(), np.array(b)
        result = fn(a, b)
        assert isinstance(result, np.ndarray)
        x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        expected = reference(x, ck.inverse(y)) if fn is ck.right_div else reference(ck.inverse(x), y)
        np.testing.assert_allclose(result, expected)
        assert isinstance(fn([1, 2], [2, 1]), list)

    @pytest.mark.parametrize("n", [-5, -1, 0, 1, 2, 7, np.int64(13)])
    def test_power_against_repeated_products(self, monkeypatch, n):
        monkeypatch.setattr(ck, "multiply", reference)
        a = np.array([[0.2, 0.3, -0.1, 0.4], [1., 0., 0., 0.]])
        factor = ck.inverse(a) if n < 0 else a
        expected = np.zeros_like(a)
        expected[..., 0] = 1
        for _ in range(abs(n)):
            expected = reference(expected, factor)
        np.testing.assert_allclose(ck.power(a, n), expected, atol=1e-12)

    def test_power_contract(self, monkeypatch):
        calls = []

        def product(a, b):
            calls.append(1)
            return reference(a, b)

        monkeypatch.setattr(ck, "multiply", product)
        assert ck.power([0, 0], 0) == [1, 0]
        assert ck.power([1, 0], 1024) == [1, 0]
        assert len(calls) <= 12
        with pytest.raises(TypeError):
            ck.power([1, 0], 1.5)
        with pytest.raises(ZeroDivisionError):
            ck.power([0, 0], -1)

    @pytest.mark.parametrize("dim", [1, 2, 4, 8, 16, 32])
    def test_exp_against_series(self, dim):
        a = np.random.default_rng(dim).normal(size=(2, dim)) * 0.1
        expected = np.zeros_like(a)
        expected[..., 0] = 1
        term = expected.copy()
        for n in range(1, 25):
            term = reference(term, a) / n
            expected += term
        np.testing.assert_allclose(ck.exp(a), expected, atol=1e-12)

    def test_exp_scalar_small_and_complex(self):
        with np.errstate(all="raise"):
            np.testing.assert_allclose(ck.exp([[0, 0], [1, 0]]), [[1, 0], [np.e, 0]])
            np.testing.assert_allclose(ck.exp([0, 1e-12]), [1, 1e-12])
        # Nonzero imaginary vector with zero complex quadratic form.
        a = np.array([0, 1, 1j, 0])
        np.testing.assert_allclose(ck.exp(a), [1, 1, 1j, 0])
        z = np.array([0.2 + 0.3j, 0.4 - 0.1j])
        np.testing.assert_allclose(ck.exp(z), np.exp(z[0]) * np.array([np.cos(z[1]), np.sin(z[1])]))

    def test_empty_batch(self):
        a = np.empty((0, 4))
        assert ck.norm_sq(a).shape == (0,)
        assert ck.inverse(a).shape == (0, 4)
        assert ck.exp(a).shape == (0, 4)


@pytest.mark.cpu  # Check device prerequisites locally before the global GPU RNG fixture.
def test_gpu_helpers():
    cp = pytest.importorskip("cupy")
    if not ck.loaded():
        pytest.skip("GPU algebra helpers require the compiled kernel")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip("CUDA device unavailable: " + str(exc))
    a = cp.array([[1., 2., 3., 4.], [2., 0., 0., 0.]])
    b = cp.array([2., -1., 3., 1.])
    host_a, host_b = cp.asnumpy(a), cp.asnumpy(b)
    for fn in (ck.conjugate, ck.norm_sq, ck.inverse, ck.exp, ck.real, ck.imag):
        result = fn(a)
        assert isinstance(result, cp.ndarray)
        np.testing.assert_allclose(cp.asnumpy(result), fn(host_a))
    for fn in (ck.right_div, ck.left_div, ck.inner):
        result = fn(a, b)
        assert isinstance(result, cp.ndarray)
        expected = (reference(host_a, ck.inverse(host_b)) if fn is ck.right_div
                    else reference(ck.inverse(host_a), host_b) if fn is ck.left_div
                    else ck.inner(host_a, host_b))
        np.testing.assert_allclose(cp.asnumpy(result), expected, atol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(ck.power(a, 3)), reference(reference(host_a, host_a), host_a))
