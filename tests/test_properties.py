"""Public CPU/CUDA products checked with generated and explicit edge cases."""

import numpy as np
import pytest

pytest.importorskip("hypothesis", reason="install the test extra for property tests")
from hypothesis import example, given, settings, strategies as st

import CayleyKernel as ck


@pytest.fixture(scope="session", params=[pytest.param("cpu", marks=pytest.mark.cpu), "cuda"])
def product(request):
    if request.param == "cpu":
        return ck.cpu_multiply
    return ck.multiply


def conjugate(a):
    result = -a.copy()
    result[0] = a[0]
    return result


def recursive_product(a, b):
    if len(a) == 1:
        return a * b
    half = len(a) // 2
    p, q, r, s = a[:half], a[half:], b[:half], b[half:]
    return np.concatenate((recursive_product(p, r) - recursive_product(conjugate(s), q),
                           recursive_product(s, p) + recursive_product(q, conjugate(r))))


@st.composite
def vector_pairs(draw, max_level=5, special=False):
    dim = 1 << draw(st.integers(0, max_level))
    values = st.floats(-8, 8, allow_nan=False, allow_infinity=False)
    if special:
        values = st.one_of(values, st.sampled_from([
            0.0, -0.0, 1.0, -1.0, np.nextafter(0.0, 1.0),
            -np.nextafter(0.0, 1.0), float("nan"), float("inf"), -float("inf"),
        ]))
    return (draw(st.lists(values, min_size=dim, max_size=dim)),
            draw(st.lists(values, min_size=dim, max_size=dim)))


@settings(max_examples=60, deadline=None)
@given(pair=vector_pairs())
@example(pair=([0.0] * 16, [0.0] * 16))
@example(pair=([1.0] * 32, [1.0] * 32))
@example(pair=([np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0)], [1.0, 0.0]))
def test_finite_identity_zero_and_reference(product, pair):
    a, b = (np.asarray(v, dtype=np.float64) for v in pair)
    one, zero = np.zeros_like(a), np.zeros_like(a)
    one[0] = 1
    for left, right in ((a, one), (one, a)):
        np.testing.assert_array_equal(product(left, right), a)
    for left, right in ((a, zero), (zero, a)):
        np.testing.assert_array_equal(product(left, right), zero)
    np.testing.assert_allclose(product(a, b), recursive_product(a, b), rtol=1e-10, atol=1e-11)


@settings(max_examples=60, deadline=None)
@given(pair=vector_pairs(max_level=3))
def test_norm_multiplicativity_in_normed_algebras(product, pair):
    a, b = (np.asarray(v, dtype=np.float64) for v in pair)
    result = np.asarray(product(a, b))
    np.testing.assert_allclose(result @ result, (a @ a) * (b @ b), rtol=1e-10, atol=1e-11)


@settings(max_examples=60, deadline=None)
@given(pair=vector_pairs(max_level=3, special=True))
@example(pair=([float("nan"), 1.0], [1.0, 0.0]))
@example(pair=([float("inf"), -float("inf")], [1.0, 1.0]))
@example(pair=([float("inf")], [0.0]))
def test_ieee_special_values_against_reference(product, pair):
    a, b = (np.asarray(v, dtype=np.float64) for v in pair)
    # Identities such as a*1=a are invalid for nonfinite coefficients: 0*Inf=NaN.
    with np.errstate(invalid="ignore", under="ignore"):
        np.testing.assert_allclose(product(a, b), recursive_product(a, b),
                                   rtol=1e-10, atol=1e-11, equal_nan=True)
