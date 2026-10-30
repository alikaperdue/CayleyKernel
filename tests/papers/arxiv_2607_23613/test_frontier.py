# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the exploratory solver, not universal existence claims."""
import unittest
try:
    import numpy as np
    import scipy
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(str(exc))
import pytest
from scripts.paper_quadratic_frontier import (
    Products, reference_product, square, square_derivative, square_root,
    newton, root_experiment, eigen_experiment,
)

pytestmark = pytest.mark.cpu


def test_signed_permutation_against_recursive_doubling():
    rng = np.random.default_rng(71)
    for level in (2, 3, 4, 6):
        dim = 1 << level
        P = Products(dim, 'cpu')
        a = np.zeros(dim)
        a[[0, 1, dim-1]] = [0.5, -2, 3]
        x = rng.normal(size=dim)
        np.testing.assert_allclose(P.fixed(a)(x), reference_product(a, x), atol=1e-12)
        np.testing.assert_allclose(P.fixed(a, False)(x), reference_product(x, a), atol=1e-12)
        np.testing.assert_allclose(square(x), reference_product(x, x), atol=1e-12)


def test_square_root_negative_scalar_and_near_negative_axis():
    for w in (np.array([-4.,0.,0.,0.]), np.array([-4.,1e-12,0.,0.]),
              np.array([4.,0.,0.,0.]), np.array([0.,1.,2.,-3.])):
        np.testing.assert_allclose(reference_product(square_root(w), square_root(w)), w, atol=1e-12)


def test_derivative_matches_independent_product():
    rng = np.random.default_rng(73)
    x, h = rng.normal(size=(2, 16))
    expected = reference_product(x, h)+reference_product(h, x)
    np.testing.assert_allclose(square_derivative(x, h), expected, atol=1e-12)


def test_unplanted_sedenion_root_checked_by_full_product():
    result = root_experiment(4, 27613, 'cpu', 0.5, 1e-9)
    assert all(b['status'] == 'root_verified' for b in result['branches'])
    assert all(b['direct_product_residual'] < 1e-8 for b in result['branches'])


def test_octonion_eigenvalue_pipeline():
    result = eigen_experiment(3, 26613, 'cpu', 1e-9)
    assert result['transformed_converged']
    assert result['transformation_original_residual'] < 1e-8
    assert result['status'] == 'eigenpair_verified'


def test_stalled_solver_is_not_a_root():
    x, info = newton(lambda x: np.ones(2), lambda x,h: np.zeros(2), np.zeros(2))
    assert not info['converged']
    assert info['residuals'][-1] > 1


def test_sedenion_transformation_failure_and_direct_recovery():
    result = eigen_experiment(4, 27613, 'cpu', 1e-9)
    assert result['transformed_converged']
    assert result['transformed_residual'] < 1e-8
    assert result['transformation_original_residual'] > 0.1
    assert result['status'] == 'eigenpair_verified'
    assert result['eigenpair_residual'] < 1e-8


def test_gpu_root_and_eigen_pipeline():
    import CayleyKernel as CK
    if CK.cp is None or not CK.loaded():
        pytest.skip('CuPy/kernel unavailable')
    try:
        devices = CK.cp.cuda.runtime.getDeviceCount()
    except CK.cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(str(exc))
    if not devices:
        pytest.skip('No CUDA device')
    roots = root_experiment(4, 27613, 'gpu', 0.5, 1e-9)
    assert all(b['status'] == 'root_verified' for b in roots['branches'])
    eigen = eigen_experiment(3, 26613, 'gpu', 1e-9)
    assert eigen['status'] == 'eigenpair_verified'


def test_saved_vectors_reproduce_original_equation(tmp_path, monkeypatch):
    import hashlib
    import json
    from scripts.paper_quadratic_frontier import main
    output = tmp_path/'run'
    monkeypatch.setattr('sys.argv', ['frontier', '--levels', '3', '--samples', '1', '--output', str(output)])
    main()
    report = json.loads((output/'results.json').read_text())
    row = report['runs'][0]
    vector_path = output/row['vector_file']
    assert hashlib.sha256(vector_path.read_bytes()).hexdigest() == row['vector_sha256']
    data = np.load(vector_path)
    coefficients = []
    for name in 'abc':
        v = np.zeros(8)
        for index, value in row['coefficients'][name].items():
            v[int(index)] = value
        coefficients.append(v)
    a, b, c = coefficients
    for key in ('root_0', 'root_1'):
        q = data[key]
        residual = reference_product(q, q) + row['strength']*(
            reference_product(a, reference_product(q, b))+reference_product(c, q))+data['constant']
        assert np.linalg.norm(residual) < 1e-8
    # Refuse overwriting saved evidence.
    before = (output/'results.json').read_bytes()
    with pytest.raises(SystemExit):
        main()
    assert (output/'results.json').read_bytes() == before
