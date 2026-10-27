"""Exact arithmetic and recording shared by the Desktop paper companions.

No changes to CayleyKernel; basis_product is an adapter to the actual public
API (currently named basis). Sparse products never stand in for GPU execution.
"""
import math
import unittest
import CayleyKernel as CK
from tests.papers.type_replay import replay_product
from tests.papers.arxiv_2608_28176.algebra import (
    Algebra, add, conj, e, neg, scale, pair, rank, nullity, recursive_basis,
)

basis_product = getattr(CK, 'basis_product', CK.basis)


def mul(x, y):
    out = {}
    for i, a in x.items():
        for j, b in y.items():
            label = basis_product(i + 1, j + 1)
            k = abs(label) - 1
            out[k] = out.get(k, 0) + (1 if label > 0 else -1) * a * b
    return {i: a for i, a in out.items() if a}


def phase(t, unit=1):
    return {0: math.cos(t), unit: math.sin(t)}


def close(case, x, y, tol=1e-12):
    for k in x.keys() | y.keys():
        case.assertAlmostEqual(x.get(k, 0), y.get(k, 0), delta=tol)


def gpu_replay(case, vectors, dim):
    if CK.cp is None:
        case.skipTest('CuPy is unavailable; multiply() was not executed')
    if not CK.loaded():
        case.skipTest('compiled kernel unavailable')
    try:
        count = CK.cp.cuda.runtime.getDeviceCount()
    except CK.cp.cuda.runtime.CUDARuntimeError as exc:
        case.skipTest('CUDA device unavailable: ' + str(exc))
    if not count:
        case.skipTest('no CUDA device')
    import numpy as np
    a = np.zeros((len(vectors), dim), dtype=np.float64)
    b = np.zeros_like(a)
    expected = np.zeros_like(a)
    for row, (x, y) in enumerate(vectors):
        for k, c in x.items(): a[row, k] = c
        for k, c in y.items(): b[row, k] = c
        for k, c in mul(x, y).items(): expected[row, k] = c
    actual = CK.cp.asnumpy(replay_product(a, b))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
