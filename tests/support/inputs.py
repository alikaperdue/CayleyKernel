# SPDX-License-Identifier: Apache-2.0
"""Deterministic inputs for shared numerical and graph tests."""

import random

import numpy as np


def normal_coefficients(shape, dtype, seed, *, divisor=1):
    """Draw real and, for complex dtypes, independent imaginary coefficients."""
    rng = np.random.default_rng(seed)
    real = rng.normal(size=shape)
    if divisor != 1:
        real /= divisor
    if np.dtype(dtype).kind == "c":
        imag = rng.normal(size=shape)
        if divisor != 1:
            imag /= divisor
        real = real + 1j * imag
    return real.astype(dtype)


def random_sparse(dim, k, rng):
    """Draw distinct indices with small signed integers for exact products."""
    ix = rng.choice(dim, size=k, replace=False)
    vx = rng.integers(1, 9, size=k) * rng.choice([-1, 1], size=k)
    return ix, vx


def small_anchor():
    """Return the fixed dimension-16 clique used in graph search tests."""
    from scripts import relation_graphs as rg

    vertices, _ = rg.candidates(4, 14, random.Random(1), xor_mask=9)
    graph = rg.build_graphs(vertices, ["orthogonality"])["orthogonality"]
    clique = rg.maximum_clique(graph, 5)
    return [vertices[i] for i in clique["vertices"]]
