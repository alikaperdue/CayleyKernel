# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Shared pytest fixtures and guards for the CayleyKernel test suite."""

import os
import sys

import pytest

# Make the repository root importable regardless of where pytest is launched
# from, so that `import CayleyKernel` resolves to the local package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cupy as cp
except Exception:
    cp = None

import CayleyKernel


def pytest_configure(config):
    # Register custom markers (also declared in pyproject.toml).
    config.addinivalue_line(
        "markers", "slow: heavy 2^20+ identity/batch tests (pytest -m slow)"
    )
    config.addinivalue_line(
        "markers", "benchmark: timing gates and scaling (pytest -m benchmark)"
    )


@pytest.fixture(autouse=True)
def _seed():
    if np is not None:
        np.random.seed(0)
    if cp is not None:
        cp.random.seed(0)


@pytest.fixture(autouse=True)
def _require_gpu_kernel():
    """Skip the whole suite when the GPU stack is unavailable.

    A fresh clone has no compiled kernel (it is built with `make kernel`).
    Skipping instead of failing keeps `pytest` green in environments
    without nvcc/CUDA, while the message tells developers exactly what
    to run.
    """
    if cp is None:
        pytest.skip("CuPy/CUDA runtime not available on this machine.")
    if not CayleyKernel.loaded():
        pytest.skip(
            "cuda_kernel.so not found -- build it first:  make kernel"
        )
