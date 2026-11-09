# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yotta Space Corp
"""Measure the PyTorch adapter overhead relative to CuPy."""

import pytest

import CayleyKernel as ck
from tests.performance.benchmark_support import PROTOCOL, measure_workloads

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.usefixtures("torch_cuda")


@pytest.mark.benchmark
def test_torch_overhead_vs_pure_cupy():
    """Apply a local ceiling of 3x the CuPy median to the Torch adapter."""
    dim, batch = 32, 4096
    device = torch.cuda.current_device()
    generator = torch.Generator(device=f"cuda:{device}").manual_seed(PROTOCOL["seed"])
    a_t = torch.randn(batch, dim, device=f"cuda:{device}", dtype=torch.float32, generator=generator)
    b_t = torch.randn(batch, dim, device=f"cuda:{device}", dtype=torch.float32, generator=generator)

    with ck.cp.cuda.Device(device):
        a_c = ck.torch._torch_to_cupy(a_t)
        b_c = ck.torch._torch_to_cupy(b_t)
        print(f"\nWorkload: batch={batch}, dim={dim}, dtype=float32")
        cases = measure_workloads(
            {
                "cupy": lambda: ck.multiply(a_c, b_c),
                "torch": lambda: ck.torch.multiply(a_t, b_t),
            },
            lambda: torch.cuda.synchronize(device),
        )
    ratio = cases["torch"]["median_seconds"] / cases["cupy"]["median_seconds"]
    print(f"Torch/CuPy median ratio: {ratio:.2f}x (limit 3.0x)")
    assert ratio <= 3.0, (
        f"Torch/CuPy median ratio {ratio:.2f}x exceeds the 3.0x adapter overhead limit. "
        "Check GPU load and repeat before diagnosing a regression."
    )
