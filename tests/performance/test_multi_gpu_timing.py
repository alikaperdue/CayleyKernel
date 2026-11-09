# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Yotta Space Corp
"""Measure the overlap between CayleyKernel launches on two GPUs."""

import pytest

from CayleyKernel import _distributed, cp
from tests.performance.benchmark_support import PROTOCOL, device_topology, measure_workloads

pytestmark = pytest.mark.usefixtures("two_gpu_devices")


@pytest.mark.benchmark
def test_split_batch_overlap_timing(monkeypatch):
    """Compare split_batch with and without a per-launch device barrier."""
    dim = 32
    batch = 2**20
    devices = _distributed.visible_devices()[:2]

    # Put both operands and the gathered result on the first participating device.
    with cp.cuda.Device(devices[0]):
        print(f"\nTopology: {device_topology(cp, devices)}")
        cp.random.seed(PROTOCOL["seed"])
        a = cp.random.rand(batch, dim, dtype=cp.float32)
        b = cp.random.rand(batch, dim, dtype=cp.float32)

        launch = _distributed._launch
        serialized = False

        def launch_chunk(*args):
            launch(*args)
            if serialized:
                cp.cuda.runtime.deviceSynchronize()

        monkeypatch.setattr(_distributed, "_launch", launch_chunk)

        def split(with_barrier):
            nonlocal serialized
            serialized = with_barrier
            return _distributed.split_batch(a, b, devices=devices)

        def synchronize():
            for dev in devices:
                with cp.cuda.Device(dev):
                    cp.cuda.runtime.deviceSynchronize()

        print(f"Workload: batch={batch}, dim={dim}, dtype=float32")
        cases = measure_workloads(
            {"queued": lambda: split(False), "serialized": lambda: split(True)},
            synchronize,
        )
    ratio = cases["serialized"]["median_seconds"] / cases["queued"]["median_seconds"]
    print(f"Serialized/queued median ratio: {ratio:.2f}x (informational)")
