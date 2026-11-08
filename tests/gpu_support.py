# SPDX-License-Identifier: Apache-2.0
"""GPU prerequisite checks shared by pytest and standalone paper runners."""

import CayleyKernel as ck


class GPUUnavailable(RuntimeError):
    """The GPU test cannot start with the installed runtime or library."""


def device_count():
    if ck.cp is None:
        raise GPUUnavailable("CuPy is unavailable")
    runtime = ck.cp.cuda.runtime
    try:
        count = runtime.getDeviceCount()
    except runtime.CUDARuntimeError as exc:
        raise GPUUnavailable(f"CUDA device unavailable: {exc}") from exc
    if count < 1:
        raise GPUUnavailable("No CUDA device is visible")
    return count


def check_compiled_kernel():
    if not ck.loaded():
        raise GPUUnavailable("Compiled kernel unavailable; build it with make kernel")


def require_gpu(case):
    """Keep the paper suites runnable with unittest and no pytest dependency."""
    try:
        device_count()
        check_compiled_kernel()
    except GPUUnavailable as exc:
        case.skipTest(str(exc))
