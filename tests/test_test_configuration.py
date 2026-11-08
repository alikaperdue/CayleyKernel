# SPDX-License-Identifier: Apache-2.0
"""Exercise the real pytest hooks with a simulated device and library."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run_pytest(tmp_path):
    suite = tmp_path / "test_prerequisites.py"
    suite.write_text(
        """
import os
from types import SimpleNamespace

import pytest

import CayleyKernel as ck

state = os.environ["CAYLEY_TEST_GPU_STATE"]
calls = []


class CUDARuntimeError(Exception):
    pass


def device_count():
    calls.append("device")
    if state == "driver-error":
        raise CUDARuntimeError("driver unavailable")
    return 0 if state == "no-device" else 1


def loaded():
    calls.append("kernel")
    return state == "ready"


def seed(value):
    assert "device" in calls
    calls.append("seed")
    assert value == 0


ck.cp = None if state == "no-cupy" else SimpleNamespace(
    cuda=SimpleNamespace(runtime=SimpleNamespace(
        getDeviceCount=device_count, CUDARuntimeError=CUDARuntimeError
    )),
    random=SimpleNamespace(seed=seed),
)
ck.loaded = loaded


@pytest.mark.cpu
def test_host():
    assert calls == []


@pytest.mark.gpu_device
def test_device_only():
    assert calls == ["device", "seed"]


def test_kernel_gpu():
    assert calls == ["device", "kernel", "seed"]


@pytest.mark.slow
def test_slow_gpu():
    assert calls == ["device", "kernel", "seed"]
""",
        encoding="utf-8",
    )

    def run(state, *options):
        env = dict(os.environ, CAYLEY_TEST_GPU_STATE=state)
        env.pop("PYTEST_ADDOPTS", None)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-rs",
                "-c",
                str(ROOT / "pyproject.toml"),
                "-p",
                "tests.conftest",
                "--confcutdir",
                str(tmp_path),
                str(suite),
                *options,
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        return result.returncode, result.stdout + result.stderr

    return run


@pytest.mark.parametrize(
    "state, reason",
    [
        ("no-cupy", "CuPy is unavailable"),
        ("no-device", "No CUDA device is visible"),
        ("driver-error", "CUDA device unavailable: driver unavailable"),
        ("no-kernel", "Compiled kernel unavailable"),
    ],
)
def test_unavailable_gpu_skips_locally_and_fails_when_required(run_pytest, state, reason):
    code, output = run_pytest(state, "-k", "kernel_gpu")
    assert code == 0 and "1 skipped" in output and reason in output, output
    code, output = run_pytest(state, "-k", "kernel_gpu", "--require-gpu")
    assert code == 1 and f"--require-gpu: {reason}" in output, output


@pytest.mark.parametrize(
    "state, selection",
    [
        ("driver-error", ["-m", "cpu"]),
        ("no-kernel", ["-m", "gpu_device"]),
        ("ready", ["-k", "kernel_gpu"]),
        ("ready", ["-m", "slow"]),
    ],
)
def test_selected_tests_only_request_their_prerequisites(run_pytest, state, selection):
    code, output = run_pytest(state, "--require-gpu", *selection)
    assert code == 0 and "1 passed" in output, output


def test_default_collection_excludes_slow_tests_and_rejects_unknown_marks(run_pytest, tmp_path):
    code, output = run_pytest("no-cupy", "--collect-only")
    assert code == 0 and "test_kernel_gpu" in output and "test_slow_gpu" not in output, output

    suite = tmp_path / "test_prerequisites.py"
    with suite.open("a", encoding="utf-8") as f:
        f.write("\n@pytest.mark.unknown_marker\ndef test_bad_marker():\n    pass\n")
    code, output = run_pytest("no-cupy", "--collect-only")
    assert code != 0 and "unknown_marker" in output, output
