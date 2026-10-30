"""GPU baseline measurement and comparison, independent of pytest/CuPy imports."""

import hashlib
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "warmup": 3, "trials": 5, "statistic": "median",
    "timer": "synchronized_batched_wall_seconds", "seed": 2026,
    "target_sample_seconds": 0.1, "max_repetitions": 16384,
}
ENVIRONMENT_KEYS = (
    "gpu_name", "pci_bus_id", "compute_capability", "cuda_runtime_version",
    "cuda_driver_version", "cupy_version", "python_version", "platform",
)
CASE_SHAPES = {
    "single_2pow20": [2**20],
    "batch_1024_dim16": [1024, 16],
    "batch_16_dim1024": [16, 1024],
    "batch_1024_dim1024": [1024, 1024],
}


def validate_baseline(report, environment):
    if not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("protocol") != PROTOCOL:
        raise ValueError("Baseline schema/timing protocol differs; record a new baseline.")
    mismatch = [key for key in ENVIRONMENT_KEYS
                if report.get("environment", {}).get(key) != environment.get(key)]
    if mismatch:
        raise ValueError("Baseline environment differs: " + ", ".join(mismatch))
    if set(report.get("cases", {})) != set(CASE_SHAPES):
        raise ValueError("Baseline must contain all four workloads.")
    for name, shape in CASE_SHAPES.items():
        case = report["cases"][name]
        samples = case.get("samples_seconds", [])
        repetitions = case.get("repetitions")
        if (case.get("shape") != shape or case.get("dtype") != "float32"
                or type(repetitions) is not int
                or not 1 <= repetitions <= PROTOCOL["max_repetitions"]
                or len(samples) != PROTOCOL["trials"]
                or any(not isinstance(s, (float, int)) or not math.isfinite(s) or s <= 0
                       for s in samples)):
            raise ValueError("Invalid baseline workload/samples: " + name)
        if case.get("median_seconds") != statistics.median(samples):
            raise ValueError("Baseline median does not match its samples: " + name)


def regression_limit(baseline_case, max_regression):
    return baseline_case["median_seconds"] * (1 + max_regression)


def environment_info(cp):
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props["name"]
    return {
        "gpu_name": name.decode() if isinstance(name, bytes) else str(name),
        "pci_bus_id": device.pci_bus_id,
        "compute_capability": device.compute_capability,
        "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver_version": cp.cuda.runtime.driverGetVersion(),
        "cupy_version": cp.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def provenance():
    import CayleyKernel as ck
    native_path = Path(ck._lib._name).resolve()
    paths = [ROOT / "cuda/cuda_kernel.cu", ROOT / "CayleyKernel/__init__.py", native_path]
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    def label(path):
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return path.name
    return {
        "git_head": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "kernel_abi": ck.abi_version(),
        "sha256": {label(p):
                   hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }


class BenchmarkSession:
    def __init__(self, cp, baseline=None, record=None, max_regression=0.20):
        if (baseline is None) == (record is None):
            raise ValueError("Choose exactly one of baseline comparison or recording.")
        if not math.isfinite(max_regression) or max_regression < 0:
            raise ValueError("Regression allowance must be finite and nonnegative.")
        self.cp = cp
        self.record = Path(record) if record else None
        if self.record and self.record.exists():
            raise ValueError("Refusing to overwrite an existing baseline: " + str(self.record))
        self.max_regression = max_regression
        self.report = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL.copy(),
            "environment": environment_info(cp),
            "provenance": provenance(),
            "cases": {},
        }
        self.baseline = json.loads(Path(baseline).read_text()) if baseline else None
        if self.baseline is not None:
            validate_baseline(self.baseline, self.report["environment"])

    @staticmethod
    def _time_calls(multiply, x, y, stream, repetitions):
        stream.synchronize()
        start = time.perf_counter()
        for _ in range(repetitions):
            result = multiply(x, y)
            del result
        stream.synchronize()
        return time.perf_counter() - start

    def measure(self, name, multiply, shape):
        cp = self.cp
        cp.random.seed(PROTOCOL["seed"])
        x = cp.random.rand(*shape, dtype=cp.float32)
        y = cp.random.rand(*shape, dtype=cp.float32)
        stream = cp.cuda.Stream.null
        for _ in range(PROTOCOL["warmup"]):
            result = multiply(x, y)
            stream.synchronize()
            del result
        # Calibration also warms this workload under sustained GPU activity.
        # Keep long operations at one call; amortize synchronization overhead
        # across repeated calls for short operations. Recording and comparison
        # both calibrate locally and use the same per-call timing statistic.
        repetitions = 1
        while True:
            elapsed = self._time_calls(multiply, x, y, stream, repetitions)
            if (elapsed >= PROTOCOL["target_sample_seconds"]
                    or repetitions >= PROTOCOL["max_repetitions"]):
                break
            repetitions *= 2
        samples = []
        for _ in range(PROTOCOL["trials"]):
            elapsed = self._time_calls(multiply, x, y, stream, repetitions)
            samples.append(elapsed / repetitions)
        case = {"shape": list(shape), "dtype": "float32", "samples_seconds": samples,
                "repetitions": repetitions,
                "median_seconds": statistics.median(samples)}
        self.report["cases"][name] = case
        print(f"\n{name}: {case['median_seconds']:.6f}s median per call; "
              f"{repetitions} calls/sample; samples={samples}")
        if self.baseline is not None:
            reference = self.baseline["cases"][name]
            limit = regression_limit(reference, self.max_regression)
            assert case["median_seconds"] <= limit, (
                f"{name}: median {case['median_seconds']:.6f}s exceeds baseline "
                f"{reference['median_seconds']:.6f}s + {self.max_regression:.0%} "
                f"({limit:.6f}s). Check GPU load/clocks and repeat before diagnosing a regression."
            )
        return case["median_seconds"]

    def finish(self):
        if self.record:
            validate_baseline(self.report, self.report["environment"])
            self.record.parent.mkdir(parents=True, exist_ok=True)
            with self.record.open("x") as stream:
                json.dump(self.report, stream, indent=2)
                stream.write("\n")
            print("\nRecorded baseline:", self.record)
