"""GPU baseline measurement and comparison, independent of pytest/CuPy imports."""

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "warmup": 3,
    "trials": 5,
    "statistic": "median",
    "timer": "synchronized_batched_wall_seconds",
    "seed": 2026,
    "target_sample_seconds": 0.1,
    "max_repetitions": 16384,
}
ENVIRONMENT_KEYS = (
    "gpu_name",
    "pci_bus_id",
    "compute_capability",
    "cuda_runtime_version",
    "cuda_driver_version",
    "cupy_version",
    "python_version",
    "platform",
)
CASE_SHAPES = {
    "single_2pow20": [2**20],
    "batch_1024_dim16": [1024, 16],
    "batch_16_dim1024": [16, 1024],
    "batch_1024_dim1024": [1024, 1024],
    "batch_4096_dim8": [4096, 8],
    "batch_16384_dim16": [16384, 16],
    "batch_65535_dim16": [65535, 16],
    "batch_70000_dim4": [70000, 4],
}

# Fused multiply-add (ABI >= 4).  The "sign" field selects which Python
# entry point (multiply_add for +1, multiply_subtract for -1) the harness
# should launch.  These cases only appear when the loaded kernel exposes
# the fused symbols; older baselines omit them and the comparison step
# skips them gracefully (see validate_baseline).
CASE_SHAPES_FMA = {
    "fma_batch_65536_dim32_add": {"shape": [65536, 32], "sign": +1},
    "fma_batch_65536_dim32_sub": {"shape": [65536, 32], "sign": -1},
    "fma_batch_1024_dim16_add": {"shape": [1024, 16], "sign": +1},
    "fma_batch_1024_dim16_sub": {"shape": [1024, 16], "sign": -1},
}

# Three-kernel "before" baseline: multiply(a, b) followed by elementwise + c.
# Recorded so the PR description's before/after throughput table (three-kernel
# path vs. single-kernel multiply_add) comes from the same harness and the
# same baseline file as the fused cases.  Optional on older baselines.
CASE_SHAPES_FMA_BASELINE = {
    "baseline_multiply_add_batch_65536_dim32": [65536, 32],
}


def _validate_case(name, case, shape, sign=None):
    samples = case.get("samples_seconds", [])
    repetitions = case.get("repetitions")
    if (
        case.get("shape") != list(shape)
        or case.get("dtype") != "float32"
        or type(repetitions) is not int
        or not 1 <= repetitions <= PROTOCOL["max_repetitions"]
        or len(samples) != PROTOCOL["trials"]
        or any(
            not isinstance(s, (float, int)) or not math.isfinite(s) or s <= 0 for s in samples
        )
    ):
        raise ValueError("Invalid baseline workload/samples: " + name)
    if case.get("median_seconds") != statistics.median(samples):
        raise ValueError("Baseline median does not match its samples: " + name)
    if sign is not None and case.get("sign") != sign:
        raise ValueError("Baseline fused sign differs: " + name)


def validate_baseline(report, environment):
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("protocol") != PROTOCOL
    ):
        raise ValueError("Baseline schema/timing protocol differs; record a new baseline.")
    mismatch = [
        key
        for key in ENVIRONMENT_KEYS
        if report.get("environment", {}).get(key) != environment.get(key)
    ]
    if mismatch:
        raise ValueError("Baseline environment differs: " + ", ".join(mismatch))
    cases = report.get("cases", {})
    # Multiply (2-operand) cases are required on every baseline, including
    # older ones that pre-date the fused entry points.
    missing = [name for name in CASE_SHAPES if name not in cases]
    if missing:
        raise ValueError("Baseline is missing all configured multiply cases: " + ", ".join(missing))
    for name, shape in CASE_SHAPES.items():
        _validate_case(name, cases[name], shape)
    # Fused cases are optional: older baselines pre-date ABI >= 4 and will
    # not contain them.  When present, they must be valid.
    for name, spec in CASE_SHAPES_FMA.items():
        if name in cases:
            _validate_case(name, cases[name], spec["shape"], sign=spec["sign"])
    for name, shape in CASE_SHAPES_FMA_BASELINE.items():
        if name in cases:
            _validate_case(name, cases[name], shape)
    # Names that are neither multiply, fused, nor the three-kernel baseline
    # are invalid.
    known = set(CASE_SHAPES) | set(CASE_SHAPES_FMA) | set(CASE_SHAPES_FMA_BASELINE)
    unknown = [name for name in cases if name not in known]
    if unknown:
        raise ValueError("Baseline has unknown cases: " + ", ".join(unknown))


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
        "launch_block_size": os.environ.get("CAYLEY_KERNEL_BLOCK_SIZE", "auto"),
        "sha256": {label(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
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
    def _time_calls(fn, args, stream, repetitions):
        stream.synchronize()
        start = time.perf_counter()
        for _ in range(repetitions):
            result = fn(*args)
            del result
        stream.synchronize()
        return time.perf_counter() - start

    def measure(self, name, multiply, shape):
        # Two-operand multiply: a*b.  Both operands share ``shape``; the
        # inner tuple tells _measure how many operands to allocate.
        return self._measure(name, multiply, (shape, shape), shape)

    def measure_fma(self, name, multiply_fn, shape, sign):
        """Record/compare a fused multiply_add (sign=+1) or
        multiply_subtract (sign=-1) workload.

        All three operands are drawn from the same RNG and share ``shape``;
        the resulting case dict additionally records the ``sign`` so that
        baselines can be validated against the expected entry point.
        """
        if sign not in (+1, -1):
            raise ValueError("sign must be +1 (multiply_add) or -1 (multiply_subtract)")
        import CayleyKernel as ck

        fn = ck.multiply_add if sign == +1 else ck.multiply_subtract
        return self._measure(name, fn, (shape, shape, shape), shape, sign=sign)

    def measure_fma_baseline(self, name, multiply_fn, shape):
        """Record the three-kernel path multiply(a, b) + c at the given shape.

        This is the "before" half of the fused FMA throughput comparison:
        two CUDA launches (multiply, then elementwise add) plus the Python
        glue and the intermediate allocation, as opposed to the single fused
        launch captured by ``measure_fma``.  Stored in the same baseline
        file so the before/after numbers in the PR description come from
        one consistent run.
        """
        def multiply_then_add(a, b, c):
            return multiply_fn(a, b) + c
        return self._measure(name, multiply_then_add, (shape, shape, shape), shape)

    def _measure(self, name, fn, operand_shapes, shape, sign=None):
        cp = self.cp
        cp.random.seed(PROTOCOL["seed"])
        operands = [cp.random.rand(*operand_shapes[0], dtype=cp.float32) for _ in operand_shapes]
        stream = cp.cuda.Stream.null
        args = tuple(operands)
        for _ in range(PROTOCOL["warmup"]):
            result = fn(*args)
            stream.synchronize()
            del result
        repetitions = 1
        while True:
            elapsed = self._time_calls(fn, args, stream, repetitions)
            if (
                elapsed >= PROTOCOL["target_sample_seconds"]
                or repetitions >= PROTOCOL["max_repetitions"]
            ):
                break
            repetitions *= 2
        samples = []
        for _ in range(PROTOCOL["trials"]):
            elapsed = self._time_calls(fn, args, stream, repetitions)
            samples.append(elapsed / repetitions)
        case = {
            "shape": list(shape),
            "dtype": "float32",
            "samples_seconds": samples,
            "repetitions": repetitions,
            "median_seconds": statistics.median(samples),
        }
        if sign is not None:
            case["sign"] = sign
        self.report["cases"][name] = case
        print(
            f"\n{name}: {case['median_seconds']:.6f}s median per call; "
            f"{repetitions} calls/sample; samples={samples}"
        )
        if self.baseline is not None and name in self.baseline.get("cases", {}):
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
