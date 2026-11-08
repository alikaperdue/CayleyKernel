# Testing CayleyKernel

[Back to the README](../README.md)

Run commands from the repository root. Use `python -m pytest` so pytest,
CayleyKernel, and the optional dependencies use the same Python interpreter.

## Where tests belong

| Directory | Responsibility |
| --- | --- |
| `tests/core/` | Products, dtypes, algebra, validation, and Python and Rust CPU backends |
| `tests/integration/` | CUDA launch ABI, JAX and PyTorch adapters, multi-GPU correctness, wheel builds, and pytest configuration |
| `tests/performance/` | Opt-in GPU measurements and CPU tests of the timing and baseline tooling |
| `tests/research/` | Relation graphs, clique growth, associativity, and structural scaling |
| `tests/papers/` | Paper-specific checks and report runners, grouped by identifier |
| `tests/support/` | Shared reference products, comparison policies, test inputs, and GPU prerequisite probes |

Shared pytest fixtures and options live in `tests/conftest.py`. Keep
checked-in witness inputs with the suite that consumes them. The
[graph fixture inventory](../tests/research/fixtures/README.md) records
the provenance of the shared clique certificate. The
[paper index](papers/README.md) links section-level checks, standalone
reproduction commands, and the canonical published evidence.

## Dependencies

| Scope | Install from the checkout | Additional prerequisites |
| --- | --- | --- |
| Base CPU tests | `python -m pip install -e '.[test]'` | NumPy and pytest are sufficient for the core host checks and most graph regressions. |
| CPU and research as in CI | `python -m pip install -e '.[test,research-test]'` | SciPy for frontier solvers, SymPy for symbolic paper checks, Matplotlib for plot/replot, and tzdata for time-zone handling on hosts without IANA data. |
| CUDA correctness and timing | `python -m pip install -e '.[test,cuda12]'` | Use `cuda11` for CUDA 11.8. Use a matching NVIDIA driver and build the kernel with `make kernel`. |
| BF16 | `python -m pip install -e '.[test,bfloat16]'` | This extra already includes the CUDA 12 CuPy build, NumPy ≥ 2.1.2 and `ml_dtypes`. GPU cases also need BF16-capable hardware and a kernel with BF16 exports. |
| PyTorch | `python -m pip install -e '.[test,cuda12,torch]'` | Use a CUDA-enabled PyTorch build compatible with the runner. |
| JAX | `python -m pip install -e '.[test,cuda12,jax]'` | The host callback supports CPU JAX arrays and uses CUDA for the CayleyKernel product. For JAX GPU execution on CUDA 12, also install `jax[cuda12]` as the GPU workflow does. |

Use a single CuPy distribution in an environment. If `nvcc` cannot build
all the Makefile's default architectures, set `ARCHS` to your GPU's
architecture. For native CPU checks, install the optional Rust extension
with `make rust-install` and set `CAYLEY_KERNEL_CPU_BACKEND=rust`.
`CAYLEY_TEST_REQUIRE_RUST=1` makes the backend regression fail if the
extension is missing. For CPU-only BF16 promotion and algebra checks,
install `ml_dtypes>=0.5` alongside `.[test]`. The wheel metadata regression
needs `build`, `setuptools>=77.0.3`, and `wheel`. The build-script regressions
use stubs and require Linux, Bash, and Make.
[JAX CUDA installation](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu)

## Test selection

| Workload | Make | Pytest |
| --- | --- | --- |
| Default | `make test` | `python3 -m pytest -q tests/` |
| Default host-only | `make test-cpu` | `python3 -m pytest -q -m "cpu and not slow" tests/` |
| Slow only | | `python3 -m pytest -q -m slow tests/` |
| Default and slow | `make test-all` | `python3 -m pytest -q -m "" tests/` |

| Marker | Meaning |
| --- | --- |
| `cpu` | Runs without a GPU. Some tests need optional host dependencies or inspect a compiled library. |
| `gpu_device` | Requires a usable CuPy device. The compiled CayleyKernel library is optional. |
| `slow` | Expensive products or reproductions selected explicitly. |
| `sparse_zd` | Exact CPU sparse zero-divisor validation and informational timings. |
| `benchmark` | Opt-in GPU timings. |
| `benchmark_baseline` | GPU timings that record or compare a local baseline. |

An explicit `-m` expression replaces the default `not slow` filter, so
`-m cpu` includes slow host checks. GPU timing workloads skip unless
`--run-benchmarks` is supplied, even with `make test-all` or `-m benchmark`.
The CPU sparse zero-divisor workflow uses its own `--run-sparse-zd` flag.

The default suite includes the dimension-16 basis table and dimension-1024
sparse-input dense product. The larger sparse-input products at dimensions
`2^14`, `2^16`, and `2^20`, the `2^20` unity check, and other expensive
products are slow. The quasilinear paper's million-component CPU check is
also slow, as are selected exhaustive paper audits and the Physics Letters B
level-20 dense GPU replay. Run just the slow host reproductions with
`python -m pytest -q -m "cpu and slow" tests/papers/`. Standalone
`unittest` and paper report runners keep their own selection and include
the expensive cases unless their own options limit the workload.

GPU tests check CuPy and device availability before seeding the GPU RNG.
Most GPU tests also need the compiled CayleyKernel library. Tests marked
`gpu_device` can run with CuPy and a device alone, including CSR conversion
and checks that can use the sparse product's host fallback.

Use `python3 -m pytest -q --require-gpu tests/` on a dedicated GPU runner.
Missing CuPy, a usable device, or a required kernel produces setup failures
for selected GPU tests.
Local runs skip those tests. Host tests, tests deselected by markers, and
disabled opt-in suites bypass GPU setup. Optional imports, missing BF16
capabilities, and a missing second device can still cause skips. The Torch
suite also requires CUDA availability in PyTorch. Check `-rs` output to
see which prerequisites were missing.

## CI coverage

- The CPU workflow runs `cpu and not slow` on Linux, macOS, and Windows
  with Python 3.10, 3.13, and 3.14, first with the Python backend and then
  with the locally built Rust extension. Both passes include fast graph,
  structural-script, and paper regressions. A separate Rust-backed step runs
  sparse zero-divisor validation at levels 4, 32, and 256 with 16 samples
  and one timing round.
- JAX, PyTorch, and multi-GPU workflows run their integration correctness
  files on `main` pushes, relevant pull requests, and manual dispatch.
  They use `--require-gpu` when `CI_GPU_LABEL` or `CI_GPU_2X_LABEL` is
  configured for the job. Otherwise the hosted fallback can skip device tests.
- The wheel workflow checks Python, Rust, and CUDA packaging on relevant
  pull requests, manual dispatch, and releases. The CPU workflow also
  runs Python and CUDA/C++ lint plus the Rust format, lint, and test checks.

## GPU performance workloads

| Workload | Make command | Result |
| --- | --- | --- |
| Dense multiply, fused add/subtract, and unfused multiply then add | `make benchmark` | Compare per-case medians with a local baseline |
| Same workloads, recording mode | `make benchmark-record` | Write a new local baseline |
| Sparse versus dense multiplication | `make benchmark-sparse` | Informational dense/sparse ratio for six float32 shapes |
| Multi-GPU overlap | `make benchmark-multi-gpu` | Informational serialized/queued ratio and device topology |
| PyTorch adapter overhead | `make benchmark-torch` | Check that the Torch median is at most 3.0 times the CuPy median |

Run the three paired workloads together with
`python3 -m pytest -q -s -m "benchmark and not benchmark_baseline" tests/performance/ --run-benchmarks`.
Add `-k <pattern>` to a direct pytest invocation to select individual cases.
The paired workloads need no baseline file. They include each API's wrapper
and output handling. Sparse inputs are converted before sampling, so the
sparse timing includes the work inside `sparse_multiply` and returns sparse
output. The dense call returns dense output. The sparse workload needs the
float32 sparse GPU kernel. The overlap workload needs two visible devices
and measures both paths with input transfers and output gather included.
It prints the GPU names, PCI bus IDs, and peer-access capability in both
directions. The Torch check needs CUDA-enabled PyTorch and applies a local
adapter overhead ceiling for its fixed float32 workload. The ordinary Torch
suite checks input and output pointer identity directly.

The GPU workloads warm up each path three times, then double the calls per
sample until reaching 100 ms or 16,384 calls. They report five synchronized
wall-clock samples as seconds per call and take their median. Paired workloads
alternate order between rounds. Input generation is outside the timed region.

### Recording and comparing a baseline

The sign evaluator is O(1) for fixed-width GPU indices. The current dense
kernel sums `dim` terms for each of `dim` outputs: `O(batch * dim^2)` total work. A single
`2^20` product evaluates `2^40` terms. Parallel execution does not justify a
universal millisecond expectation or a fixed five-second performance limit.

Record a local reference while the selected GPU is otherwise idle:

```bash
make benchmark-record PYTHON=python3
# Later, compare without modifying that reference:
make benchmark PYTHON=python3
```

Both use `benchmark-results/gpu-local.json`, ignored by Git. Override the path with
`BENCHMARK_BASELINE=benchmark-results/my-gpu.json`. Recording refuses to
overwrite an existing file; choose a new path for a deliberate new reference.
Select a GPU explicitly with `CUDA_VISIBLE_DEVICES=0` (or `1`) before either
command, and use the same device and software environment for comparison.

Direct pytest invocations select `tests/performance/test_benchmark.py` with
`--run-benchmarks` and exactly one of `--benchmark-record PATH` and
`--benchmark-baseline PATH`. Recording requires all configured dense
multiply cases. Reports include the samples, repetitions, device and PCI
identity, driver and runtime, Python and CuPy versions, source and loaded
binary hashes, and Git revision and dirty state. Comparison checks the
device, software, protocol, and recorded cases for compatibility. Source
and binary hashes provide provenance when evaluating a change.
Fused and unfused cases absent from an older baseline report informational
timings.

The default comparison permits a **20% median slowdown** from the recorded
reference (`BENCHMARK_MAX_REGRESSION=0.20`, or
`--benchmark-max-regression 0.20` in pytest). Investigate competing GPU load
and temperature or clock changes before diagnosing a regression. The batch
scaling workload also reports per-item costs for its two batch sizes.

`make clean` preserves local performance baselines so rebuilding does not
erase your reference. Remove `benchmark-results/` explicitly when no longer
needed. Measurements are separate from numerical correctness tests.

## Huge-index zero-divisor showcase

```bash
make zd-showcase
# Push to 100-million-bit indices, with three witnesses:
make zd-showcase ZD_HUGE_LEVELS=100000000 ZD_WITNESSES=3 ZD_SEED=42
# Save complete witnesses as hexadecimal integers for replay:
python3 scripts/zd_showcase.py --levels=1000000 --witnesses=10 \
    --seed=42 --output=huge-zero-divisors.json
```

The default showcase finds ten random zero-divisor pairs at each of levels
4,096, 65,536, and **1,000,000**. At the last level, all four support indices
are million-bit integers, in an algebra of dimension `2^1000000`.
It runs on the CPU using only `twist` and `basis`: no old doubling
reference, GPU, pytest, or dense arrays. This target is separate from all
test targets. Larger indices require more memory and computation time.

For each candidate, three indices are drawn uniformly from the n-bit integers;
the fourth is fixed by `a XOR b XOR c`. Four distinct indices are required.
The formula selects compatible signs for `(e_a + s e_b)(e_c + t e_d)`.
Each accepted pair is then re-evaluated through four signed `basis`
calls: the two pairs of terms must cancel exactly and the accumulated result
must be zero. Flipping `t` must give a nonzero result. Both factors have
squared Euclidean norm 2, so neither is zero.

The display reports confirmations, attempted candidates, search time, and
confirmation time (including the negative control). It previews the first
witness at each level without printing millions of digits. The seed reproduces
the search, and `--output` saves all full indices in hexadecimal, avoiding
Python's large decimal conversion limit. Saved and displayed indices are
zero-based. These are exact checks using the new formulas, rather than an
independent verification of those formulas or a search for previously unknown
types of zero divisor.

## Optional sparse zero-divisor validation and timings (CPU only)

```bash
python3 -m pip install -e '.[test]'
make test-sparse-zd
# Customize doubling levels n (dimension = 2^n), candidates per level, and seed:
make test-sparse-zd ZD_LEVELS=4,32,256,4096 ZD_SAMPLES=1000 ZD_SEED=42 ZD_REPEATS=5
# Equivalent direct invocation:
python3 -m pytest -q -s tests/core/test_sparse_zd.py --run-sparse-zd \
    --zd-levels=4,32,256,4096 --zd-samples=1000 --zd-seed=42 --zd-repeats=5
```

These tests are skipped unless `--run-sparse-zd` is supplied, including during
`make test` and `make test-all`. They need no GPU or compiled kernel. The default
levels are 4, 8, 32, 128, 1024, and 4096, with 128 candidates per level. Only
sparse terms and integer indices are stored; no dimension-sized arrays are
allocated. Increasing the level or sample count increases reference-check time.

The suite checks:

* All signed basis products on the first 64 basis units against an independent
  reference derived from the recursive doubling definition.
* All two-term support pairs in the sedenions, with all four relative sign
  choices, using exact integer cancellation.
* The known `(e_3 + e_10)(e_6 - e_15) = 0` witness embedded at each requested level.
* Structured and seeded random candidates that actually use the highest bit
  at each level, plus controls guaranteed to have nonzero products.

For `(e_a + s e_b)(e_c + t e_d)`, with distinct indices within each factor,
cancellation requires `a XOR b = c XOR d`. Candidates therefore choose `a,b,c`
and derive `d`; all signs are checked against the independent reference, which
uses doubling branches rather than the implementation's popcount formula.
The high-level reference is iterative to avoid Python's recursion limit.

Printed counts distinguish XOR-compatible products from guaranteed nonzero
controls and include the level and seed. Counts describe this sampling scheme,
not the density of all zero divisors. Failures include reproducible support
indices and signs. Indices in test reports use mathematical **zero-based**
notation; the adapter calls `basis` with its signed **one-based** labels.
These checks validate the selected CPU basis-product backend, not GPU dense multiplication;
finite testing supplies evidence of correctness, not an all-level proof.

The same command also prints a speed comparison at each level for three
workloads: sign calculation, signed basis multiplication, and complete
two-term sparse multiplication (four basis products and coefficient
cancellation). The new path uses `twist` / `basis`; the baseline is
the independent Python doubling reference in `tests/support/oracles.py`.
This baseline follows the older level-by-level approach; it is not a
reproduction of a particular author's optimized implementation.

Both paths receive identical pre-generated inputs, and their results are
checked before timing. Each path is warmed up, batch counts are calibrated
to at least 20 ms where practical (capped at 65,536 batches), and execution
order alternates across rounds. The table reports median microseconds per
operation and speedup (`doubling time / new time`). `ZD_REPEATS` defaults to 3.
Input generation, correctness assertions, and printing are excluded from
timings; sparse timings include sign calculation and output accumulation
without cached signs. Machine architecture, Python version, seed, and workload
sizes accompany the results. Timings are informational, with no required
speedup threshold. These CPU measurements do not measure dense GPU products
or establish asymptotic complexity; Python integer costs grow with bit length.
