# Testing CayleyKernel

[Back to the README](../README.md)

Commands below run from the repository root.

## Test selection

| Workload | Make | Pytest |
| --- | --- | --- |
| Default | `make test` | `python3 -m pytest -q tests/` |
| Default host-only | `make test-cpu` | `python3 -m pytest -q -m "cpu and not slow" tests/` |
| Slow only | | `python3 -m pytest -q -m slow tests/` |
| Default and slow | `make test-all` | `python3 -m pytest -q -m "" tests/` |

An explicit `-m` expression replaces the default `not slow` filter, so
`-m cpu` includes slow host checks. GPU timing workloads skip unless
`--run-benchmarks` is supplied, even with `make test-all` or `-m benchmark`.
The CPU sparse zero-divisor workflow uses its own `--run-sparse-zd` flag.

The default suite includes the dimension-16 basis table and dimension-1024
sparse-input dense product. The larger sparse-input products at dimensions
`2^14`, `2^16`, and `2^20`, the `2^20` unity check, and other expensive
products are slow. The quasilinear paper's million-component CPU check is
also slow.

GPU tests check CuPy and device availability before seeding the GPU RNG.
Most GPU tests also need the compiled CayleyKernel library. Tests marked
`gpu_device` can run with CuPy and a device alone, including CSR conversion
and checks that can use the sparse product's host fallback.

Use `python3 -m pytest -q --require-gpu tests/` on a dedicated GPU runner.
Missing GPU prerequisites produce setup failures for selected GPU tests.
Local runs skip those tests. Host tests, tests deselected by markers, and
disabled opt-in suites bypass GPU setup. Optional dependencies and
capability-specific skips such as CuPy 14+ for BF16 or a second device
still apply.

## GPU performance workloads

| Workload | Make command | Result |
| --- | --- | --- |
| Dense multiply, fused add/subtract, and unfused multiply then add | `make benchmark` | Compare per-case medians with a local baseline |
| Same workloads, recording mode | `make benchmark-record` | Write a new local baseline |
| Sparse versus dense multiplication | `make benchmark-sparse` | Informational dense/sparse ratio for six float32 shapes |
| Multi-GPU overlap | `make benchmark-multi-gpu` | Informational serialized/queued ratio and device topology |
| PyTorch adapter overhead | `make benchmark-torch` | Check that the Torch median is at most 3.0 times the CuPy median |

Run the three paired workloads together with
`python3 -m pytest -q -s -m "benchmark and not benchmark_baseline" tests/ --run-benchmarks`.
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

Direct pytest invocations select `tests/test_benchmark.py` with
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
python3 -m pip install pytest
make test-sparse-zd
# Customize doubling levels n (dimension = 2^n), candidates per level, and seed:
make test-sparse-zd ZD_LEVELS=4,32,256,4096 ZD_SAMPLES=1000 ZD_SEED=42 ZD_REPEATS=5
# Equivalent direct invocation:
python3 -m pytest -q -s tests/test_sparse_zd.py --run-sparse-zd \
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
