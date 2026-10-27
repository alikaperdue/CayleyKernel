# CayleyKernel

**O(1)-order GPU Cayley-Dickson doubling products.**

CayleyKernel computes the Cayley-Dickson doubling product of arbitrary
`2^n`-dimensional hypercomplex algebras (complex numbers, quaternions,
octonions, sedenions, and beyond) on NVIDIA GPUs with **constant-order
per-element speed**: every output component is produced by its own thread
using a branch-free, closed-form basis-sign function, and arbitrarily many
independent products are batched in a single kernel launch via a
grid-stride loop.

* Kernel source: [`cuda/kernel.cu`](cuda/cuda_kernel.cu) (NVIDIA CUDA) and
  [`cuda/amd_kernel.cu`](cuda/amd_kernel.cu) (AMD HIP/ROCm port).
* Python package: [`CayleyKernel`](CayleyKernel/__init__.py) — a thin,
  dependency-light CuPy/ctypes wrapper.
* First public release of a high-dimensional Cayley-Dickson product
  running at O(1) thread-order complexity, in parallel, with NVIDIA CUDA
  hardware acceleration based on the paper by [`Maher Ben Abdessalem`](https://figshare.com/articles/preprint/A_Proven_Sign_Law_for_Cayley-Dickson_Algebras_Ordinary_and_Split_Constructions/33705022)

---

## Computation

Given two algebra elements `x, y ∈ A` of dimension `2^n`, the Cayley-Dickson
doubling product is defined recursively by

```
(a, b) * (c, d) = ( a*c − conj(d)*b ,  d*a + b*conj(c) )
```

with `conj(x) = (x0, −x1, …, −x_{n−1})`.  This construction yields the
complex numbers from the reals, the quaternions from the complex numbers,
the octonions, sedenions, and so on.

This repository implements exactly that product — including the full
non-commutative, non-associative sign structure at every level — but
evaluates it on the GPU in one parallel pass:

* each of the `2^n` output components is computed by an independent thread;
* the basis sign `e_row · e_col = ± e_{row⊕col}` is obtained from a
  branch-free closed form using `__ffs()` and `__popc()`;
* a `(B, …, dim)` input is treated as `prod(B, …)` independent products
  in a single launch (grid-stride over the batch axis).

## Installation

### build from source

Requires an NVIDIA GPU, the CUDA toolkit (`nvcc` on PATH), and Python ≥ 3.8
with `numpy`, `cupy`, and `build`.

```bash
git clone git@github.com:alikaperdue/CayleyKernel.git
cd CayleyKernel

make kernel                 # compile cuda/cuda_kernel.cu -> CayleyKernel/cuda_kernel.so
make test                   # run the pytest suite

pip install -e .            # optional: editable install of the wrapper
```
Build a redistributable wheel (compiles the kernel, bundles it, and runs
`auditwheel repair` when available):

```bash
pip install build auditwheel
make wheel
```

The kernel is built directly into `CayleyKernel/cuda_kernel.so` and bundled in
the wheel in `dist/`. The wheel is tagged for the build platform (for example,
`py3-none-linux_x86_64`), because its native library uses `ctypes` rather than
the Python extension ABI. To select GPU architectures explicitly, run
`ARCHS="sm_80 sm_89" make wheel`; otherwise the Makefile's defaults apply.
Use `make wheel PYTHON=/path/to/python` to select the build interpreter.

When `auditwheel` is available, repaired wheels are written to `wheelhouse/`.
Repair can fail if required shared libraries or system tools are unavailable;
a platform tag alone does not guarantee manylinux portability or CUDA driver
and GPU compatibility. Validate release wheels on the environments you support.
Publish wheels as release assets or on a package index; keep generated binaries
out of Git.

To run the wheel-build regression tests, install the test's build tools in your
virtual environment and run:

```bash
python -m pip install build 'setuptools>=70.1.0' wheel
python tests/test_wheel_build.py -v
```

The metadata tests run offline without build isolation and skip with an
explanation when build tools are missing. Normal `make wheel` builds use
isolation and install their build dependencies automatically.

## Quick start

```python
from CayleyKernel import multiply
import cupy as cp

# A single quaternion-like product (dim = 4 = 2^2)
a = cp.array([1, 2, 3, 4], dtype=cp.float32)
b = cp.array([1, 2, 3, 4], dtype=cp.float32)
print(multiply(a, b))

# 1024 independent 16-dimensional (sedenion) products in one launch
A = cp.random.rand(1024, 16, dtype=cp.float32)
B = cp.random.rand(1024, 16, dtype=cp.float32)
C = multiply(A, B)                       # shape (1024, 16)

# Broadcasting: one vector against a whole stack
C = multiply(A, cp.random.rand(16, dtype=cp.float32))
```

Supported dtypes: `float16`, `float32`, `float64`, `int8/16/32/64`, `uint8/16/32/64`,
`bool`, `complex64`, and `complex128`. Inputs may be CuPy arrays, NumPy arrays,
or plain Python lists; the output container type matches the inputs.

`float16` (FP16 / `__half`) uses 2-byte input/output coefficients with **float32
multiplication and accumulation**, then rounds to float16 on output. This avoids
intermediate FP16 overflow and repeated FP16 rounding, but the final result can
still overflow the float16 range (maximum finite magnitude 65,504). Mixed dtypes
follow CuPy's `result_type` promotion, for example float16 + float32 inputs produce
float32 output. Plain real Python lists still default to float32; pass typed
NumPy or CuPy arrays to request float16 explicitly.

```python
a = cp.array([1, 2, 3, 4], dtype=cp.float16)
b = cp.array([4, 3, 2, 1], dtype=cp.float16)
c = multiply(a, b)  # float16 output, float32 arithmetic inside the kernel
```

Run `python -m pytest -q tests/test_half.py` after rebuilding to check FP16
products against a recursive reference, batching, dtype promotion, both C entry
points, subnormals, output rounding, and cancellation of terms that exceed the
FP16 range. GPU tests require CuPy and a working GPU runtime; wrapper and ABI
checks can run on the CPU with a loadable compiled library.

Complex dtypes supply **complex coefficients** for the existing Cayley–Dickson
basis (for example, complexified quaternions at `dim=4`). They do not pack a
real vector into half as many complex elements: `dim` still counts coefficients.
Multiplication is complex-bilinear, using the same basis signs as real inputs.
`conjugate(x)` negates all coefficients except the first along the last axis;
it does not complex-conjugate the coefficients. Real norm/positivity identities
should not be assumed for complex coefficients.

```python
a = cp.array([1 + 2j, 3 - 1j], dtype=cp.complex64)
b = cp.array([2 - 1j, 1 + 4j], dtype=cp.complex64)
c = multiply(a, b)  # [a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]]
```

Both CUDA and HIP builds export `mult_half`, `mult_complex64`, `mult_complex128`, and their
batched `_nd` entry points. Rebuild the shared library after updating the source
(`make kernel`, or `make kernel-amd` for HIP), then restart Python.

### API

| Function | Description |
| --- | --- |
| `multiply(a, b)` | Cayley-Dickson product; last axis = algebra dimension `2^n`, leading axes batched + broadcast |
| `conjugate(x)` | Standard conjugation `(x0, −x1, …, −x_{n−1})` |
| `library_path()` | Path of the loaded `cuda_kernel.so` |
| `loaded()` | `True` once the kernel library has been found |
| `abi_version()` | Kernel ABI version (current builds report `2`) |
| `twist(r, c)` | Exact CPU basis sign; nonnegative, zero-based integer indices |
| `basis(r, c)` | Exact CPU basis product; nonzero, signed one-based integer labels |
| `twist_many(pairs)`, `basis_many(pairs)` | Ordered CPU batch operations for iterables of pairs |
| `cpu_backend()` | Active CPU implementation: `python` or `rust` |

### Optional Rust CPU acceleration

The sparse CPU operations used by the research scripts can use a Rust extension
while retaining arbitrary-size integer indices. Dense CUDA/HIP products keep
their existing implementation. With a stable Rust toolchain installed:

```bash
python -m pip install ./rust
python -c "from CayleyKernel import cpu_backend; print(cpu_backend())"
```

Rust is selected automatically when installed; Python remains the fallback.
Set `CAYLEY_KERNEL_CPU_BACKEND=python` or `rust` before starting Python to select
explicitly. The separate extension is optional and does not change the main
package's build requirements. See [Rust setup, API, and verification](rust/README.md)
for CPU-only installation, input conventions, and reproducible benchmarks.

## Repository layout

```
CayleyKernel/
├── CayleyKernel/           Python package (ctypes wrapper, CuPy glue)
├── cuda/
│   ├── kernel.cu            NVIDIA CUDA kernel (primary target)
│   ├── amd_kernel.cu        AMD HIP/ROCm port of the same kernel
│   └── tools/
│       └── compare_sign_tables.cpp   host-side sign-function equivalence test
├── tests/                   pytest suite (reference impl + identities)
├── scripts/build_wheel.sh   kernel -> package -> wheel -> auditwheel
├── Makefile                 kernel / test / wheel / clean targets
├── pyproject.toml           PEP 621 metadata (name: cuda_kernel)
└── LICENSE                  Apache-2.0
```

The compiled `.so` is **not** committed: build it with `make kernel`, or
download a wheel from Releases.

## Testing

```bash
make test          # default suite: correctness + algebraic identities
make test-all      # adds: pytest -m slow and pytest -m benchmark
```

* Default suite: correctness against the recursive reference and algebraic
  identities (norm multiplicativity, conjugation laws, `x·x* = |x|²`,
  zero-divisor checks, basis sign table) up to `2^16`, plus spot checks at
  `2^20` — target well under ~2 minutes.
* `slow`: identities and batched tests at `2^20`–`2^22`, opt-in via
  `pytest -m slow`.
* `benchmark`: timing gates and batch-scaling checks, opt-in via
  `pytest -m benchmark`.

### Huge-index zero-divisor showcase

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

### Optional sparse zero-divisor validation and timings (CPU only)

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
These checks validate the Python basis-product path, not GPU dense multiplication;
finite testing supplies evidence of correctness, not an all-level proof.

The same command also prints a speed comparison at each level for three
workloads: sign calculation, signed basis multiplication, and complete
two-term sparse multiplication (four basis products and coefficient
cancellation). The new path uses `twist` / `basis`; the baseline is
the independent Python doubling reference in `tests/test_sparse_zd.py`.
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

## Naming

| Context | Name |
| --- | --- |
| GitHub repository / branding | `CayleyKernel` |
| Python package / import | `CayleyKernel` |
| Wheel / PyPI project name | `cuda_kernel` |
| Compiled kernel binary | `cuda_kernel.so` |
| C ABI symbols | `mult_<type>`, `mult_<type>_nd` (stable) |

This project is the **O(1) (constant-order)** generation of this work: the
doubling product's per-element complexity is constant, independent of the
algebra dimension, thanks to the closed-form sign function and fully
parallel evaluation.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

© 2026 Yotta Space Corp



## Paper verification: arXiv:2608.28176v1

A section-by-section review of Zhilina's *Orthogonality graphs of real
Cayley–Dickson algebras. Part I* is in
[docs/papers/2608.28176](docs/papers/2608.28176/README.md), including the
coded test inventory, finite coverage limits, saved results, and two
reproducible discrepancies in the version 1 statements.

```bash
python3 -m tests.papers.arxiv_2608_28176.run
```

The CPU checks use `twist()` and `basis()` with exact arithmetic
and an independent doubling oracle. Optional `multiply()` checks run when
the CUDA prerequisites are available and otherwise report explicit skips.
