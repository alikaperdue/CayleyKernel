# CayleyKernel

**GPU Cayley-Dickson doubling products with O(1) basis-sign determination.**

CayleyKernel computes the Cayley-Dickson doubling product of arbitrary
`2^n`-dimensional hypercomplex algebras (complex numbers, quaternions,
octonions, sedenions, and beyond) on NVIDIA GPUs. For a given row and column
of the Cayley-Dickson sign table, a closed-form function determines the sign
in **O(1) time** using a fixed number of bitwise operations and a population
count on fixed-width GPU integers, without traversing the doubling levels
or storing the sign table. Each output component is computed by its own
thread, and independent products are batched in a single kernel launch via
a grid-stride loop. Each thread sums `dim` signed terms for its output;
a full dense product therefore requires `O(dim^2)` work.

* Kernel source: [`cuda/cuda_kernel.cu`](cuda/cuda_kernel.cu) (NVIDIA CUDA) and
  [`cuda/amd_kernel.cu`](cuda/amd_kernel.cu) (AMD HIP/ROCm port).
* Python package: [`CayleyKernel`](CayleyKernel/__init__.py) — the CuPy/ctypes GPU wrapper
  and CPU basis operations.
* The O(1) basis-sign method is based on the paper by [`Maher Ben Abdessalem`](https://figshare.com/articles/preprint/A_Proven_Sign_Law_for_Cayley-Dickson_Algebras_Ordinary_and_Split_Constructions/33705022)

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

This repository implements that product at each doubling level. The algebras
become noncommutative from the quaternions and nonassociative from the
octonions onward. The GPU evaluates the product in a single kernel launch:

* each of the `2^n` output components is computed by an independent thread;
* the basis sign `e_row · e_col = ± e_{row⊕col}` is determined in O(1)
  time for fixed-width GPU indices using bitwise operations and `__popc()`;
* a `(B, …, dim)` input is treated as `prod(B, …)` independent products
  in a single launch (grid-stride over the batch axis).

## Installation

### Build from source

Dense CUDA multiplication requires an NVIDIA GPU, the CUDA toolkit (`nvcc`
on PATH), and a Python environment with NumPy and a compatible CuPy build.
The package requires Python ≥ 3.10; dependency versions must also
support the selected interpreter. Tests need pytest; research suites also use
SciPy and SymPy.

```bash
git clone https://github.com/alikaperdue/CayleyKernel.git
cd CayleyKernel

python3 -m pip install -e '.[test]' scipy sympy
make kernel                 # compile cuda/cuda_kernel.cu -> CayleyKernel/cuda_kernel.so
make test                   # run the default suite, excluding slow tests
```

The Makefile targets several GPU architectures by default. If your CUDA
toolkit does not support every target, select an architecture supported by
both your GPU and toolkit, for example:

```bash
make kernel ARCHS="sm_80"  # example for an sm_80 GPU
```

AMD HIP/ROCm is experimental and must be built separately with
`make kernel-amd` in a ROCm-compatible CuPy environment. This target writes
`CayleyKernel/cuda_kernel.so`, the filename the wrapper loads, replacing any
NVIDIA build at that path. There is no automatic CUDA/ROCm probing or OpenCL
backend. The HIP kernel does not include the CUDA launch tuning below or BF16
support; rebuild with `make kernel` when returning to NVIDIA.

CUDA launch tuning:

```bash
CAYLEY_KERNEL_BLOCK_SIZE=512 python3 your_script.py
```

The optional override accepts `32`, `64`, `128`, `256`, `512`, or `1024`
threads and is read on every multiplication. Invalid values raise `ValueError`
in Python; direct C callers fall back to the default. With the variable unset,
the heuristic uses 128 threads for dimensions up to 16 and 256 otherwise,
reducing to as few as 32 threads for small workloads. These are starting
points, not a measured autotuner: benchmark your dimensions, batch sizes and
dtypes on your GPU before choosing an override. The benchmark suite now also
covers batches of 4096, 16384, 65535 and 70000 small vectors; record a new
baseline under a new filename because older baselines lack these cases.
Rebuild the CUDA library to use these settings; older libraries ignore the
override.

Blocks cover a flattened batch/component axis, packing multiple vectors per
block where possible. All batch sizes use a grid-stride loop, capped at 65535
work blocks. Dimensions up to 16 reuse both input vectors through shared
memory. Each block-size specialization has `__launch_bounds__(block_size, 1)`;
more aggressive occupancy hints need measurements before adoption.

FP32 retains its existing arithmetic. No `CAYLEY_KERNEL_TF32` switch is
provided: TF32 throughput comes from Tensor Core matrix operations, and
rounding scalar inputs to TF32 or BF16 does not turn this kernel into a
Tensor Core implementation. A future reduced-precision path needs a separate
matrix formulation and accuracy/performance validation. See
[NVIDIA’s TF32 explanation](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/).

To build a wheel containing the compiled kernel:

```bash
python3 -m pip install build
make wheel
```

See [wheel building and portability](docs/building.md) for architecture
selection, optional `auditwheel` repair, and wheel-build regression tests.

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

Supported dtypes: `float16`, `float32`, `float64`, `int8/16/32/64`,
`uint8/16/32/64`, `bool`, `complex64`, `complex128`, and CUDA `bfloat16`.
Inputs may be CuPy arrays, NumPy arrays, or Python lists. If either input is a
CuPy array, the output is CuPy; otherwise a NumPy input produces a NumPy
output, and two lists produce a list. Dense multiplication requires a GPU
regardless of input container.

FP16 and BF16 use float32 products and accumulation before rounding to the
output type. BF16 requires CuPy 14+, NumPy 2.1.2+, `ml_dtypes`, and a CUDA
build; HIP does not export BF16 entry points. Complex dtypes represent
complex coefficients in the existing basis; multiplication is bilinear and
basis conjugation does not complex-conjugate those coefficients.

See [coefficient types and precision](docs/precision.md) for promotion rules,
examples, overflow behavior, prerequisites, and validation commands.

### API

| Function | Description |
| --- | --- |
| `multiply(a, b)` | Cayley-Dickson product; last axis = algebra dimension `2^n`, leading axes batched + broadcast |
| `conjugate(a)` | Standard basis conjugation `(a0, −a1, …, −a_{n−1})` |
| `norm_sq(a)` | Sum of coefficient squares over the last axis |
| `inverse(a)` | `conjugate(a) / norm_sq(a)`; raises on zero norm, warns at dimensions ≥ 16 |
| `right_div(a, b)` | `multiply(a, inverse(b))` |
| `left_div(a, b)` | `multiply(inverse(a), b)` |
| `power(a, n)` | Integer powers by binary exponentiation; negative exponents use `inverse` |
| `exp(a)` | Exponential via scalar exponential and imaginary-part sine/cosine |
| `inner(a, b)` | Scalar coefficient of `a * conjugate(b)`; dot product for real coefficients |
| `real(a)` | Extract `e0`, removing the algebra axis |
| `imag(a)` | Copy with `e0` set to zero, preserving the algebra axis |
| `library_path()` | Path of the loaded `cuda_kernel.so` |
| `loaded()` | `True` once the kernel library has been found |
| `abi_version()` | Kernel ABI version (`0` = no kernel library, `1` = legacy, `2` = current) |
| `twist(r, c)` | Exact CPU basis sign; nonnegative, zero-based integer indices |
| `basis(r, c)` | Exact CPU basis product; nonzero, signed one-based integer labels |
| `twist_many(pairs)`, `basis_many(pairs)` | Ordered CPU batch operations for iterables of pairs |
| `cpu_backend()` | Active CPU implementation: `python` or `rust` |

The algebra helpers accept lists, NumPy arrays, and CuPy arrays, with the last
axis as the power-of-two algebra dimension. Results preserve the container
family (tuples return lists); scalar helpers return Python scalars for lists
and array scalars for array inputs. `imag` retains the full coefficient vector.
Coefficient-only helpers work without CUDA. Division and nontrivial powers use
the GPU `multiply` backend. Integer norm, inverse, power, exponential, and inner
product inputs promote to float64; float16 inputs promote to float32 for these
operations. Unsigned/boolean conjugation promotes to float64 to allow negation.

For complex coefficients, `norm_sq` and `inner` are **bilinear**, matching basis
conjugation, rather than Hermitian; `real` extracts the possibly complex `e0`
coefficient. A nonzero complex vector can have zero `norm_sq` and cannot then be
inverted by this formula. Any zero-norm member makes a batched inverse fail.

At dimensions ≥ 16, `inverse` emits `RuntimeWarning` because positive norm does
not exclude zero divisors. The formula gives a two-sided element inverse for
nonzero real vectors, but cancellation and unique division solutions need not
hold. The warning applies to all inputs at these dimensions; it is not a
zero-divisor detector. Division computes the stated ordered products.

```python
from CayleyKernel import conjugate, norm_sq, inverse, exp, real, imag

q = [1.0, 2.0, 3.0, 4.0]
assert norm_sq(q) == 30.0
assert conjugate(q) == [1.0, -2.0, -3.0, -4.0]
assert imag(q) == [0.0, 2.0, 3.0, 4.0]
q_inverse = inverse(q)
q_exponential = exp(q)
```

### CPU basis operations

`twist` and `basis` work without CUDA, NumPy, or CuPy. From a source checkout,
install the package without its GPU dependencies if only these operations
are needed:

```bash
python3 -m pip install --no-deps -e .
```

```python
from CayleyKernel import twist, basis

assert twist(1, 1) == -1  # zero-based indices: e1 * e1 = -e0
assert basis(2, 2) == -1  # signed one-based labels: e1 is 2, -e0 is -1
assert basis(-2, 2) == 1  # (-e1) * e1 = e0
```

`twist(r, c)` returns the sign at row `r`, column `c`; the output basis index
is `r ^ c`. `basis(r, c)` encodes both the sign and output basis unit in one
signed label. Zero is a valid `twist` index and an invalid `basis` label.

For the CPU test collection, install its dependencies and run:

```bash
python3 -m pip install numpy pytest scipy sympy
make test-cpu
```

### Optional Rust CPU acceleration

The sparse CPU operations used by the research scripts can use a Rust extension
while retaining arbitrary-size integer indices. Dense CUDA/HIP products keep
their existing implementation. With a stable Rust toolchain installed:

```bash
python3 -m pip install ./rust
python3 -c "from CayleyKernel import cpu_backend; print(cpu_backend())"
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
│   ├── cuda_kernel.cu       NVIDIA CUDA kernel (primary target)
│   └── amd_kernel.cu        AMD HIP/ROCm port of the same kernel
├── tests/                   pytest suite (reference impl + identities)
├── docs/                    build, precision, validation, and research docs
├── rust/                    optional CPU basis-operation accelerator
├── scripts/build_wheel.sh   kernel -> package -> wheel -> auditwheel
├── Makefile                 kernel / test / wheel / clean targets
├── pyproject.toml           PEP 621 metadata (name: cayley_kernel)
└── LICENSE                  Apache-2.0
```

The compiled `.so` is **not** committed: build it with `make kernel`, or
download a wheel from Releases.

## Testing

```bash
make test          # default suite, excluding slow tests
make test-cpu      # CPU-marked checks
make test-all      # default suite followed by slow tests
```

`test-all` excludes the opt-in sparse zero-divisor and benchmark checks;
run `make test-sparse-zd` and `make benchmark` separately to enable them.
The obsolete `run-native-zd` target has been removed because its C source
is unavailable. Use `make zd-showcase` for huge-index zero-divisor checks.

The default suite covers small dense products against a recursive reference,
batching, broadcasting, dtypes, and algebraic identities. Norm multiplicativity
is checked only in dimensions 1, 2, 4, and 8. Higher-dimensional checks cover
other identities and sparse CPU operations; they do not establish norm
multiplicativity beyond the octonions.

Some experimental correctness functions are currently named `FAILStest_...`
and are not collected by pytest. They are excluded from this coverage summary.
Runtime depends on the selected tests, hardware, and installed dependencies.

* `slow`: large dense products at dimensions `2^20`–`2^22`, selected with
  `python3 -m pytest -q -m slow tests/`. Dense work is quadratic in dimension.
* `benchmark`: GPU timings require `--run-benchmarks` and an explicit baseline
  recording or comparison path. Selecting `-m benchmark` alone does not enable them.
* `sparse_zd`: additional CPU sparse validation requires `--run-sparse-zd`.

Details and commands for [GPU performance baselines, huge-index showcases,
and sparse zero-divisor validation](docs/testing.md) are documented separately.
For research-specific checks, see the [paper verification index](docs/papers/README.md).

## Naming

| Context | Name |
| --- | --- |
| GitHub repository / branding | `CayleyKernel` |
| Python package / import | `CayleyKernel` |
| Wheel / PyPI project name | `cayley_kernel` |
| Compiled kernel binary | `cuda_kernel.so` |
| C ABI symbols | `mult_<type>`, `mult_<type>_nd` (stable) |

The **O(1) (constant-order)** designation refers to determining the sign at
a given row and column of the Cayley-Dickson sign table. For fixed-width GPU
indices, this uses a fixed number of operations independent of the doubling
level. Dense multiplication uses this sign function for every term and
requires `O(batch * dim^2)` total work. Python's arbitrary-size integer
operations additionally depend on the indices' bit lengths.

## Development and AI assistance

CayleyKernel was initially authored by hand and is developed with AI
assistance for implementation, refactoring, and analysis. The maintainer
directs the work and is responsible for reviewing changes and assessing
test results before integration.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

© 2026 Yotta Space Corp
