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
  hardware acceleration based on the paper by Maher Ben Abdessalem:

https://figshare.com/articles/preprint/A_Proven_Sign_Law_for_Cayley-Dickson_Algebras_Ordinary_and_Split_Constructions/33705022?file=69441246

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

make kernel                 # compile cuda/kernel.cu -> cuda_kernel.so
make test                   # run the pytest suite

pip install -e .            # optional: editable install of the wrapper
```
Build a redistributable wheel (compiles the kernel, bundles it, and runs
`auditwheel repair` when available):

```bash
pip install build auditwheel
make wheel
```

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

Supported dtypes: `float32`, `float64`, `int8/16/32/64`, `uint8/16/32/64`,
`bool`.  Inputs may be CuPy arrays, NumPy arrays, or plain Python lists;
the output container type matches the inputs.

### API

| Function | Description |
| --- | --- |
| `multiply(a, b)` | Cayley-Dickson product; last axis = algebra dimension `2^n`, leading axes batched + broadcast |
| `conjugate(x)` | Standard conjugation `(x0, −x1, …, −x_{n−1})` |
| `library_path()` | Path of the loaded `cuda_kernel.so` |
| `loaded()` | `True` once the kernel library has been found |
| `abi_version()` | Kernel ABI version (current builds report `2`) |

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


