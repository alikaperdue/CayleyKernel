# Optional Rust CPU accelerator

This crate accelerates dense real `multiply`, exact `twist`, and signed `basis` operations used by
CayleyKernel's sparse algebra and research scripts. It is also a standalone Rust
library. Dense GPU multiplication continues to use the existing CUDA/HIP kernels.

The core uses `u64` arithmetic through 64 bits, stack-only two/four-limb paths
through 128/256 bits, and `num-bigint` for larger indices.
No basis index is truncated to machine width. The large-integer sign calculation
uses trailing-zero counts and bit parity instead of general integer division.
Cost and memory still grow with index bit length; this is not constant-time
arithmetic for arbitrarily large indices.

## Install from this repository

Install a stable Rust toolchain with Cargo, activate your Python environment, and
run from the repository root:

```sh
python -m pip install ./rust
# Install the main Python wrapper and declare the Rust dependency
python -m pip install -e ".[rust]"
python -c "from CayleyKernel import cpu_backend; print(cpu_backend())"
```

Prebuilt companion wheels are built for Linux x86_64, macOS arm64, and
Windows x86_64 on Python 3.10–3.14. See the main
[wheel installation guide](../docs/building.md) for release downloads and the
`cayley-kernel[rust]` extra. A source installation requires a Rust toolchain.

The extension is distributed separately as `cayley-kernel-rust` and imports as
`cayley_kernel_rust`. Building from source requires Rust; using a built wheel does
not. The ordinary CayleyKernel package retains its setuptools build and does not
require Rust. Neither the extension nor CPU basis operations require CUDA, HIP,
NumPy, or CuPy. The main package requests NumPy; GPU support uses optional CuPy
extras. The `cpu_multiply` wrapper returns a NumPy array.

`CAYLEY_KERNEL_CPU_BACKEND` selects the backend at import time:

| Value | Behavior |
| --- | --- |
| `auto` (default) | Use Rust if importable; otherwise use Python. |
| `python` | Always use the Python implementation. |
| `rust` | Require Rust; raise an actionable import error if unavailable. |

Both backends accept integer objects implementing `__index__`. `twist` uses
zero-based nonnegative indices; `basis` uses signed one-based labels. Negative
twist indices and zero basis labels now raise `ValueError` consistently; they
were outside the mathematical API previously. Noninteger inputs raise `TypeError`.

```python
from CayleyKernel import basis, basis_many, cpu_backend, cpu_multiply, twist, twist_many

print(cpu_backend())
assert twist(3, 6) == 1
assert basis(4, 7) == 6
assert twist_many([(3, 6), (6, 3)]) == [1, -1]
assert basis_many([(4, 7), (-4, 7)]) == [6, -6]
assert cpu_multiply([3, 4], [1, 2]).tolist() == [-5.0, 10.0]
```

Batch functions accept iterables of pairs and return lists in input order. They
reduce Python-to-Rust calls but materialize their input, so chunk large streams.
They preserve exact integer results. They do not launch GPU kernels.

## Dense real products

`cayley_kernel_rust.multiply(a, b)` accepts real coefficient sequences and returns
`list[float]`. `CayleyKernel.cpu_multiply(a, b)` selects the configured CPU backend
and returns a one-dimensional NumPy `float64` array. Inputs must have the same
nonzero power-of-two length. Invalid lengths raise `ValueError` in Python;
the standalone Rust `multiply(&[f64], &[f64]) -> Vec<f64>` asserts this precondition.
The wrapper rejects multidimensional and complex inputs consistently for both
backends. An older companion extension without `multiply` uses the Python fallback.

Dense multiplication still takes O(n²) work and O(n) output memory. Each sign uses
the fixed-width twist law. Four output coefficients are tallied together with
manual loop unrolling, preserving ascending row order for every floating-point
sum. This provides instruction-level parallelism without a Rayon dependency or
thread startup cost; it does not promise SIMD instructions. The Python binding
releases the GIL during the calculation after copying inputs. Sequence conversion
and the result allocation remain part of end-to-end costs. `py_multiply` stays
the pure Python reference; the existing CuPy `multiply` API continues to use GPUs.

## Embedded and WASM library builds

```sh
cargo build --manifest-path rust/Cargo.toml --locked --no-default-features --features no_std
rustup target add wasm32-unknown-unknown
cargo check --manifest-path rust/Cargo.toml --locked --no-default-features \
  --features no_std --target wasm32-unknown-unknown
```

`no_std` enables `#![no_std]` with `alloc`; applications must supply an allocator
and their platform's panic/runtime support. It is not an allocation-free API.
`num-bigint` disables default features. PyO3 remains optional; enabling `python`
uses `std` even when `no_std` is also selected, allowing additive Cargo features.
The crate builds as an `rlib` for native Rust consumers. Maturin requests the
`cdylib` type when building the Python extension.

The Python build backend is already explicitly declared as
`maturin>=1.9.4,<2` in `rust/pyproject.toml`; setuptools-rust is not needed.
To build a release wheel without installing it, run from the repository root:

```sh
python -m pip install 'maturin>=1.9.4,<2'
python -m maturin build --release --locked --manifest-path rust/Cargo.toml \
  --interpreter python --out rust/target/wheels
```

## Verify and benchmark

From the repository root:

```sh
cargo test --manifest-path rust/Cargo.toml --locked
make rust-check
make test-cpu
CAYLEY_KERNEL_CPU_BACKEND=rust CAYLEY_TEST_REQUIRE_RUST=1 \
  python -m pytest -q tests/core/test_cpu_backend.py
python scripts/benchmark_cpu_backends.py
# Compare native fixed-width dispatch with the generic BigUint calculation:
cargo test --manifest-path rust/Cargo.toml --locked --release \
  benchmark_index_paths -- --ignored --nocapture
```

The Python tests compare both backends with an independent doubling oracle,
exercise signs and 32/64/128/256-bit boundaries, and check million-bit indices, invalid
inputs, batch ordering, and import behavior without optional dependencies.
CI runs CPU tests with and without the extension on Linux, macOS, and Windows.

The benchmark verifies equal outputs before timing identical pre-generated
inputs and reports median end-to-end operation times, including Python/Rust
integer conversion. Results depend on index width, workload, and machine; there
is no universal speedup claim or timing gate. Compare real research workloads
with `CAYLEY_KERNEL_CPU_BACKEND=python` and `rust` before drawing conclusions.
The default benchmark includes dense products and boundary widths. The native
microbenchmark uses sparse high-bit indices and five-sample medians; the Python
benchmark uses random indices and includes Python/Rust conversion costs. See the
[measured crossover and dense-product results](../docs/rust-cpu-benchmarks.md).

The Rust crate exposes `twist(&BigUint, &BigUint) -> i8` and
`basis(&BigInt, &BigInt) -> Option<BigInt>` without enabling the `python` feature.
`None` denotes a zero basis label. Python builds enable the optional PyO3 bindings
through Maturin. `Cargo.lock` records the tested dependency resolution.

## Build cleanup

`rust/target/` contains generated Cargo/Maturin binaries, caches, and wheels;
it is ignored by Git and can be regenerated. From the repository root,
`make clean` removes it along with local Rust build and distribution folders
and the Python/CUDA build artifacts. This does not uninstall an extension
already installed in a Python environment or clear Cargo's global cache.
Keep `src/`, `Cargo.toml`, `Cargo.lock`, `pyproject.toml`, `README.md`, and
`LICENSE` in Git.
