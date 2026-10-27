# Optional Rust CPU accelerator

This crate accelerates the exact `twist` and signed `basis` operations used by
CayleyKernel's sparse algebra and research scripts. It is also a standalone Rust
library. Dense GPU multiplication continues to use the existing CUDA/HIP kernels.

The core uses `u64` arithmetic when possible and `num-bigint` for larger indices.
No basis index is truncated to machine width. The large-integer sign calculation
uses trailing-zero counts and bit parity instead of general integer division.
Cost and memory still grow with index bit length; this is not constant-time
arithmetic for arbitrarily large indices.

## Install from this repository

Install a stable Rust toolchain with Cargo, activate your Python environment, and
run from the repository root:

```sh
python -m pip install ./rust
# Optional: install the main Python wrapper without downloading GPU dependencies
python -m pip install --no-deps -e .
python -c "from CayleyKernel import cpu_backend; print(cpu_backend())"
```

The extension is distributed separately as `cayley-kernel-rust` and imports as
`cayley_kernel_rust`. Building from source requires Rust; using a built wheel does
not. The ordinary CayleyKernel package retains its setuptools build and does not
require Rust. Neither the extension nor CPU basis operations require CUDA, HIP,
NumPy, or CuPy. Installing the main package with dependencies still requests its
existing NumPy/CuPy dependencies.

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
from CayleyKernel import basis, basis_many, cpu_backend, twist, twist_many

print(cpu_backend())
assert twist(3, 6) == 1
assert basis(4, 7) == 6
assert twist_many([(3, 6), (6, 3)]) == [1, -1]
assert basis_many([(4, 7), (-4, 7)]) == [6, -6]
```

Batch functions accept iterables of pairs and return lists in input order. They
reduce Python-to-Rust calls but materialize their input, so chunk large streams.
They preserve exact integer results. They do not launch GPU kernels, and this
first version does not promise Python thread parallelism.

## Verify and benchmark

From the repository root:

```sh
cargo test --manifest-path rust/Cargo.toml --locked
make rust-check
make test-cpu
CAYLEY_KERNEL_CPU_BACKEND=rust CAYLEY_TEST_REQUIRE_RUST=1 \
  python -m pytest -q tests/test_cpu_backend.py
python scripts/benchmark_cpu_backends.py
```

The Python tests compare both backends with an independent doubling oracle,
exercise signs and 32/64-bit boundaries, and check million-bit indices, invalid
inputs, batch ordering, and import behavior without optional dependencies.
CI runs CPU tests with and without the extension on Linux, macOS, and Windows.

The benchmark verifies equal outputs before timing identical pre-generated
inputs and reports median end-to-end operation times, including Python/Rust
integer conversion. Results depend on index width, workload, and machine; there
is no universal speedup claim or timing gate. Compare real research workloads
with `CAYLEY_KERNEL_CPU_BACKEND=python` and `rust` before drawing conclusions.

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
