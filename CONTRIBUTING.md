# Contributing to CayleyKernel

Thank you for considering a contribution. This document covers code style,
branch naming, the build/test/lint entry points, and the concrete steps for
the most common non-trivial change (adding a supported dtype).

The project has three source languages:

| Component | Path | Toolchain |
| --- | --- | --- |
| Python wrapper + CPU helpers | `CayleyKernel/` | Python ≥ 3.10 |
| Rust CPU accelerator (optional) | `rust/` | Rust 1.74+ (stable) |
| CUDA / HIP kernels | `cuda/` | `nvcc` / `hipcc` |
| Tests | `tests/` | `pytest` |
| Build / packaging | `Makefile`, `pyproject.toml`, `setup.py` | `setuptools`, `auditwheel` |

## Branch naming

No convention is enforced by CI, but please follow it in pull requests so the
review history stays readable:

```
<type>/<short-slug>
```

`<type>` is one of: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`,
`build`, `perf`. `<short-slug>` is a kebab-case summary, e.g.:

```
feat/bfloat16-complex64
fix/broadcast-out-buffer
docs/dtype-addition-walkthrough
test/sparse-zd-regression
```

## Local development setup

### Python (no GPU required for CPU work)

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e .            # core package + NumPy
python -m pip install -e ".[test]"    # pytest
```

For the optional Rust CPU backend:

```bash
python -m pip install -e ./rust
python -c "from CayleyKernel import cpu_backend; print(cpu_backend())"
# -> rust
```

### GPU build

Requires an NVIDIA GPU, the CUDA toolkit with `nvcc` on `PATH`, and a
matching CuPy build:

```bash
python -m pip install cupy-cuda12x   # or cupy-cuda11x
make kernel                          # -> CayleyKernel/cuda_kernel.so
```

`make kernel ARCHS="sm_86 sm_90"` limits the build to specific architectures
for faster local iteration. AMD: `make kernel-amd` (requires ROCm/HIP).

## Lint and format

**Enforced by CI (`.github/workflows/cpu.yml`):**

```bash
# Python (ruff: linter + formatter)
ruff check .            # style + bugbear + isort
ruff format --check .   # formatter drift check

# C/CUDA (clang-format, Google base + project overrides)
clang-format --dry-run --Werror cuda/cuda_kernel.cu cuda/amd_kernel.cu cuda/tools/compare_sign_tables.cpp

# Rust (format + lint + tests + no_std)
cargo fmt     --manifest-path rust/Cargo.toml --check
cargo clippy  --manifest-path rust/Cargo.toml --locked --all-targets \
              --features python -- -D warnings
cargo test    --manifest-path rust/Cargo.toml --locked
cargo check   --manifest-path rust/Cargo.toml --locked \
              --no-default-features --features no_std
```

The one-liner for the Python + C/CUDA checks is `make lint`.
The one-liner for the Rust checks is `make rust-check`.
The one-liner to auto-fix Python + C/CUDA drift is `make format`
(do **not** use it to fix Rust drift; run `cargo fmt --manifest-path rust/Cargo.toml` separately if Rust drift ever appears).

### Python (ruff)

Configuration lives in `pyproject.toml` under `[tool.ruff]`. Rules enabled:
`E` (pycodestyle errors), `W` (pycodestyle warnings), `F` (pyflakes),
`I` (isort), `B` (flake8-bugbear). Rules deliberately **off**:
`ANN` (type hints) and `D` (docstrings) — the codebase does not use type
hints on internal helpers and does not require docstrings on every helper;
forcing them would be a separate policy change. Line length is 100;
the formatter owns wrapping (`E501` is ignored so the linter and formatter
don't fight).

Style notes for new code:

- 4-space indentation, double-quoted docstrings.
- `# SPDX-License-Identifier: Apache-2.0` header on every new `.py` file.
- No trailing whitespace, no type hints on internal helpers (the existing
  code does not use them).
- Import ordering follows `isort` (ruff's `I` rules); group stdlib / third-
  party / local, alphabetical within group.

### C/CUDA (clang-format)

Configuration lives in `.clang-format` (Google base with project overrides:
`ColumnLimit: 100`, `SortIncludes: false`, `PointerAlignment: Left`,
`DerivePointerAlignment: false`, `AlignTrailingComments: true`,
`AllowShortFunctionsOnASingleLine: Inline`).

Style notes for new code:

- `// SPDX-License-Identifier: Apache-2.0` header on every new `.cu` / `.cpp` file.
- 4-space indentation, `//` comments, no tabs.
- `constexpr` for compile-time constants.
- `const T* a` (pointer left-aligned) in signatures.
- Preserve the existing include order in the kernel files (CUDA runtime
  headers before project headers); do **not** add `#include <...>` in
  alphabetical order — `.clang-format` is configured to not sort includes
  for this reason.

## Testing

All tests live in `tests/` and use `pytest` markers declared in
`pyproject.toml`:

| Marker | Meaning |
| --- | --- |
| *(default)* | Core correctness, identity, broadcast, dtype tests. |
| `cpu` | Runs without a GPU. |
| `slow` | Heavy identity / batch tests at `2^20` and above. |
| `sparse_zd` | CPU sparse zero-divisor validation (opt-in via `--run-sparse-zd`). |
| `benchmark` | GPU timing checks (opt-in via `--run-benchmarks`). |

Run the default suite (no GPU needed if `CAYLEY_KERNEL_CPU_BACKEND=python`):

```bash
make test                  # pytest -q -m "not slow" tests/
make test-all              # + the slow suite
make test-cpu              # -m cpu only, safe without a GPU
```

The opt-in suites and their flags are in the `Makefile`; they are not run in
CI. See [`docs/testing.md`](docs/testing.md) for details and
[`docs/benchmark-methodology.md`](docs/benchmark-methodology.md) for the
benchmark reproduction workflow.

## Pull requests

1. Fork and create a branch following the naming convention above.
2. Keep the change focused. A PR that adds a dtype and also refactors the
   kernel should be two PRs.
3. Update the relevant `docs/` page when user-visible behaviour changes.
4. Run `make lint` (Python + C/CUDA) and `make rust-check` (Rust) before
   pushing. Run `make test` (or `make test-all` if you touched the slow
   path) if you touched the kernel, the Python wrapper, or `tests/`.
5. Open a PR against `main`. CI will run the `lint` job, the CPU/Rust
   matrix, and the wheel builds.

## Adding a supported dtype

This is the most common non-trivial change and touches exactly three places.
The existing `float16` and `bfloat16` dtypes are good templates — read
`tests/test_half.py` and `tests/test_bfloat16.py` first.

### 1. The kernel entry points (`cuda/cuda_kernel.cu`, `cuda/amd_kernel.cu`)

Each supported type exposes two symbols (the suffix is part of the stable
ABI; keep it in sync across the Python wrapper and both kernels):

```
extern "C" {
    void mult_<suffix>(const T* a, const T* b, T* out, int dim);
    void mult_<suffix>_nd(const T* a, const T* b, T* out, int dim, long long batch);
}
```

The templated kernel and the `accum_traits` / `arithmetic_traits` /
`negation` specializations already cover the common cases; you typically only
need to add the `extern "C"` exports for the new suffix, and a `static_assert`
if the new type has a platform-dependent size.

If you are adding a **new kernel family** (e.g. fused multiply-add), you
must also emit the new entry point for the same suffix in **both**
`cuda/cuda_kernel.cu` and `cuda/amd_kernel.cu`, add a matching benchmark
case in `tests/benchmark_support.py` (a new `CASE_SHAPES_*` dict), and add
corresponding tests in `tests/`. Bump `cayley_abi_version()` in the kernel
and the `abi_version()` docstring in `CayleyKernel/__init__.py`.
Record a **new** baseline file under `benchmark-results/` (do not overwrite
an existing one).

Rebuild and smoke-test:

```bash
make kernel
python -c "import CayleyKernel as ck; ck._select_kernel(np.dtype('<new>'))"
```

### 2. The Python wrapper (`CayleyKernel/__init__.py`)

Add the suffix to `_TYPE_SUFFIXES` (the tuple that drives the `getattr`
loop that binds `mult_<suffix>` and `mult_<suffix>_nd`), and add a mapping
from the NumPy/CuPy dtype name to that suffix in `_DTYPE_TO_SUFFIX`:

```python
_TYPE_SUFFIXES = (
    # ... existing entries ...
    "mynewtype",  # e.g. "float8_e4m3"
)

_DTYPE_TO_SUFFIX = {
    # ... existing entries ...
    "mynewtype": "mynewtype",
}
```

If the new dtype needs special accumulation (e.g. it accumulates in a wider
type), add a specialisation to `accum_traits<T>` in the kernel rather than in
Python. If it needs a custom promotion rule (like `bfloat16` does), extend
`_result_dtype` in `__init__.py` and add a test for it.

### 3. Tests (`tests/`)

Copy the closest existing dtype test file (`test_half.py` or
`test_bfloat16.py`) and adapt it. At minimum cover:

- `_select_kernel` returns a `(legacy, nd)` pair for the new dtype.
- A small batch product matches the reference implementation in
  `tests/reference.py` for a few dimensions (2, 4, 8).
- Broadcasting against a smaller batch shape works.
- The `out=` buffer reuse path works and does not allocate.
- If the dtype has promotion quirks (like `bfloat16`), assert the promoted
  dtype explicitly.

Run the full suite before pushing:

```bash
make lint              # Python + C/CUDA format/lint
make rust-check        # Rust format/lint/tests/no_std (if you touched rust/)
make test              # or: make test-all for the slow path
```

## Multi-GPU

`CayleyKernel/_distributed.py` (`visible_devices` / `split_batch`) ships a
zero-dependency multi-GPU batch split built on CuPy's `cp.cuda.Device`
context manager and one non-blocking `cp.cuda.Stream` per device. The CPU
queues work on every device before calling
`cp.cuda.runtime.deviceSynchronize()` at the end, and `gather_into`
copies each chunk back onto the caller's current device in the original
batch order.

Guidance for adding a new device topology (NVLink vs. PCIe, or a specific
GPU pair) without breaking the helper:

1. **Do not hard-code device IDs or interconnect bandwidth in the helper.**
   `visible_devices()` reads `cp.cuda.runtime.getDeviceCount()` and
   `split_batch` only uses `len(devices)` to compute chunk sizes. A new
   topology is therefore a *deployment* change (set `CUDA_VISIBLE_DEVICES`
   before starting Python), not a code change.

2. **The one place topology matters is the gather cost.** The helper
   copies one chunk per device. NVLink makes this ~2–5× cheaper than
   PCIe 3.0. If you are profiling a new GPU pair:

   ```bash
   nvidia-smi topo -m          # show the link type (NV# vs. SYS/PHB)
   ```

   and, if the D2D copy is a significant fraction of `split_batch` time
   on PCIe-only machines, raise it as a separate PR that adds an optional
   `devices=` policy (e.g. a user-supplied `gather` callback) rather than
   silently changing the default.

3. **Keep the kernel launch path identical.** `split_batch` calls the same
   private `_launch` entry point that `multiply` uses. Do not add a
   per-topology kernel or stream flag here; topology should influence
   *which devices are used* and *how results are gathered*, never *how a
   single device's kernel is launched*.

4. **Tests.** Any new topology-related test belongs in
   `tests/test_multi_gpu.py` and must keep the module-level
   `skipif(len(visible_devices()) < 2, ...)` guard so single-GPU CI
   runners and the `cpu.yml` lint matrix stay green. CI runs
   `tests/test_multi_gpu.py` with `continue-on-error: true` on
   single-GPU runners.

5. **Documentation.** Update `docs/multi-gpu.md` with a new row in the
   "When split_batch wins / does not" tables and, if the topology changes
   the recommended `CUDA_VISIBLE_DEVICES` ordering, add a short note in
   the `CUDA_VISIBLE_DEVICES` workflow section. The README "Multi-GPU"
   paragraph and the `split_batch` docstring must remain accurate.

6. **No new dependencies.** Do not introduce `torch.distributed`,
   `mpi4py`, `pynvml`, or `nvidia-ml-py`. CuPy's `cp.cuda.Device`,
   `cp.cuda.Stream`, `cp.cuda.runtime.getDeviceCount`, and
   `cp.cuda.runtime.deviceSynchronize` are the only sanctioned primitives.

## Things that are deliberately **not** in this repository

- Precompiled `*.so`, `*.whl`, `*.pyd` binaries (built by `make kernel` /
  `make wheel`; see [`docs/building.md`](docs/building.md)).
- Machine-specific benchmark JSON (`benchmark-results/`).
- Raw paper PDFs, extracted text, page images, and bulk search vectors under
  `docs/papers/` (only the hash-stable CSV evidence is public; see
  [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md)).

See `.gitignore` for the full list and the rationale.

## License

By submitting a contribution, you agree that your contribution will be
licensed under the Apache-2.0 License (see `LICENSE`).
