---
name: cayleykernel
description: Install and use CayleyKernel for Cayley-Dickson basis operations, CPU real-vector products, and GPU batched hypercomplex products, including octonions and sedenions.
---

# CayleyKernel

Use the case-sensitive import `CayleyKernel` (distribution: `cayley_kernel`).
Use this library for Cayley-Dickson products in power-of-two dimensions.
Preserve operand order and parentheses: products are noncommutative from
quaternions and nonassociative from octonions onward.

## CPU first

From a checkout, with Python ≥ 3.10:

```bash
python -m venv .venv
# POSIX; on Windows use .venv\Scripts\activate
source .venv/bin/activate
python -m pip install .
python skill/examples/cpu_octonion.py
```

The [runnable octonion example](examples/cpu_octonion.py) checks
`(1 + 2e1) * (3 + 4e1) = -5 + 10e1` in eight dimensions.
CPU multiplication accepts two one-dimensional real array-likes of equal,
nonzero power-of-two length and returns a NumPy float64 array. It handles
one product per call; NumPy is installed automatically. No CuPy, GPU, native
GPU library, or Rust toolchain is needed.

These are public signatures (checked by `check_drift.py`):

```python
def cpu_multiply(a, b): ...
def multiply(a, b, out=None, stream=None): ...
def twist(r, c): ...
def basis(r, c): ...
def cpu_backend(): ...
def loaded(): ...
def library_path(): ...
def abi_version(): ...
```

`twist` uses nonnegative zero-based basis indices and returns a sign;
`basis` uses nonzero signed one-based labels and returns a signed label.
For example, e1 squared is expressed as `twist(1, 1) == -1` or
`basis(2, 2) == -1`. Ordered batches use `twist_many` and `basis_many`.
Consult the [public exports](../CayleyKernel/__init__.py) and
[README API guide](../README.md#api) before using other helpers.

## Optional Rust acceleration

With a Rust toolchain, run `python -m pip install ./rust`, then
`python -m pip install '.[rust]'` from the checkout. The companion accelerates
CPU basis operations and dense real multiplication. Before importing, set
`CAYLEY_KERNEL_CPU_BACKEND` to `python`, `rust`, or the default `auto`.
`auto` falls back to Python; `rust` requires the companion to import.
Inspect `cpu_backend` to see the selected backend. An older companion without
dense multiplication still uses Python for that operation.

For prebuilt wheels and exact companion versions, follow
[release installation and build support](../docs/building.md#release-wheels-and-cpu-installations).
Release assets are not automatically indexed by pip; supply downloaded
wheels with `--find-links`. See the [Rust guide](../rust/README.md) for details.

## GPU batching

The [GPU batching example](examples/gpu_batch.py) requires an NVIDIA GPU,
compatible driver, exactly one matching CuPy distribution, and a compiled
kernel supporting the GPU architecture. Source builds additionally need
the CUDA toolkit with `nvcc` on PATH and `make`:

```bash
python -m pip install '.[cuda12]'  # cuda11 for CUDA 11.8
make kernel ARCHS="sm_80"         # replace with your GPU/toolkit architecture
python skill/examples/gpu_batch.py
```

Alternatively install a matching CUDA release wheel as described in the
[build guide](../docs/building.md). Restart Python after building or switching
native libraries. `multiply` always uses the GPU, including for lists and
NumPy inputs; it does not fall back to CPU. The last axes must have equal,
nonzero power-of-two length; leading axes broadcast. CuPy inputs return CuPy
arrays, otherwise NumPy inputs return NumPy arrays and lists/tuples return
lists. `out` must be a CuPy destination with the result shape and dtype;
`stream` is a CuPy stream synchronized before the native launch.

For dtype support, promotion, overflow, BF16 prerequisites, and complex
coefficient conventions, read [precision](../docs/precision.md). In particular,
basis conjugation does not conjugate complex coefficients.

AMD HIP/ROCm support is experimental: use `hipcc`, a ROCm-compatible CuPy
environment, and `make kernel-amd`. This overwrites the same
`CayleyKernel/cuda_kernel.so` used by CUDA. Restart Python after switching;
rebuild with `make kernel` to return to NVIDIA. HIP supports batched dense
products but omits BF16 and CUDA block-size tuning. There is no automatic
CUDA/ROCm probing or OpenCL backend. Consult the build guide for current support.

## Diagnose failures

- **Import:** use the same interpreter for pip and execution. Inspect
  `python -m pip show cayley_kernel` and `CayleyKernel.__file__` to detect an
  uninstalled package, wrong virtual environment, or source shadowing. NumPy
  is required for dense CPU products. A forced Rust import failure needs the
  companion installed, or select `python` before import and restart.
- **Native library:** inspect `loaded`, `library_path`, and `abi_version`.
  A successful import or library load does not prove device availability.
  ABI 0 means no loaded library. Build the intended backend and restart;
  missing symbols generally require rebuilding a stale library. If the file
  exists but cannot load, inspect its shared-library dependencies (e.g. `ldd`
  on Linux) and check runtime/platform compatibility. The wrapper searches
  beside the package first, then its parent.
- **Device:** inspect `nvidia-smi` and CuPy's `show_config` and test a small
  CuPy allocation. Check driver/runtime compatibility, device visibility, and
  whether the compiled architectures include your GPU. Rebuild with suitable
  `ARCHS` if needed. Use `cpu_multiply` explicitly for CPU-only execution.

## Maintain this skill

Keep this folder with its repository-relative documentation links, or preserve
the same layout when copying it. From the checkout, after installing the package:

```bash
python skill/check_drift.py --self-test
```

This compares the signature block with the imported public API, binds example
API calls, executes the CPU example in isolation, and checks GPU syntax on
CPU CI. GPU numerical execution is separate: run `gpu_batch.py` on the
configured GPU machine; a syntax check is not a GPU pass.
