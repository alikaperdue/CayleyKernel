# Coefficient types and precision

[Back to the README](../README.md)

Commands below run from the repository root. Python examples assume:

```python
import cupy as cp
from CayleyKernel import multiply
```

Supported dtypes: `float16`, `float32`, `float64`, `int8/16/32/64`, `uint8/16/32/64`,
`bool`, `complex64`, `complex128`, and CUDA `bfloat16`. Inputs may be CuPy arrays, NumPy arrays,
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

Run `python3 -m pytest -q tests/test_half.py` after rebuilding to check FP16
products against a recursive reference, batching, dtype promotion, both C entry
points, subnormals, output rounding, and cancellation of terms that exceed the
FP16 range. GPU tests require CuPy and a working GPU runtime; wrapper and ABI
checks can run on the CPU with a loadable compiled library.

BF16 uses `ml_dtypes.bfloat16` with **2-byte storage and float32 products and
accumulation**, rounding once to BF16 on output. It retains float32's exponent
width with less precision; float32 intermediates and BF16 output can still
overflow. It requires a CUDA build (CUDA 11+ headers), CuPy 14+, NumPy 2.1.2+,
and `ml_dtypes`. See the [CuPy upgrade guide](https://docs.cupy.dev/en/stable/upgrade.html#minimal-support-for-bfloat16)
for runtime limitations, including CUDA 12.1 compilation issues. The HIP build
does not export BF16 entry points. Existing FP16 and other dtypes do not require
`ml_dtypes` or a CuPy upgrade.

```python
import ml_dtypes

a = cp.array([1, 2, 3, 4], dtype=ml_dtypes.bfloat16)
b = cp.array([4, 3, 2, 1], dtype=ml_dtypes.bfloat16)
c = multiply(a, b)  # BF16 output, FP32 arithmetic
```

Two BF16 operands preserve BF16. For mixed dtypes, BF16 is first treated as
float32 for promotion, then CuPy's `result_type` applies (BF16 + FP16 gives
float32; BF16 + float64 gives float64). A real Python list in the second
operand follows the first operand's BF16 dtype. Norms, inverses, powers, exponentials, inner products, and division
helpers promote BF16 to float32 before arithmetic.

Install `ml_dtypes` in your CuPy 14+ environment, rebuild, and run:

```bash
python3 -m pip install ml_dtypes
make kernel ARCHS="sm_120"  # RTX 50 series; select your own architecture
python3 -m pytest -q tests/test_bfloat16.py tests/test_half.py tests/test_complex.py
```

BF16 CPU wrapper tests run with `python3 -m pytest -q -m cpu tests/test_bfloat16.py`.
GPU checks in that file skip when CuPy 14+, ml_dtypes, or a CUDA device is
unavailable; with a supported runtime, a missing/stale BF16 library fails.

FP8 and packed INT4/FP4 are not supported. This scalar kernel does not use
Tensor Core MMA instructions, so registering these storage types would not
provide Tensor Core throughput. Packed formats also need a separate logical
shape, packing, scaling, and rounding contract; `uint8` continues to mean one
ordinary integer coefficient per byte. The optional Rust CPU basis backend
does not transport coefficient buffers and needs no BF16 bridge.

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
