# Multi-GPU Batch Splitting

CayleyKernel ships a lightweight multi-GPU helper —
`CayleyKernel.split_batch` — that distributes a large batch of
Cayley-Dickson products across all (or a subset of) visible CUDA devices
using only CuPy's `cp.cuda.Device` context manager and `cp.cuda.Stream`.
No external dependency (torch.distributed, mpi4py, NCCL, …) is required.

See [`CONTRIBUTING.md` § Multi-GPU](../CONTRIBUTING.md#multi-gpu) for
guidance on adding new device topologies without breaking the helper.

---

## API

```python
import CayleyKernel as ck
import cupy as cp

# List visible CUDA device IDs
ck.visible_devices()          # -> [0, 1]  (example: two GPUs)

# Split a batch across all visible devices
C = ck.split_batch(A, B)      # A, B: (batch, dim) arrays

# Restrict to specific devices (e.g. the first two)
C = ck.split_batch(A, B, devices=[0, 1])

# Reuse a pre-allocated output buffer (on the caller's device)
C = cp.empty_like(A)
ck.split_batch(A, B, out=C)
```

### `visible_devices()`

Returns a sorted list of visible CUDA device IDs, e.g. `[0, 1]`.
Returns `[]` when CuPy or the CUDA runtime is unavailable.

### `split_batch(a, b, devices=None, axis=0, out=None, stream=None)`

| Parameter | Description |
|-----------|-------------|
| `a`, `b`  | Operands; last axis must be equal and a power of two |
| `devices` | List of CUDA device IDs; `None` = all visible devices |
| `axis`    | Batch axis to split (only `0` is supported) |
| `out`     | Pre-allocated CuPy output buffer on the caller's device |
| `stream`  | Accepted for API symmetry with `multiply`; per-device streams are created internally |

Returns a single CuPy array on the caller's current device, reassembled
in the original batch order.

---

## Recommended `CUDA_VISIBLE_DEVICES` workflow

The helper uses `cp.cuda.runtime.getDeviceCount()` to discover devices.
If your machine has more GPUs than you want to use for this workload,
set `CUDA_VISIBLE_DEVICES` **before** starting Python:

```bash
# Use only GPUs 0 and 2
CUDA_VISIBLE_DEVICES=0,2 python3 my_script.py
```

This limits `visible_devices()` to `[0, 1]` (the remapped indices)
without any code changes. It also prevents other processes on the
machine from accidentally allocating memory on GPUs 0 and 2.

---

## Memory budget per device

Each device must hold its share of the input chunks plus the output chunk.
For a batch of size `B`, algebra dimension `dim`, and dtype item size `s`
bytes (e.g. 4 for float32, 8 for complex64):

| Tensor | Per-device bytes |
|--------|-----------------|
| Input chunk `a` | `(B / N) × dim × s` |
| Input chunk `b` | `(B / N) × dim × s` |
| Output chunk    | `(B / N) × dim × s` |

where `N = len(devices)`.

**Rule of thumb:** a batch "fits on one GPU" when
`B × dim × s × 3` is below the free VRAM on that device.
`split_batch` wins when the batch is roughly **2× or more** of the
single-GPU capacity, because:

1. The two H2D copies overlap with the kernel launch on the other device.
2. The D2D gather step (one copy per chunk) is amortised over a large
   batch, making it a negligible fraction of total time.

For small batches (e.g. `B < 4096` at dim 32, float32) the per-device
overhead — two copies in, one copy out, plus kernel launch latency —
dominates and `split_batch` is **slower** than a single `multiply` call.

---

## When `split_batch` wins vs. does not

### Wins

| Condition | Why |
|-----------|-----|
| `batch` > ~2× single-GPU capacity | GPU work dominates; copies are amortised |
| `dim` ≤ 256 (compute-bound regime) | Kernel is compute-heavy; splitting hides launch latency |
| Two or more physically separate GPUs | True parallel execution; no memory-bandwidth contention |
| NVLink-connected GPUs (H100/A100) | D2D copy is ~2–5× faster than PCIe; gather cost is low |

### Does not win

| Condition | Why |
|-----------|-----|
| Small batches (`B < 4096` at dim 32, f32) | Copy + launch overhead > compute time |
| `dim` ≥ 4096 (memory-bound regime) | Each thread already touches `dim` coefficients; the kernel is memory-bound, and splitting does not reduce per-device memory traffic proportionally |
| Single-GPU machine | `visible_devices()` returns `[0]`; the helper falls back to a single device with extra copy overhead |
| PCIe 3.0 GPUs with tight VRAM | D2D copy bandwidth is the bottleneck; use NVLink or a larger single GPU |

---

## Worked example

```python
import numpy as np
import cupy as cp
import CayleyKernel as ck

# --- Setup: 2 GPUs, 65536 independent 32-dim float32 products ---
batch = 65_536
dim   = 32
rng   = np.random.default_rng(42)

a = cp.asarray(rng.normal(size=(batch, dim)).astype(np.float32))
b = cp.asarray(rng.normal(size=(batch, dim)).astype(np.float32))

# Verify device visibility
print("Visible devices:", ck.visible_devices())  # [0, 1]

# Single-GPU baseline
import time
cp.cuda.runtime.deviceSynchronize()
t0 = time.perf_counter()
c_single = ck.multiply(a, b)
cp.cuda.runtime.deviceSynchronize()
t_single = time.perf_counter() - t0
print(f"Single GPU: {t_single*1e3:.2f} ms")

# Multi-GPU
t0 = time.perf_counter()
c_multi = ck.split_batch(a, b)          # uses all visible devices
cp.cuda.runtime.deviceSynchronize()
t_multi = time.perf_counter() - t0
print(f"Multi GPU:  {t_multi*1e3:.2f} ms")
print(f"Speedup:    {t_single/t_multi:.2f}x")

# Results are identical (within float32 tolerance)
np.testing.assert_allclose(cp.asnumpy(c_multi), cp.asnumpy(c_single),
                           rtol=1e-6, atol=1e-6)
```

Expected: on a 2-GPU NVLink machine, `t_multi ≈ t_single / 1.7–1.9`
(after subtracting the copy overhead). The exact speedup depends on
NVLink bandwidth, kernel duration, and whether the two GPUs are
interleaved or striped (see the topology note in CONTRIBUTING.md).

---

## Design notes

* **No lock-step synchronisation.** The CPU queues work on device 0,
  immediately moves to device 1, and only calls
  `cp.cuda.runtime.deviceSynchronize()` *after* all devices have been
  launched. The kernels on different devices run in parallel.

* **One `cp.cuda.Stream` per device.** A non-blocking stream is created
  for each device inside `split_batch`; it is used for the H2D/D2D
  copies. The kernel itself is launched on the device's default stream
  (the ctypes path in `_launch` does not take a stream argument), but
  because the copy has already been synchronised, the kernel reads
  valid data.

* **Original batch order is preserved.** `np.array_split` produces
  contiguous, roughly-equal chunks. `gather_into` writes each chunk
  to the correct slice of the caller-side output array.

* **No new dependencies.** The helper uses only `cupy.cuda.Device`,
  `cupy.cuda.Stream`, `cp.cuda.runtime.getDeviceCount`, and
  `cp.cuda.runtime.deviceSynchronize` — all part of the standard
  CuPy runtime.
