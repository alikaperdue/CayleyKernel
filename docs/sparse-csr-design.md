# Sparse (CSR-style) Cayley–Dickson Product — Design Document

**Status:** experimental / research — implemented on this branch (Phase A design, Phase B Python API, Phase C CUDA kernel; ABI 5).
**Branch:** `feat/sparse-csr` · **Scope:** dense dimensions ≥ 256 (2⁸), sparse inputs, CUDA + Python API.

---

## 1. Motivation and the algebraic facts we can exploit

The dense kernel (`cuda/cuda_kernel.cu`) assigns one thread per output
component; each thread sums `dim` signed terms, so a single product costs
`O(dim²)` term evaluations. At `dim = 256` and above that is wasteful for
inputs that have only a few nonzero basis coefficients — which is exactly the
regime where zero-divisor structure dominates, and where the existing CPU
evidence lives.

Three facts make a sparse path both correct and cheap:

1. **Basis products are single basis vectors.** For zero-based indices
   `a, b < dim`, `e_a · e_b = sign(a,b) · e_{a ⊕ b}` with `sign(a,b) ∈ {±1}`.
   The sign is the O(1) closed form in `CayleyKernel/_cpu.py`
   (`_python_twist`, mirrored by `cayley_sign` in the CUDA source and by the
   Rust backend), so the sparse kernel can reuse the *same* evaluator — no
   tables, no recursion.
2. **Supports compose by XOR.** If `x` has support `S_x = {a₁,…,a_Ka}`
   (coefficients `va_j`) and `y` has support `S_y = {b₁,…,b_Kb}`, then
   `x·y` has support inside `{a⊕b : a∈S_x, b∈S_y}`, whose size is
   **≤ min(dim, K_a·K_b)**. (Not ≤ 2k in general: e.g. `{0,1} ⊕ {0,2} =
   {0,1,2,3}` for k = 2. The k²-style bound is the correct one and is the
   one used below.)
3. **Cancellations are common.** Two distinct input pairs can map to the same
   output index with opposite effective signs, summing to exactly zero. The
   CPU suite `tests/core/test_sparse_zd.py` (run via `make test-sparse-zd`)
   exhaustively verifies four-term products at level 4, embedded
   zero-divisor witnesses, and high-bit samples at arbitrary level —
   including the identity `expected == {}` exactly when the two canceling
   sign pairs satisfy the cancellation predicate. That file is the
   pre-existing CPU evidence that the support/XOR/cancellation model matches
   the real product, and it doubles as the correctness oracle for the
   sparse kernel (see §5).

## 2. CSR-style representation

Per vector we store the three arrays side-by-side, row-major contiguous
across the batch axis (batch = all leading axes flattened, same convention
as the dense `_nd` entry points):

```
struct SparseCSR {
    int64_t* support;   // K basis indices per vector, 0 <= support < dim
    int8_t*  signs;     // K coefficient signs (+1/-1) per vector
    T*       value;     // K magnitudes per vector (dtype T, same rules as dense)
    int64_t  K;         // max support size across the batch (K <= dim)
};
```

Memory per vector is `K·(8 + 1 + sizeof(T))` bytes plus `K` metadata, i.e.
`O(K)` instead of `O(dim)`. For `dim = 4096`, `K = 8`, `T = float32`:
208 B vs 16 KiB — a 77× working-set reduction, which is what makes large
batches of large dimensions fit in L2 where the dense path streams DRAM.

Support indices need not be sorted (the kernel does not assume order);
duplicates are treated as repeated basis terms (summed), matching the dense
semantics of coefficient addition.

## 3. Kernel algorithm (one warp per output basis index)

Grid: `dim` warps per product (dim is a power of two ≥ 256, so ≥ 8 warps;
launch config follows the existing `launch_threads` heuristics, block size
a multiple of 32). Warp `w` computes output basis index `i = w`.

Each warp:

1. Loads `K_a` support/sign/value triples of `x` and `K_b` of `y` into
   registers/`__shared__` (K ≤ 32 in the target regime; K > 32 falls back
   to a chunked loop).
2. Enumerates the `K_a·K_b` candidate pairs across the 32 lanes (lane ℓ
   owns pairs where `(pair_id) % 32 == ℓ`, looping while `pair_id += 32`).
3. Lane ℓ evaluates, for its pairs: `a⊕b == i` and, if so, the term
   `sign(a) · sign(b) · cayley_sign(a,b) · va · vb` using the **existing**
   `cayley_sign` device function (unchanged, same evaluator as the dense
   kernel).
4. Butterfly-shuffle reduction (`__shfl_down_sync`) sums the matching
   terms; lane 0 writes `out[i]`, all other outputs are left as zero
   (the out buffer is zeroed by a first pass or `cudaMemsetAsync` on the
   stream, exactly as an `out=` buffer is documented to be).

Complexity per output index: `O(K_a·K_b / 32 + log 32)` ≈
`O(K_a·K_b / log dim)` for the supported range, i.e. the
`O(K_a·K_b / log K)`-class bound stated in the plan, achieved with warp
shuffles and no atomics, no shared-memory cross-warp traffic, and no
serialization. Total per product: `O(dim·K_a·K_b/32)` vs dense
`O(dim²)`.

Why this shape rather than pair-scatter: it reuses the existing
"one output per thread/warp" launch geometry, keeps the zero-fill of
untouched output components trivial and race-free, and needs no atomics —
the correctness property we rely on most is *determinism*, since the
cancellation cases in §1.3 must cancel bit-for-bit in the same way as the
dense kernel's sequential sum.

## 4. When sparse wins: crossover and expected speedup

Model. Dense per product: `dim²` signed-term evaluations (each term:
multiply + sign + add/sub) plus `2·dim` coefficient loads. Sparse per
product: `dim·(K_a·K_b/32)` term evaluations (sign + compare + multiply
per pair) plus `2·K` loads and a `dim`-wide zero write. Letting α ≈ 2–4
capture the extra compare/XOR and shuffle cost per sparse pair and the
lower arithmetic intensity, the crossover in K (with `K_a = K_b = K`) is:

    K* ≈ sqrt( dim / α )

| dim | α=2 | α=4 |
|---|---|---|
| 256 | 11 | 8 |
| 1024 | 22 | 16 |
| 4096 | 45 | 32 |

So the whole `K ∈ {2, 4, 8, 16}` regime is at or below the crossover for
the dimensions in scope; only `K = 16` at `dim = 256` sits right at it.

Expected (modeled, not yet measured) speedup ≈ `(dim / K²) / α`, capped at
low end by kernel-launch + zero-fill overhead (~2×) and at the top by
memory/launch overhead (~20×). Representative table, `K_a = K_b = K`,
float32:

| dim \ K | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| 256 | 15–20× | 8–15× | 4–8× | 2–4× |
| 1024 | 20× (cap) | 10–20× | 5–10× | 3–6× |
| 4096 | 20× (cap) | 10–20× | 8–15× | 4–10× |

The table gives design-time estimates. Use `make benchmark-sparse` for
synchronized measurements of six float32 workloads, including dimension
32,768. It reports medians and an informational dense/sparse ratio.
Inputs are prepared in each API's native layout before timing, and the sparse
timing includes the work to produce sparse output.

## 5. Correctness guarantees

- **Single source of truth for signs:** the basis sign rule in
  `CayleyKernel/_cpu.py` (`_python_twist`, with the Rust backend as an
  equivalent accelerated implementation) is authoritative. The sparse
  kernel reuses `cayley_sign` in `cuda/cuda_kernel.cu` unchanged; it must
  **not** re-derive or re-table signs.
- **Bit-for-bit reproduction:** for each supported dtype, the sparse path
  must agree with the dense `mult_<type>` result on the same inputs.
  FP16/BF16 accumulate in float32 and round once, exactly like the dense
  `accum_traits`. Integer and bool dtypes accumulate in the dense path's
  accumulator type (int for bool, own type otherwise).
- `tests/core/test_sparse_zd.py` is the cancellation regression
  harness for exact cancellation: every case where it asserts
  `expected == {}` must also produce an all-zero output on the sparse GPU
  path. The known-witness checks in `tests/core/test_sparse_csr.py` use a
  CuPy device and run independently of the CPU suite's opt-in flag.
- **Dense kernels untouched:** no `mult_*` / `fma_mult_*` symbol, launch
  path, or ABI behavior is modified; the sparse path is additive (new
  symbols + new ABI version 5 if it ships, reported by `cayley_abi_version`).
- **No new third-party dependencies** (CuPy/NumPy already required for the
  dense path; no new Python packages).

## 6. Python API (shipped on this branch)

All symbols are exported from the package root and listed in `__all__`
(see `CayleyKernel/__init__.py`):

```python
from CayleyKernel import (
    SparseVector,            # dataclass: support [B,K] int64,
                             # signs [B,K] int8, values [B,K] T, K, dim
    to_sparse,               # cp.ndarray -> SparseVector
    from_sparse,             # SparseVector -> cp.ndarray (dense, (..., dim))
    sparse_multiply,         # (SparseVector, SparseVector) -> SparseVector
    is_sparse_beneficial,    # (dim, k_a, k_b) -> bool  (crossover rule, §4)
)
```

`SparseVector` is a plain dataclass holding three CuPy arrays (row-major,
contiguous across the batch axis, same convention as the dense `_nd` entry
points) plus the scalar fields `K` and `dim`. `sparse_multiply` dispatches
to the `sparse_mult_<suffix>` CUDA kernel (float16 / bfloat16 / float32 /
float64 / complex64 / complex128) via ctypes; if the kernel is unavailable
(e.g. ABI < 5, or a non-GPU environment) it transparently falls back to a
bit-for-bit CPU reference so the API remains usable in tests.

The experimental status is stated in the docstrings and in `README.md`.

## 7. Non-goals

- Sparse *output* (CSR output format) — dense output only, for now.
- `K > 32` fast path (falls back to chunked loop; not tuned).
- AMD/HIP port and BF16-specific tuning (follow NVIDIA, as with the dense
  kernel's policy for experimental features).
- Fused `sparse_multiply_add`.

## References

- `tests/core/test_sparse_zd.py` — CPU zero-divisor / cancellation evidence.
- `CayleyKernel/_cpu.py` — authoritative O(1) sign rule.
- `cuda/cuda_kernel.cu` — dense reference implementation and dtypes.
- `docs/testing.md` — benchmark protocol the measured table must follow.
