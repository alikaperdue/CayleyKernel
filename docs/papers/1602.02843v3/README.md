# Periodicity of the Cayley-Dickson twists

John W. Bales, arXiv:1602.02843v3, 14 May 2016, eight PDF pages.
This paper reduces large basis-sign queries to smaller indices by identifying
repeating parts of the twist table. It uses the shuffle basis. Your module
uses a block basis; fixed-width bit reversal converts the coordinates for
P3 transpose. Applying the printed periodicities directly to unconverted
module indices would test a different statement.

## Reproduce and inspect results

```bash
python3 -m pytest tests/papers/desktop_batch/test_1602_02843v3.py -v
```

See `results/` for per-test outcomes and source/PDF hashes. All implemented
tests live in `tests/papers/desktop_batch/test_1602_02843v3.py`.

## Every test, in section order

| Section | Test suffix | Finite coverage |
| --- | --- | --- |
| 1 introduction | None | Motivation, no separate numerical assertion. |
| 2 background / Table 1 | `02_table_1_transposes_and_module_conversion` | All 64x64 pairs, all four transpose relations; all module signs checked via fixed-width bit reversal against independent P3-transpose shuffle doubling. The signed one-based `basis()` output support is converted and checked with zero-based `twist()`. |
| 3-5 twists and trees | `03_04_05_tree_navigation` | Every 128x128 pair, P0-P3: independently expanded Table 1 doubling vs Figures 3-4 tree transitions. Both printed worked products are tested. |
| 6 Theorem 1 | `06_theorem_1` | Identity, imaginary squares and anticommutation, all eight products, dimension 128. |
| 6 Theorems 2-3 | `06_theorems_2_3_periodicity` | N=1-6, all p,q in the specified blocks, eight products; T2 k=0,1,2,7,31; T3 k=0,1,3,16. No out-of-hypothesis periodicity is assumed. |
| 7 Theorems 4-5 | `07_theorems_4_5_and_high_level_module` | Seed 160202843, 24 samples per bit width 8,20,32,128,1024,4096, all eight products. For T4 use p of bit length N and q>=2^N. For T5 generate equal-bit-length distinct p,q and choose N=bit_length(p XOR q). Check modular reductions; replay both equalities through converted module basis signs. |
| 8 conclusion / GPU supplement | `08_gpu_standard_products` | Direct `multiply()` on all 64 ordered octonion basis products, expected from exact sparse module multiplication; prerequisites explicitly skipped if absent. |

## Findings and boundaries

- Section 5's worked example writes 25 and 17 as 1101 and 1001, which are
13 and 9. Their XOR still equals 8; the correct five-bit representations
11001 and 10001 give the stated product too. No multiplication discrepancy
is inferred from the binary transcription error.
- Section 2 says the transpose relation holds for basis vectors but need not
hold for arbitrary x,y. For bilinear products on the same finite coordinate
space, equality on every basis pair extends to arbitrary x,y. This sentence
therefore needs clarification. Tests verify the finite basis relation and
do not infer failure for sums.
- Table 1 is implemented independently for all eight products by reducing
shuffle indices one bit at a time. Tree expectations do not call that code.
Figures 1-4 are checked as transition rules, not pixel images. Infinite
Hilbert-space extensions and convergence are not modeled; all elements have
finite support.
- T5 excludes p=q via its hypothesis on positive p XOR q. Tests preserve that
condition; the identity/diagonal cases are in T1.
- High levels exercise sparse Python integer indices, including direct
module signs. They allocate no dense vectors at dimensions 2^4096 and do
not validate vector-product performance or prove an infinite theorem.
- General isomorphisms among all eight algebras, literature/history claims,
signal processing, compression and cryptographic usefulness are not coded.
An interesting next experiment is to count how much these reductions save
for actual query distributions; no such speedup is claimed here.

## Later-version qualification

The supplied 1707.07318v4 companion distinguishes P1/P2 and their transposes
as pseudo products, with zero divisors at dimension eight. Its finite
alternativity and zero-product checks support that distinction. Thus the
earlier paper's general claim that all eight give isomorphic Cayley-Dickson
algebras must not be inferred from the sign/periodicity checks here. See
[the later paper report](../1707.07318v4/README.md).

## Coefficient-type coverage

All standard octonion device basis products also run typed replays. Bit reversal and high-index periodicity are integer-index statements and do not need new coefficient types.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **6 tests: 6 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:25.837498+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
