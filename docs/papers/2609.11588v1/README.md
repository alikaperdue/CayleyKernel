# Quasilinear multiplication in the real Cayley-Dickson tower

Harrison Lemley, arXiv:2609.11588v1, 10 September 2026.
This paper directly concerns the module's multiplication. It also cites
CayleyKernel. It reduces total arithmetic work using a two-call alternating
product recursion over one fixed central complex coefficient field.

The independent implementation is [quasilinear_cd.py](../../../scripts/quasilinear_cd.py).
It is an experimental NumPy CPU companion. It does not replace or modify
the module. Its central coefficient i commutes with every algebra component;
it is not an extra Cayley-Dickson generator. Complex pairings are bilinear,
with no coefficient conjugation. Three-real-product complex multiplication
implements (3.5). This distinction is essential to the algorithm.

```bash
python3 -m unittest tests.papers.desktop_batch.test_2609_11588v1 -v
```

Direct use from the repository root:

```python
from scripts.quasilinear_cd import multiply
result = multiply(a, b)  # real arrays, coefficient axis last; batch broadcasting
```

Tests: `tests/papers/desktop_batch/test_2609_11588v1.py`. Exact arithmetic
counts, dense extension measurements, environment and source/PDF hashes
are in `results/`. NumPy is required for this companion.

## Every test in section order

| Section / statement | Test suffix | What is checked |
| --- | --- | --- |
| 1 introduction | Later count/correctness checks | No priority claim or literature chronology verified. |
| 2 Lemma 2.3 / Proposition 2.4 | `02_scalar_imaginary_decomposition` | Levels 1-7, three seeded dense pure pairs per level: real alternating recursion equals half the difference of two independent four-call products; scalar output is zero. |
| 3 Theorem 3.3 / fixed coefficient field | `03_1_complex_two_call_recursion_is_bilinear_not_hermitian` | Levels 1-6, four seeded complex-coefficient pure pairs each: recursive cross product vs independent doubling commutator. Check complex linearity explicitly. The CD involution leaves central complex coefficients fixed. |
| 3 Corollary 3.4 / Algorithm 3.5 / Proposition 3.6 | `03_2_all_basis_and_dense_products_against_module_and_doubling` | Every ordered basis pair in dimensions 1-64, plus 16 dense integer pairs per dimension: QL vs independent four-call recursion and module signed basis table. No QL output is used as its own expectation. |
| 4 Theorem 4.2 / Proposition 4.4 / Table 1 | `04_operation_counts_and_table_1` | All ten printed M/A/T rows reproduced from formulas. At levels 1-12, instrument recursive node visits and count six complex/real bilinear products per lower imaginary coordinate, then reconstruction products; real-product totals agree with the bound. Formula tables extended through level 20. Check total-operation crossover vs direct at N>=16 and CC at N>=32. |
| 5 numerical protocol | `05_broadcasting_and_float_error_protocol` | Seeded normalized floating inputs at levels 3,5,8, batch shape (2,3), against independent recursion; finite results satisfy the paper's 128*N*epsilon norm-scaled bound (operand norms are one). |
| 6 extension beyond reported dimensions | `06_dense_extension_through_level_20` | Actual dense vectors at levels 10,12,14,16,18,20, up to **1,048,576 coefficients**. Dense bipolar output coordinates 0,1,N//3,N-1 checked by independent full signed-XOR sums using `twist()`. A dense-by-four-term product at every level checks **every output coordinate** against module signs. Exact equality is required for these small-integer/dyadic calculations. CPU elapsed times saved as observations. |
| Device comparison | `07_gpu_direct_comparison` | When GPU prerequisites exist, direct `CayleyKernel.multiply()` vs independent QL for eight dense integer pairs per dimension 8,16,32,256,1024. The saved JSON snapshot records this test as passed with CuPy available; current execution still depends on GPU prerequisites. |

## Interpretation

The tested quasilinear recursion agrees with the standard module convention.
The paper's operation reduction is meaningful independently of a parallel
GPU implementation: counting total arithmetic work and measuring elapsed
GPU time answer different questions. The saved Python CPU timings are not
a replication of the paper's optimized C benchmarks, and no CPU/GPU speedup
ratio is claimed here. The user's supplied timing table is not treated as
an independently reproduced benchmark.

The 20-level extension allocates real dense input/output arrays; it is not
merely a high-index sparse test. Nevertheless, for two fully dense operands
at these larger levels only four output coordinates have an independent
quadratic-sum oracle. Full output coverage there comes from a separate
dense-by-sparse case. Finite agreement does not prove the universal theorem
or a numerical stability bound for arbitrary magnitudes.

## What is not implemented or established

- No fetch/build/run of fastCD or its archived source revisions, CC numerical
  implementation, AUTO selector, native SIMD kernel or 10,000-pair C timing
  campaign. The local supplied PDF is the source reviewed.
- The multiplication counter records the displayed bilinear schedule, excluding
  known-zero coordinates. Addition/halving totals are formula/recurrence
  checks against the printed table, not an instruction-level trace of NumPy.
  Built-in array allocation, Python overhead and fixed-coefficient operations
  are not claimed to have the same runtime as those scalar counts.
- The sequential recursion uses only linear-size temporary arrays by its
  structure, but peak resident memory is not measured and the optimality of
  the storage/tensor-rank bounds is not proved. No lower bound is supplied.
- No general forward-error proof, all high-dimension output coordinates for
  fully dense pairs, extreme dynamic-range/overflow study, integer dtype API,
  or complex-coefficient full public multiplication API is delivered.
- Sections 6-7's future-work suggestions, program organization, AI disclosures
  and responsibility statements add no separate arithmetic test.

The implementation supports real float64 inputs and matching last-axis
power-of-two dimensions with batch broadcasting. It is kept in `scripts/`
to make review and further performance experiments straightforward.

## Coefficient-type coverage

Every real QL/device comparison also runs typed replays, including half products of the quantized bounded fixtures. `test_08_gpu_complex_quasilinear_extension` adds four seeded complex pairs per dimension 8,16,32 in each complex precision, comparing the GPU with four real QL calls combined by bilinearity and independently with direct doubling. The Python QL API remains real-only and its operation counts are not complex-input counts.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **7 tests: 7 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:33:07.922813+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
