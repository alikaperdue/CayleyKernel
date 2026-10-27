# All Hurwitz algebras from 3D geometric algebras

Corradetti, Clawson and Irwin, arXiv:2311.02269v1, 3 November 2023.
The paper builds complex/quaternion subalgebras inside a three-generator
Clifford algebra, then changes the product to obtain octonions and split
octonions. A Clifford product is associative; the changed product is the
one compared to CayleyKernel. These are deliberately separate operations.

```bash
python3 -m unittest tests.papers.desktop_batch.test_2311_02269v1 -v
```

Code: `tests/papers/desktop_batch/test_2311_02269v1.py`; saved signature
tables, counterexample, outcome/source/PDF provenance in `results/`.

## Every coded test in section order

| Section | Test suffix | Coverage |
| --- | --- | --- |
| 1 | None | Motivation, no separate history/priority test. |
| 2 Hurwitz examples | `02_hurwitz_subalgebras_via_module_map` | All 64 octonion basis products compare the module to the paper's bullet product via an explicitly constructed signed blade map. Includes lower complex/quaternion spans. |
| 3 Clifford algebra / involutions | `03_geometric_product_and_involutions` | Independent Clifford blade-swap implementation for all eight ordered metric signatures; all 512 triples verify associativity and all basis pairs verify reverse/Clifford anti-involutions and grade automorphism. |
| 4.1 | `04_1_even_and_pseudoscalar_subalgebras` | All signatures: even-span closure, pseudoscalar square and commutation with every blade. Checks the algebraic ingredients of the tensor-product decomposition, not an arbitrary tensor factorization algorithm. |
| 4.2 (4.20)-(4.28),(4.39)-(4.40) | `04_2_norm_composition_and_all_eight_signature_isomorphisms` | Both bullet variants, eight signatures: all 64 products vs an independent parameterized CD oracle, with explicit basis images and parameters (-lambda1*lambda2,-lambda2*lambda3,-variant*lambda1). Norm composition on all 36x36 basis/pair-sum combinations, plus scalar conjugate products. Polarization covers coefficients of the biquadratic composition identity in each fixed algebra. |
| 4 preliminary (4.5), regression | `04_3_equation_4_5_odd_part_counterexample` | For x=1+e1+e12 in G(3,0), odd parts of xx-dagger and x-dagger-x differ. The claimed equality is not generally valid. Both sides saved. |
| 4 Table 2 | `04_4_table_2_norm_signatures` | Four displayed representative signatures, both variants: exactly the two stated division-octonion cases have positive definite norm; the remaining cases have signature (4,4). Norm-sign lists saved in blade-bit order. |
| 5 extension | `05_extension_standard_subalgebra_embedding_through_level_4096` | All octonion basis pairs embedded at indices shifted by n-3 bits, n=8,20,128,1024,4096; module twist agrees with the standard dimension-eight table. This is a sparse subalgebra check, not a claim that the full higher algebra is a composition algebra. |
| Device supplement | `06_gpu_octonion_products` | Direct `multiply()` for all octonion ordered basis products; explicit prerequisite skip. |

## Interpretation / limitations

The core modified-product construction and Table 2 classification agree with
the finite computations. Split-signature comparisons use a reference oracle;
CayleyKernel itself implements the all-negative standard CD sequence only.
The module comparison uses G(3,0), bullet+, and an explicit coordinate map.
No change to the module's public arithmetic was needed.

The paper's (4.18) should be a direct-sum decomposition, not a disjoint union
of the even and odd subspaces: a general mixed multivector belongs to neither
subspace alone. Likewise, unit octonions form a loop rather than an
associative group. These wording issues are not assumptions in the code.
The equation (4.5) regression is a distinct finite discrepancy; it does not
refute the modified-product theorem.

No separate exhaustive transcription of Section 2's cyclic Fano labels,
Table 3 tensor-coordinate involutions, (4.33)'s long expanded polynomial or
every proof intermediate is implemented. Norm expectations are evaluated
independently using the compact quadratic form (4.27)/(4.40), not the long
expansion. The displayed scalar/odd norm identities require careful grade
selection; the code follows the explicit product definition. The full grade
inversion is tested in the modified algebra, not assumed to be an
anti-automorphism of ordinary Clifford multiplication.

Section 5's proposed para-Hurwitz/symmetric-composition extensions are not
constructed. No arbitrary-field classification, formal Hurwitz theorem or
unbounded embedding proof is claimed. Higher ambient CD levels contain the
tested octonion subalgebra but also have zero divisors and nonmultiplicative
norm. High-bit tests do not allocate dense vectors of dimension 2^4096.

## Coefficient-type coverage

All standard octonion device products also run typed replays. `test_07_gpu_complex_split_signatures` adds all eight +/- CD signatures in dimension eight, in both complex precisions, using the generator-phase embedding with all basis products and eight dense pairs per signature. This tests their transported CD multiplication, not a Clifford geometric product or a real positive norm.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **8 tests: 8 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:36.761993+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
