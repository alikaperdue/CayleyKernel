# A calibration in R16 and Federer's product question

Roger Zuest, arXiv:2503.12402v1, 16 March 2025.
This paper has a direct module connection: its Cayley form and Clifford
spinor construction use octonion right multiplication. The new 8-form in
R16 is built independently by exterior algebra; it is **not** an ordinary
sedenion product. The paper explicitly notes that obstruction in Remark 2.4.

```bash
python3 -m unittest tests.papers.desktop_batch.test_2503_12402v1 -v
```

Code: `tests/papers/desktop_batch/test_2503_12402v1.py`. Exact scalar
invariants and sampled frame values are saved in `results/` with provenance.
The form arithmetic uses exact integers/fractions. Complex volume expansion
has integral real/imaginary coefficients; these are converted to real integer
forms before the rational exterior computations.

## Every coded test, in paper order

| Section / statements | Test suffix | Calculation |
| --- | --- | --- |
| 1 introduction | Later Sections 3-5 | Main results checked at their constructions. |
| 2.1 exterior algebra | `02_1_exterior_wedge_and_hodge` | Wedge anticommutation for all 16 coordinate pairs; Hodge-star squared for all middle-degree coordinate blades in R8/R16. |
| 2.2 Lemmas 2.2-2.3, (2.8) | `02_2_octonion_Q8_and_cayley_form` | Q8 on the ordered standard basis; all 70 four-coordinate combinations, each fully alternated over 24 permutations using module octonion products. Exact agreement with all 14 printed signed Cayley-form coefficients and squared norm 14. |
| 2.4 (2.12)-(2.15) | `02_4_octonionic_clifford_relations` | Octonionic right-product pinor actions check every Clifford square/anticommutator on all 16 basis pinors. Composition is operator composition with explicit order, not reassociated octonion multiplication. |
| 3 Theorem 1.1 / Proposition 1.3 / Corollary 1.4 | `03_main_form_exact_self_duality_norm_and_product_bound` | Build Phi from holomorphic volume and Kahler forms; exact self-duality, squared norm 294, Phi wedge Phi=294 volume. Diagonal-plane restriction of the two orthogonal copies gives 294/256=147/128>1, reproducing the algebraic counterexample bound. |
| 4 Theorem 1.5 / Proposition 4.1 | `04_calibrated_coordinate_planes_and_sampled_comass` | Special Lagrangian, complex and product-of-Cayley coordinate representatives give value 1. Sixty-four seeded orthonormal 8-frame samples give absolute value<=1; maximum saved. This is evidence, not a proof of global comass or the full classification. |
| 5 spinor product construction / Theorem 1.7 | `05_octonion_spinor_form_has_same_norm_and_self_duality` | Build all 12,870 degree-eight Clifford coefficients from sparse octonion-derived actions on s=1+ tensor 1+, paired with s+i+ tensor i+. Exactly 294 nonzero coefficients, squared norm 294 and self-duality. This is a second module-backed route to the invariant. |
| Device supplement | `06_gpu_cayley_basis_replay` | All octonion basis products through `multiply()`, exact sparse expectations; explicit prerequisite skip. |

## Analysis and limits

The two independent constructions agree on the key invariant 294. The exact
ratio 147/128 exceeds 1, supporting the stated orthogonal-product obstruction.
Random-frame tests do not establish a maximum; the all-plane comass bound
still relies on the paper's proof. The sparse spinor form is expressed in the
original octonion coordinates, while the exterior form uses adapted complex
coordinates. Their full coordinate-change equality is not implemented;
matching invariants alone does not prove they are the same orbit.

The normed-algebra background, general calibration/current theory, systolic
ratio optimality, analytic estimates in Section 3, normal-form necessity and
sufficiency for every calibrated plane in Section 4, and all spinor component
identifications in Section 5 are not exhaustively reproduced. In particular,
the code does not transcribe all of Lemmas 5.1-5.3 or construct every even
degree of the spinor form. It tests the central degree-eight invariant directly.

The diagonal-plane factor 2^-8 follows by pulling both 8-forms back through
v -> (v,v)/sqrt(2); its exact exterior value is computed without materializing
the much larger full 32-dimensional product form. No arbitrary-dimensional
calibration extension is claimed. A sedenion analogue cannot be justified by
norm multiplicativity, since sedenions have zero divisors. The useful higher
dimension here is the R16/R32 geometric construction, not a forced dense
higher-level CD benchmark. GPU execution is unavailable when prerequisites
are missing; the report marks that explicitly.

## Coefficient-type coverage

All octonion basis products in the calibration device replay also run typed checks. Real exterior forms, plane bounds, Clifford relations and calibration geometry are not inferred over complex coefficients or rounded half frames.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **7 tests: 7 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:37.860322+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
