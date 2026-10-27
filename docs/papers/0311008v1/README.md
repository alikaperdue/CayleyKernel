# Differential characters on orbifolds and string connections I

Ernesto Lupercio and Bernardo Uribe, math/0311008v1, 15 PDF pages.

This is a geometry paper, not a paper about Cayley-Dickson multiplication.
The key idea is to describe geometric connections using their curvature and
holonomy (a phase picked up around a loop or surface). Complex phases live in
the first Cayley-Dickson algebra, so the module can check their multiplication.
The calculations below are finite models of ingredients, **not verification
of the main cohomology or string-connection existence theorems**.

## Reproduce

From the repository root:

```bash
python3 -m pytest tests/papers/desktop_batch/test_0311008v1.py -v
```

See `results/results.json` for source/PDF hashes, environment and individual
outcomes, and `results/test-output.txt` for the complete execution log.

## Tests in paper order

| Section / pages | Test suffix | Calculation and scope |
| --- | --- | --- |
| 1, pp.1-2 | `01_intro_character_additivity` | Nine rational phase pairs: addition in R/Z corresponds to complex multiplication; independent Python complex arithmetic supplies a second expectation. |
| 2.1, pp.3-7 | `02_1_differential_squared_and_character_boundary` | On the full simplex with six vertices, degrees 1-3, exact rational/integer cochains check delta squared zero, the middle component of d squared zero for d(c,h,w)=(delta c,w-c-delta h,delta w), the truncation boundary case, and the character boundary equation modulo integers. Discrete forms use the same simplicial complex; no smooth integration is asserted. |
| 2.2, pp.7-10 | `02_2_finite_group_cocycle` | Cyclic groups of orders 2,3,5,11; all triples verify a nonconstant normalized coboundary's 2-cocycle condition in R/Z. This is the group-cochain ingredient of (2.2.3), not sheaf hypercohomology. |
| 3, pp.11-14 | `03_string_connection_gluing_and_group_action` | All 16 products of fourth roots of unity check the group-action identity on p.13. Nine phase pairs check additive surface-integral gluing, orientation reversal, and a scalar Stokes compatibility model. The phase normalization is radians in (3.0.11), inspected visually. |
| Beyond paper | `04_extension_complex_planes_at_high_levels` | Complex phase multiplication in span(1,e_u) at doubling levels 4,8,16,32,128,1024,4096; u has a high bit and low bit set. These are exact sparse basis products plus floating trigonometric comparisons (absolute tolerance 1e-12), not dense vectors of dimension 2^4096. |
| Device replay | `05_gpu_phase_products` | Direct `multiply()` on two complex integer products compared with sparse expectations. Explicit skip when CuPy/device/library is unavailable. |

## Arithmetic and interpretation

The shared `common.py` uses signed one-based `basis()` (called
`basis_product` in the adapter), zero-based `twist()`, and sparse bilinearity.
The current module has no public `basis_product()` name. High levels describe
index bit widths; no claim about dense high-dimensional GPU speed is made.
CPU calculations require only the standard library. Pytest is optional.

## Not coded / further work

- Section 1 Theorems 1.0.1-1.0.2, Section 2 Lemmas 2.1.5, 2.2.2-2.2.3 and Theorem 2.2.4: arbitrary smooth orbifolds, equivariant differential forms, Borel/Stiefel limits, sheaf resolutions, Morita invariance, hypercohomology and canonical isomorphisms need a separate geometry implementation or formal proof. Finite cochains cannot establish these results.
- Section 3 Definitions 3.0.1-3.0.6 and Theorem 3.0.7: no Riemann-surface category, principal bundles, actual gerbe, loop groupoid, transgression or arbitrary surface integration is represented. The group action and Stokes tests are scalar consistency examples.
- A larger algebra does not replace U(1) in the theorem: unit sedenions do not form a group. The extension retains a single associative complex plane; it does not claim higher-algebra gerbes.
- The sentence on p.9 claiming a globally glued locally exact form is exact is not justified by local exactness alone. No verdict on the main theorem is inferred from that sentence.

All coded tests are listed above. A passing toy model is evidence about its
specified identity only. References and historical claims are not tested.

## Coefficient-type coverage

The existing two-coordinate phase fixtures also run typed device replays. `test_06_gpu_native_complex_holonomy` additionally represents U(1) phases directly as one complex coefficient in both precisions, checks broadcast products for four angles and multiplication by inverse phases. Differential cochain and geometric existence statements are unaffected.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **6 tests: 6 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:22.096076+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
