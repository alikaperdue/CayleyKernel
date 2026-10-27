# Alternating roots of polynomials over Cayley-Dickson algebras

Adam Chapman and Ilan Levin, arXiv:2306.15889v4, 22 October 2023,
Communications in Mathematics 32 (2024), 63-70, eight PDF pages.

Alternating evaluation replaces powers by alternating conjugate factors:
the displayed reduction is sum a_(2i) N^i + sum (a_(2i+1) lambda) N^i,
where N=Norm(lambda). This gives a scalar polynomial for possible root
norms in a composition division algebra. A second construction, the
"alternating friend", turns these into regular roots in the next doubling.

```bash
python3 -m unittest tests.papers.desktop_batch.test_2306_15889v4 -v
```

Code: `tests/papers/desktop_batch/test_2306_15889v4.py`; results and the
exact rank counterexample: `results/`. CPU arithmetic is exact except the
explicit numerical root-solver test. NumPy is optional for that test.

## Every test in paper order

| Section | Test suffix | Scope |
| --- | --- | --- |
| 1 introduction | None | Motivation; no distinct numerical assertion. |
| 2 definitions | `02_quadratic_identity_and_power_associativity` | Levels 1-7, eight seeded dense integer vectors each: x^2-Tr(x)x+Norm(x)=0 and cubic left/right powers agree. Cubic checks do not alone prove full power associativity. |
| 3 Claims 3.1-3.2 / Proposition 3.3 | `03_1_central_norm_polynomial_and_constructed_roots` | Dimensions 2,4,8, twelve seeded degree-five polynomials with constructed roots: full coefficient convolution of conjugate-polynomial times polynomial is central; each constructed root norm satisfies the scalar equation. |
| 3 Examples 3.4,3.7 / Remark 3.5 | `03_2_examples_3_4_3_5_3_7_and_spherical_case` | Exact norm polynomial 2-3N+N^2; verify isolated roots j and i+j for x^2+ix-1-ij. x^2+1 has no nonnegative possible norm; x^2-1 has tested basis and rational norm-one spherical roots. |
| 3 Remark 3.6, scope check | `03_3_remark_3_6_nondivision_counterexample` | Exact sedenion linear polynomial a1*x-target: the coefficient multiplication matrix has rank 12, augmented rank 13. Thus no root exists despite odd degree. The printed remark needs a division-algebra restriction; this does not contradict the division-only Corollary 4.3. |
| 3.1 Octave implementation | `03_4_python_port_of_octave_root_algorithm` | Python numerical scalar-root method, isolated-root inversion and spherical-family detection reproduce the two isolated roots, sphere N=1, and no-root x^2+1 examples. Residual checks and repeated-root deduplication included; explicitly limited to composition algebras. |
| 4 Theorem 4.1 / Corollary 4.2 | `04_theorem_4_1_and_root_correspondence` | Component levels 1-7 (doubled dimensions 4-256), eight seeded examples each, degree-five polynomials: friend(lambda*v)=conjugate(f_alt(lambda)); constructed roots map to zero and outputs lie in the lower algebra. Conjugations were checked visually on PDF p.7. |
| 5 Remark 5.1 / Theorem 5.2 | `05_theorem_5_2_requires_conjugate_lambda_for_g` | Component levels 2-5, eight seeded degree-four examples each: exact full decomposition h(lambda*v)=pair(conj(f_alt(lambda)),conj(g_alt(conj(lambda)))). Construct common roots with the required **conjugated lambda for g**, and verify h vanishes. |
| Extension | `06_extension_sparse_correspondence_through_level_4096` | Exact sparse friend correspondence at component levels 8,20,128,1024,4096 with high-bit coefficient and argument supports. No division or norm multiplicativity is assumed. |
| Device replay | `07_gpu_root_and_friend_products` | Direct `multiply()` replays three dimension-eight products from the example and friend evaluation; explicit GPU prerequisite skip. |

## Finding: do not extend the division-only root solver blindly

The proof of Proposition 3.3 needs multiplicativity and anisotropy of the
norm. The module's sedenions have positive norm but zero divisors; positivity
alone does not give that proof. Remark 3.6 is worded for arbitrary real An,
but a degree-one polynomial can already be unsolvable in the sedenions.
The saved non-surjective multiplication matrix gives a finite exact witness.
The higher-level friend correspondence is nevertheless valid for the tested
families because it uses the doubling identities rather than division.

## Not coded / limits

No arbitrary-field or split-parameter theorem verification is included;
friend() accepts a scalar parameter for coefficient construction but only
gamma=-1 is replayed through the module. General non-real norm roots,
arbitrary spherical geometry, multiplicity classification, robust
ill-conditioned numerical root finding and a production polynomial solver
are not supplied. The numerical port is tested on explicit quaternion
examples, not asserted complete for sedenions or arbitrary data.

The norm-reduced alternating evaluation follows the grouping displayed in
Section 3; the unparenthesized alternating-factor string must not be read as
arbitrary reassociation in a nonalternative algebra. No infinite-dimensional
proof, arbitrary odd-degree root existence, or general field norm-image
computation is claimed. The high-index extension uses sparse integers and
does not allocate vectors of dimension 2^4096. All module products use the
signed one-based `basis()` adapter; `multiply()` is used only in device checks.

## Coefficient-type coverage

The existing root/friend product device fixtures also run typed replays. These check bilinear products and phase scaling, without asserting that real composition-division root-existence hypotheses or exact polynomial-root identities survive arbitrary complexification or repeated half rounding.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **9 tests: 9 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:36.086232+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
