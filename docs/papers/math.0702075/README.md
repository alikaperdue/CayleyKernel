# Verification of Biss et al., arXiv:math/0702075v2

Daniel K. Biss, J. Daniel Christensen, Daniel Dugger and Daniel C. Isaksen, **Large Annihilators in Cayley-Dickson Algebras II**, version 2, 22 February 2007, 22 PDF pages. Printed and PDF page numbers agree. The directory uses `math.0702075` to distinguish this legacy arXiv identifier.

Input: `/home/alika/Desktop/CayleyKernel/0702075v2.pdf`.
SHA-256: `8a13b3b2ae6a2c0cf1b6f086400d0b50eeebcdde9be861e68012224a51c95f8c`.
The complete extracted text was reviewed. Pages 6–7, 9, 13–16, 18 and 20–21 were additionally checked as rendered pages, especially the projections, division notation, normalization and `c/4` stability threshold. The PDF is unchanged and not copied into git. This review concerns the supplied version, without a later-version or cited-literature audit.

## Recorded outcome

On 23 October 2026 the suite completed **37 tests: 34 passed, 3 skipped**, with no failures or errors. All CPU algebra and linear algebra use exact integers and rational fractions. The three direct GPU tests skipped because CuPy is unavailable. No floating-point rank, tolerance-based nullity, or GPU execution is claimed.

The central multiplication formula, Theorem 4.5, passes every bilinear basis pair in the relevant subspace of **dimensions 16,32,64,128**, certifying it for arbitrary real vectors in those fixed subspaces. The main annihilator dimension formulas, D-locus distinction, full Theorem 6.12 decomposition, Section 7 algebraic criteria and Section 8 constructions pass on the specified finite families. Exact annihilators were computed through **dimension 256**, including a maximal example with **nullity 228**.

There are also literal-statement discrepancies: missing restrictions in the projection lemma/corollary, omitted nonzero conditions, and the wrong annihilator in Lemma 6.9's uniqueness condition. Normalizations also need attention in Section 7 and Corollary 8.13. The tests preserve these as counterexamples or scope checks. **A passing counterexample test reproduces a printed failure; it does not verify that printed statement.** See [FINDINGS.md](FINDINGS.md).

The global topology claims, diffeomorphisms and arbitrary-dimension stability classification are not established by finite computation. Their algebraic ingredients, sample fibers, explicit witnesses and parameter arithmetic are tested separately and labelled accordingly.

## What the paper means

An annihilator Ann(a) is the space of all partners x giving `a*x=0`. Computing it is a linear-algebra problem: form the matrix of left multiplication by a and find its kernel. A large kernel means a has many independent annihilating partners.

The paper introduces special coordinates `{a,b}` for a large subspace in the next doubled algebra. In those coordinates, multiplication becomes much simpler. Usually the annihilator dimension is the sum of the two component dimensions. On a special set called the **D-locus**, it gains four extra dimensions. The D-locus depends both on the components and on their annihilator spaces, not just on ordinary orthogonality of the two vectors.

The last section uses these formulas to study elements whose annihilators are near the largest possible size. Its “stability” concerns whether a geometric space becomes two copies of a previous space after doubling. This is different from the subnorm stability in the earlier 1505.01670 paper.

## Code, data and reproduction

- [algebra.py](../../../tests/papers/arxiv_math_0702075/algebra.py): sparse module-based products, independent doubling, rational row reduction, kernels, modular rank certificates, projections, splitting, D-locus, partial division and predicted subspaces.
- [test_sections.py](../../../tests/papers/arxiv_math_0702075/test_sections.py): all section-ordered tests and optional GPU replays.
- [run.py](../../../tests/papers/arxiv_math_0702075/run.py): standalone runner, provenance and certificate export.
- [results/results.json](results/results.json): test outcomes, exact dimension censuses, witnesses, constructions, source/PDF hashes, runtime details and timings.
- [results/test-output.txt](results/test-output.txt): complete section-ordered test log.
- [results/annihilators.json](results/annihilators.json): **27 named examples** with their exact elements, entire annihilator bases and independent modular rank checks, including examples both inside and outside the D-locus and stable-boundary examples.

From the repository root:

```bash
python3 -m tests.papers.arxiv_math_0702075.run
```

Regenerate all saved files:

```bash
python3 -m tests.papers.arxiv_math_0702075.run \
  --paper /home/alika/Desktop/CayleyKernel/0702075v2.pdf \
  --output docs/papers/math.0702075/results
```

Python and NumPy suffice. No pytest, symbolic algebra system, GAP, or GPU runtime is needed for the CPU suite. With pytest installed:

```bash
python3 -m pytest -v tests/papers/arxiv_math_0702075/test_sections.py
```

Both classes carry the repository's `cpu` marker to bypass its general GPU prerequisite fixtures. The GPU class checks CuPy, device and library prerequisites independently. Once they succeed, multiplication errors and mismatches fail rather than being replaced by CPU products. The standalone runner runs CPU tests in paper order, followed by GPU checks.

Existing uncommitted changes mean git HEAD alone is not the tested source identity. Hashes of all paper Python files and `CayleyKernel/__init__.py` are recorded in both result datasets. No module arithmetic or older paper package was modified.

## Exact arithmetic and the square-root normalization

The basis convention is the module's standard convention, matching the paper's doubling formula. Mathematical indices are zero-based. `twist(i,j)` takes those indices; `basis(i+1,j+1)` returns a signed one-based label. `mul()` extends that public product bilinearly. The independent recursive oracle calls neither sign routine, so agreement is stronger than comparing `basis()` to the `twist()` it uses internally.

Write d=2^n. In A_n:

- `i_n=e_(d/2)`;
- C_n is the span of indices 0,d/2;
- H_n is the span of indices 0,d/4,d/2,3d/4;
- the Hermitian product is the C_n projection of `a*conj(b)`.

The paper defines `{a,b}=(a+b, i_n*(-a+b))/sqrt(2)`. To avoid irrational coefficients in nullspace calculations, the code uses

```text
B(a,b) = (a+b, i_n*(-a+b)) = sqrt(2)*{a,b}_paper.
```

This is an **explicit normalization convention**, not a change to the paper. Nonzero scalar multiplication does not change an annihilator or D-locus conditions. All scalar factors in product and metric formulas are restored exactly. For example, Theorem 4.5 is checked as `B(a,b)*B(x,y)=2*(its three rational terms)`, and `N(B(a,b))=2*(N(a)+N(b))`.

In Theorem 6.12, the third C-linear generator becomes

```text
B(b/a,-a/b) + 2*(0,i_n),
```

up to a common nonzero scalar. The relative factor **2** is essential. The code checks the full resulting subspace and its C-orthogonality.

All matrices are built from actual products with basis vectors. Sparse rational Gauss-Jordan gives kernel bases. Each saved certificate additionally checks:

1. every kernel vector multiplies to exactly zero;
2. the operator rank modulo the prime 1000003 is `dimension-nullity`;
3. the kernel basis has modular rank equal to its length.

A modular operator rank is a lower bound for the rational rank. The independently verified, independent zero-product vectors give the matching upper bound. Together these certify the exact nullity without a numerical singular-value cutoff. Section 2 also checks selected kernel products with the separate doubling oracle.

## Complete inventory in paper order

Identifiers below are methods in `PaperVerification`, with the common `test_` prefix omitted. Every coded CPU test is listed, including regressions and scope checks. Levels mean A_n; algebra dimension is 2^n.

### 1. Introduction and statement of results, pages 1–4

| Test | Computation and interpretation |
| --- | --- |
| `01_intro_splitting_annihilator_dimension_and_d_locus_examples` | Representative pairs in source levels 3,4,5: compute both component kernels, D membership independently from Definition 6.1, and the doubled kernel. Check the two dimension options of Theorem 1.3 and the D/non-D distinction of Theorem 1.5. Exact inputs and outcomes are saved. Proposition 1.2 and the stability summary are checked in their detailed later sections. |

The introduction's topology motivation and comparison with the earlier DDD paper are not computational claims tested here. The existence of a two-sided conjugate inverse does not imply cancellation or absence of zero divisors in these nonassociative algebras.

### 2. Basic algebra and projections, pages 4–7

| Test | Computation and finite scope |
| --- | --- |
| `02_01_all_basis_signs_and_signed_labels_against_independent_doubling` | Every ordered basis pair in dimensions 2,4,8,16,32,64, with all four input-sign combinations. Sample dense products in levels 3,4,5 agree with the independent oracle; conjugate norms and product reversal are exact. |
| `02_02_complex_action_real_hermitian_and_lemma_2_3` | Four sampled full pairs per source level 3,4,5: C-linear scalar action (Lemma 2.2), the real dot product, Hermitian symmetry/positivity, and `dot(x,x*y)=0` for imaginary y (Lemma 2.3). |
| `02_03_lemmas_2_4_to_2_6_conjugate_linearity_and_orthogonality` | Left conjugate-linearity and anti-Hermitian identity on sampled C-perpendicular a and full x,y; the restricted two-factor scalar formula on C-orthogonal basis directions; same-vector formula and norm scaling with explicit nonreal C scalars. Source levels 3,4,5. |
| `02_04_h_quaternion_subalgebra_and_perpendicular_coordinates` | All 64 H-basis triples at levels 3–6 are associative, and products remain in H. Explicit B coordinates avoid the four H positions. Closure and associativity extend by multilinearity to the real quaternion spans at these levels. |
| `02_05_exact_annihilator_kernels_modular_rank_image_and_bounds` | Six fixtures per level 3–7: basis, recursively maximal examples, dense C-perpendicular input and maximal examples plus a real or distinguished imaginary component. Verify exact kernels, left/right zero products, C-perpendicular kernels, independent modular ranks, image orthogonality, divisibility of nullity by four, and the nonzero-element upper bound. Six selected maximal-source certificates through A8 are exported across this and Section 8 tests. |
| `02_06_projection_lemmas_with_required_restriction` | Check Lemma 2.15 with a in C-perp, its scalar/perpendicular product components, Corollaries 2.16–2.17, and the **whole** preimage of C under left multiplication against `span_C(a)+Ann(a)` (restricted Corollary 2.18). Source levels 3,4,5, exact sample pairs and subspaces. |
| `02_07_lemma_2_15_and_corollary_2_18_printed_generality_counterexample` | At a=b=1+e1 in A3, b is its own projection onto span_C(a), but a*b=2*e1 is outside C3. Reproduces failure of the two statements as printed for arbitrary a,b. The a-in-C-perp restriction used later avoids this example. |
| `02_08_zero_exception_in_theorem_2_11` | In levels 3,4,5, `Ann(0)=A_n` violates the printed upper bound if “any annihilator” includes zero. The bound is applied only to nonzero inputs in all substantive checks. |

The code uses ordinary real Euclidean geometry for real orthogonality, and the paper's C projection for C-orthogonality. They are checked separately, not treated as interchangeable.

### 3. Splitting notation, pages 7–8

| Test | Computation and finite scope |
| --- | --- |
| `03_01_split_complex_scalar_action_and_metric` | Three pairs per source level 3–6: B and its inverse recover the exact inputs, the complex scalar action of Lemma 3.4 has conjugation in the correct slot, and the metric normalization of Lemma 3.5 has the required factor two in B coordinates. |

### 4. Multiplication formulas and constructions, pages 8–12

| Test | Computation and finite scope |
| --- | --- |
| `04_01_proposition_4_1_orthogonal_special_cases` | Explicit complex-line combinations in two C-orthogonal directions at levels 3,4,5. Verify the orthogonality hypotheses, Proposition 4.1 / Proposition 1.2 and cross-slot zero products. |
| `04_02_lemma_4_3_corollary_4_4_and_J_action` | Three sampled a per source level 3,4,5, with dependent b=(1+i_n)*a. Check the commutator involving J=(0,i_n), all four dependent-product formulas and both left/right J actions from Lemma 4.7. |
| `04_03_theorem_4_5_exhaustive_bilinear_basis_and_orthogonal_terms` | All 144,784,3600,15376 ordered basis pairs of H-perp in target dimensions 16,32,64,128. Compare direct products to the general three-term formula and verify orthogonality of its pieces. Since both sides are bilinear in the two combined inputs, this certifies the formula for **all real inputs in those fixed subspaces**. It is not an arbitrary-n proof. |
| `04_04_inner_products_lemmas_4_9_and_4_10` | Exact Hermitian and real metric formulas on several distinct sampled quadruples in source levels 3,4,5, including the conjugated first slot and the B normalization factor. |
| `04_05_iterated_nonquaternion_four_dimensional_subalgebras` | Start with two orthonormal octonion basis directions and iterate the construction in Section 4.11. Levels 3–7 realize lambda=1,2,3,4,5. Exact unit lengths, C-orthogonality, four-dimensional independence, closure and the multiplication relations in the basis (1,a,b,ab) are checked. The normalized ab/sqrt(lambda) gives the paper's displayed unit-basis rules. |
| `04_06_mutually_annihilating_families_lemma_4_12_corollary_4_13` | Build both inductive families in levels 3–7, of sizes 1,2,4,8,16. Every distinct within-family product is zero, the families are independent, and every cross-family Hermitian product vanishes. Gives the stated degenerate subalgebra dimensions 2,3,5,9,17. |

Question 4.14, asking whether still larger degenerate subalgebras exist, is not solved by constructing the advertised lower-bound examples.

### 5. Annihilation in H-perp, pages 12–13

| Test | Computation and finite scope |
| --- | --- |
| `05_01_proposition_5_1_annihilation_equations_positive_and_negative` | Representative pairs in source levels 3,4,5, with positive kernel-derived targets, zero targets and negative controls. Check zero multiplication iff all four displayed projection equations vanish. Both positive and negative counts are saved. |
| `05_02_proposition_5_2_entire_parameterized_intersection` | Build the parameter space from independently computed Ann(a), Ann(b), alpha,beta. Impose the four real equations of conditions (3)–(4), map the resulting entire nullspace into B coordinates, and compare its span to the directly computed `Ann(B(a,b)) intersect H-perp`. Levels 3,4,5. |
| `05_03_theorem_5_10_exhaustive_basis_pairs_and_sedenion_sample_census` | Exhaust the 36 C-perpendicular basis pairs in A3 and 196 in A4. Then fix one sedenion zero divisor and test all 84 ordered signed octonion-basis-frame zero divisors as its second component. Direct ranks agree with the D-locus dimension formula on all **316** pairs. |
| `05_04_proposition_5_11_entire_annihilators_and_zero_exception` | For source levels 3–6, basis/maximal nonzero a: compare the **entire** predicted left- and right-slot annihilator spans with direct kernels and dimensions. At a=0, record the exceptional actual dimension 16 versus the printed expression 12 in A4. |

The 316-pair census gives:

| Components | Cases | Direct target nullity |
| --- | --- | --- |
| A3 basis pair, not C-orthogonal | 12 | 0 |
| A3 basis pair, C-orthogonal | 24 | 4 |
| A4 basis pair, not C-orthogonal | 28 | 0 |
| A4 basis pair, C-orthogonal | 168 | 4 |
| Fixed A4 zero divisor with another in the D-locus | 6 | 12 |
| Fixed A4 zero divisor with another outside the D-locus | 78 | 8 |

This exhausts the named finite families, not arbitrary real component pairs.

### 6. The D-locus, pages 14–16

| Test | Computation and finite scope |
| --- | --- |
| `06_01_d_locus_vanishing_constraints_image_characterization` | The fourth parameter condition vanishes identically iff the independently checked D conditions hold, using all parameter-basis directions. Check real/C orthogonality against annihilators and solve both image equations on D examples. Also retain a nonzero a*y with zero C projection, exposing the missing projection in Lemma 6.3's proof. Levels 3,4,5. |
| `06_02_lemma_6_4_theorem_6_7_off_d_and_one_zero_exception` | For 19 non-D pairs with nonzero components, direct kernels have the sum dimension and lie in H-perp. Save nontrivial off-D kernel certificates. Separately, B(e1,0) is nonzero and non-D but has dimension 4, not the 8 claimed by a literal unrestricted Lemma 6.4. Theorem 6.7 explicitly requires both components nonzero and passes its tested cases. |
| `06_03_definition_6_10_division_and_lemma_6_9_wrong_annihilator` | Solve ax=b with the Definition 6.10 source-kernel orthogonality, including the zero/zero case and rejection of an impossible target. Give two distinct solutions satisfying Lemma 6.9's **printed** condition x perpendicular to Ann(b), proving its uniqueness claim fails as worded. |
| `06_04_lemma_6_11_and_theorem_6_12_full_orthogonal_decomposition` | Seven nonzero D pairs in source levels 3,4,5: check quotient location/orthogonality; construct all three predicted C-subspaces with correct relative factors; verify their C-orthogonality and equality with the whole direct annihilator. Exact kernels are exported. |

The finite off-D bases provide concrete answers to Question 6.8 for these individual examples; they are not a new general closed formula for all off-D annihilators.

### 7. The D-locus in A5, pages 16–19

| Test | Computation and finite scope |
| --- | --- |
| `07_01_lemmas_7_1_7_2_and_proposition_7_3` | Two octonion first components, all pairs from zero plus six C-perpendicular basis units, restricted to nonzero sedenion zero divisors: 28 positive and 44 negative Proposition 7.3 cases. Also test Lemma 7.2's implication against all 84 signed basis-frame zero divisors for each first component. Confirm relevant image orthogonality; reproduce the degenerate zero-target exception to the literal Lemma 7.1. |
| `07_02_annihilator_eigenspace_geometry_and_missing_norm_factor` | Twelve basis-frame zero divisors a: construct the quaternion block, four-dimensional kernel, complementary four-plane and six-plane. Check direct sums and orthogonality. On the geometric four-plane the exact equation is `a*(a*x)=-2*N(a)*x`; the printed `-2*x` needs N(a)=1. Tested a have N(a)=2, so the distinction is explicit. |
| `07_03_corollary_7_4_six_plane_membership_against_D_definition` | Six selected first zero divisors against all 84 second ones: 504 exact cases, 36 in D and 468 out. The six-plane criterion agrees with the original D definition using full annihilator bases. |
| `07_04_theorem_7_5_local_six_plane_and_bundle_map_checks` | Six sample base points: the second-half sign flip identifies the two four-dimensional fibers; coefficient Gram identities show all combinations in the six-plane have imaginary, equal-length, orthogonal halves. Dense six-plane samples are actual zero divisors in D with target nullity 12. Check the Stiefel-frame normalization: two unit halves have total squared length 2, requiring division by sqrt(2). No global bundle/diffeomorphism certification is claimed. |

Remark 7.6's S5-bundle and G2 statements involve global smooth topology. The computed fiber rank six and algebraic identification are ingredients, not proofs of those statements.

### 8. Stability, pages 19–21

| Test | Computation and finite scope |
| --- | --- |
| `08_01_theorem_8_1_large_annihilators_in_H_perp` | Constructed maximal and two-component large-annihilator elements in levels 4–8 satisfy the H-perp conclusion whenever nullity is at least half the ambient dimension. Includes and exports the A8 maximal example, nullity 228. |
| `08_02_proposition_8_2_full_outside_H_annihilators` | For target levels 4–8, use a=e1; verify its alternative-element hypothesis on every source basis vector. The proposed full kernel, computed independently from C-orthogonality constraints, multiplies to zero and equals the direct kernel. Nullities 4,12,28,60,124 occur outside H-perp, four below half-dimension. |
| `08_03_propositions_8_6_8_7_maximal_branches_and_stable_boundary` | Source levels 4–7, at c=4*(n-4): both left/right lifts meet the exact next-level threshold, preserve the appropriate normalized ray, and split into precisely one nonzero component. The eight target kernels are saved. These are examples on the stable boundary, not an exhaustive description of T_n^c. |
| `08_04_lemmas_8_8_8_9_maximal_D_pairs_and_sharp_bounds` | Source levels 3–7: recursively construct D pairs whose component kernels are maximal. Their doubled nullities are 4,12,36,92,212. For source n>=4, these lie exactly four below Proposition 8.6's one-component forcing bound. |
| `08_05_proposition_8_11_theorem_8_12_threshold_and_nonstable_witnesses` | Check the arithmetic partition `n>=c/4+4` against both stability branches on all 1,981 allowed integer parameter cases in levels 4–12. For levels 5,6,7 construct actual two-component witnesses at the first nonstable c=8,12,16, with target nullities 36,92,212. Normalize the rays implicitly; never mistake raw B coefficients for unit points. |
| `08_06_corollaries_8_13_8_14_dimension_examples_and_unit_scope` | Actual left/right lifts give A5 nullity 16 and A6 nullity 40 from A5 nullity 12. Exact scalar multiples preserve kernels but are unbounded in norm, demonstrating why Corollary 8.13's compact Stiefel interpretation requires a unit-length locus. Global diffeomorphisms are not tested. |

The extracted fraction `c/4` can look like `4c` in plain text. The rendered pages confirm **c divided by 4**, and the code uses the exact equivalent inequality `4*n>=c+16`.

### Optional GPU replays

| Test in `GPUVerification` | Direct `multiply()` calls |
| --- | --- |
| `test_02_multiply_basis_and_dense_independent_oracle` | All basis pairs and four dense pairs in dimensions 2–64, float64, compared exactly to independent doubling. |
| `test_04_multiply_full_splitting_products` | Sample general B products at source levels 3,4,5 compared exactly to the three-term formula. |
| `test_05_08_multiply_exact_annihilator_basis_certificates` | Batch each maximal-source annihilator basis in dimensions 16,32,64,128 and check zero products with absolute tolerance 1e-12 after converting rational coefficients to float64. |

All three skipped in this run. CPU sparse products explicitly use `basis()`; they are not a hidden fallback for `multiply()`.

## Representative numerical data, computed exactly

For maximal-source examples:

| Algebra | Dimension | Maximum formula | Computed nullity |
| --- | --- | --- | --- |
| A3 | 8 | 0 | 0 |
| A4 | 16 | 4 | 4 |
| A5 | 32 | 16 | 16 |
| A6 | 64 | 44 | 44 |
| A7 | 128 | 104 | 104 |
| A8 | 256 | 228 | 228 |

Stable-boundary lift examples:

| Source level | c | Source nullity | Target nullity, either slot |
| --- | --- | --- | --- |
| 4 | 0 | 4 | 16 |
| 5 | 4 | 12 | 40 |
| 6 | 8 | 36 | 96 |
| 7 | 12 | 92 | 216 |

All entries come from exact multiplication-matrix kernels, not floating-point eigenvalues. Unit-length versions are obtained by dividing each recorded element by the square root of its saved squared norm. This does not alter its kernel.

## Limits and parts not computationally established

| Paper content | Limitation or next work |
| --- | --- |
| Results for every n and every real vector | Basis exhaustions prove the named bilinear formulas only in the listed fixed dimensions; general dimension theorems and sampled nonlinear identities are not universally proved by this suite. |
| Lemmas 2.2–2.6 on every input | Explicit exact tests are included, but no all-variable tensor certificate for every lemma in every dimension is constructed. The independent full product convention and main bilinear formula receive stronger exhaustive coverage. |
| Global classification of all zero divisors or D-locus points | The named basis-frame families, representative dense inputs, local six-planes and recursive examples are computed. They do not exhaust the continuum of real points. |
| Theorem 7.5 and Remark 7.6 diffeomorphisms, G2, bundle topology | No charts, transition-function verification, smooth inverse maps, homotopy computation or global bundle certificate is provided. Fiber algebra and dimensions are tested separately. |
| Theorem 8.12 as an iff classification of whole geometric spaces | Arithmetic thresholds and explicit stable/nonstable witnesses are tested. Exhausting all points of T_n^c, proving equality of spaces or a diffeomorphism for arbitrary n requires the mathematical arguments. |
| Unit length and irrational coefficients | Exact ranks are computed on rational nonzero representatives of the same scalar rays. Normalization is given algebraically, not approximated numerically and used to guess rank. |
| Question 4.14: larger degenerate subalgebras | The constructed lower bound through dimension 128 is checked; no upper bound or maximality result is claimed. |
| Question 6.8: general off-D annihilator formula | Specific full nullspaces are saved. This is finite solution data, not a closed classification valid for all inputs. |
| Question 8.3: all elements with half-minus-four annihilator dimension | The explicit Proposition 8.2 family is checked through dimension 256; the full classification remains unimplemented. |
| Entire annihilator varieties, smoothness and dimensions of parameter spaces | No symbolic real-algebraic elimination or stratification is performed beyond the displayed finite linear and Gram identities. |
| Other signatures, alternate doublings or arbitrary base fields | CayleyKernel's ordinary real sequence is used. No split/general-parameter kernel is introduced. |
| Historical claims and referenced DDD/CD/Moreno results | The needed finite consequences are checked where specified; the cited papers are not independently reviewed here. |
| GPU execution/performance | CuPy is unavailable. No device result or benchmark is claimed. |

A narrow `.gitignore` exception makes this paper's documentation and results visible to git. Existing unrelated changes and previous paper packages were preserved.

## Coefficient-type coverage

Basis/dense products, full splitting-formula product fixtures and the maximal-annihilator certificate products also run typed device replays. Phase-scaled complex fixtures preserve zero products. No half numerical rank or complex extension of real orthogonal decomposition, annihilator-dimension classification or topology is inferred.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.
