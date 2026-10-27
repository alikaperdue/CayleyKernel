# Article 617-0702/2024: finite-field Cayley–Dickson elements

Saba Zahid, Muazzam Ali, M Usman Hashmi, Abdul Manan, and Affan Ahmad, *Graphical and Computational Analysis of Cayley-Dickson Algebra Elements over Z_p*, Journal of Computing & Biomedical Informatics, Volume 07 Issue 02 (2024). The supplied PDF prints DOI `10.56979/702/2024` and article ID `617-0702/2024`.

Reviewed source: `617.pdf`, eight pages. All pages' extracted text was reviewed. The equations, graphs and tables on PDF pages 2–6 were checked visually; Table 1 was transcribed from an enlarged rendering. Page references below are PDF page numbers, since the article does not print ordinary page numbers.

**Suite: 27 tests. Local result: 23 CPU tests passed. Four CUDA tests are written for your machine and were not selected locally.** No `multiply()` call or CUDA-device probe was made here. Normal pytest runs the GPU tests automatically when the CUDA prerequisites are present.

The results confirm the printed octonion idempotent and odd-prime nilpotent counts in the tested cases. Other assertions need corrections: Theorem 5 omits a factor of `p^2`, Corollary 6 and Theorem 7 have incorrect counts already in dimensions 4 and 8, and a unit/zero-divisor dichotomy cannot be extended to sedenions. See [FINDINGS.md](FINDINGS.md) for exact evidence and the distinction between correct results and faulty proof steps.

## Files and execution

- [All paper-ordered tests](../../../tests/papers/journal_617_0702_2024/test_sections.py).
- [Finite-field arithmetic, explicit isomorphisms and independent counting](../../../tests/papers/journal_617_0702_2024/finite.py).
- [CPU/CUDA report runner](../../../tests/papers/journal_617_0702_2024/run.py) and [SVG chart generator](../../../tests/papers/journal_617_0702_2024/figures.py).
- [Recorded results and provenance](results/results.json), [complete CPU log](results/test-output.txt), [exact certificates](results/certificates.json), and [graph comparison data](results/figure-data.csv).
- Recomputed [Figure 1](results/figure1.svg), [Figure 2](results/figure2.svg), and [Figure 3 comparison](results/figure3-comparison.svg).

Your usual command runs all 27 tests, including the GPU tests when CUDA is available:

```bash
pytest -v tests/papers/journal_617_0702_2024/test_sections.py
```

To execute only the four GPU tests:

```bash
pytest -v tests/papers/journal_617_0702_2024/test_sections.py::GPUVerification
```

To save a report including your GPU execution:

```bash
python3 -m tests.papers.journal_617_0702_2024.run --cuda \
  --paper /path/to/617.pdf \
  --output /tmp/cayley-paper-617-0702-2024/results-cuda
```

The local CPU-only report is reproduced with:

```bash
python3 -m tests.papers.journal_617_0702_2024.run \
  --paper /path/to/papers/617.pdf \
  --output /tmp/cayley-paper-617-0702-2024/results
```

Only this report runner's default is CPU-only; pytest has no environment-variable opt-in. The GPU tests skip for missing CuPy, compiled library, or device. The tests are marked to avoid the repository's global GPU fixtures suppressing the CPU cases. CPU execution requires NumPy and the local CayleyKernel module; pytest is optional because the suite also supports unittest.

## Models, definitions and exact arithmetic

`Z_p` is interpreted as the prime field `F_p`. Standard Cayley–Dickson multiplication uses all doubling parameters `-1`. The split model changes the **last** parameter to `+1`. In dimension 8 their norms are respectively

```
N(x) = x0^2+x1^2+x2^2+x3^2+x4^2+x5^2+x6^2+x7^2
Ns(x)= x0^2+x1^2+x2^2+x3^2-x4^2-x5^2-x6^2-x7^2.
```

Every result is reduced modulo p. Tuple position i means mathematical basis element `e_i`. `basis()` uses signed one-based labels, so the adapter sends `i+1` and converts its result back. `twist(i,j)` uses mathematical zero-based indices.

The standard model uses the public CPU routines `basis()` and `twist()`. An independently recursive parameterized doubling oracle checks their convention and implements split signatures. CayleyKernel's CUDA `multiply()` has a fixed signature: the code never pretends it accepts a split-signature argument. One GPU test instead transports split inputs through an explicitly verified finite-field algebra isomorphism, multiplies in the standard algebra, then transports the result back.

All CPU counts and certificates are exact. Complete product enumeration covers **795,784 elements** across eight standard/split cases: dimensions 4 and 8 over `F3` and `F5`. Characteristic-two, Zorn-case and ordinary-matrix controls are additional computations. Products are expanded from actual signed basis products; a separately implemented convolution of square-residue distributions supplies independent counts. For larger dimensions, the convolution counts all coefficient assignments without allocating the exponentially large element set.

The conventions matter:

- Idempotents include `0` and `1`.
- Nilpotents include `0`; nonzero nilpotents in the odd-characteristic quadratic models have square zero.
- A zero divisor follows the paper's stated definition: both the element and its annihilated partner must be nonzero. Norm-zero counts **including zero** are reported separately.
- A unit witness here satisfies both `xy=1` and `yx=1`, which is stronger than the paper's one-sided “or” phrasing. Such witnesses can still be zero divisors in a nonassociative algebra.
- The intersection in Theorem 5 uses the same eight coordinates for the two displayed norm forms. An isomorphism between two algebras does not by itself specify an intersection of their subsets.

## Tests in the order of the paper

The exact executable method names are listed below. Counterexample tests pass when the discrepancy is reproduced; they do not certify the corresponding printed assertion.

### 1. Introduction, tables, Zorn model and definitions (pp. 1–3)

1. **`test_01_01_intro_complex_plus_sign_and_split_quaternion_relations`** checks two introductory sign problems at `p=3,5,7`. The printed complex-pair product has `ac+bd`, making the imaginary generator square +1 rather than -1. The printed split-quaternion relations `i^2=1`, `j^2=k^2=-1`, `ij=k`, with anticommutation, contradict associativity: `(ij)^2=-i^2 j^2=+1`.
2. **`test_01_02_basis_twist_and_parameterized_doubling`** checks all **5,460 basis pairs** in dimensions 2, 4, 8, 16, 32 and 64 against independent recursive doubling, including all four signed input-label choices. This verifies the model used before any count is compared.
3. **`test_01_03_printed_octonion_table_and_split_table_norm_conflict`** transcribes and verifies **all 64 cells of Table 1**: they agree exactly with CayleyKernel. Four selected, clearly readable cells of Table 2 do not satisfy split-octonion norm composition: the printed `e1*e2=e4` has `N(e1)N(e2)=1` but `N(e4)=-1`. Table 2 is not silently used as a valid split algebra. Its full table is not transcribed or classified.
4. **`test_01_04_zorn_product_determinant_and_split_isomorphism`** implements the Zorn multiplication with the signs printed on p. 3, verifies a complete bilinear basis map from the valid split Cayley–Dickson model at `p=3,5,7,11`, and checks inverse coordinate conversion and equality of the two norms. This connects the matrix notation to an actual algebra, rather than ordinary matrix multiplication.
5. **`test_01_05_standard_and_split_octonions_are_isomorphic_over_odd_fields`** constructs explicit maps for all eleven plotted primes from 3 through 37. It finds a quaternion u with `N(u)=-1` and uses `(a,b) -> (a,u*b)` to map the standard octonions to the split model. Every basis-pair product, inverse map and sampled norm is checked. The maps are exported. This verifies the equality of element-type counts in the two models for these fields, without assuming a real-algebra isomorphism.
6. **`test_01_06_norm_composition_coefficients_for_composition_dimensions`** checks every coefficient in the multiplication-matrix similitude identity `L_x^T D L_x = N(x)D` at `p=3,5`, dimensions 4 and 8, for both signatures. This certifies composition and the invertibility of `L_x` for **arbitrary coefficients** when `N(x)!=0` in these cases; it is not just random norm testing. It also justifies the exact norm-zero/zero-divisor classification used in those complete enumerations.

The historical and physical narrative is not tested as history or physics. The real quaternion assertion “division algebra ... not a division ring” is internally inconsistent; it is not a definition used by the code. The valid model and the paper's problematic displayed relations are kept separate.

### 2. Methodology (p. 3)

7. **`test_02_01_full_enumeration_method_and_quadratic_identity`** enumerates every element in the eight finite algebras described above. It computes `x*x` and `x*conjugate(x)` from basis multiplication, verifies the quadratic identity for every element, counts idempotents, square-zero elements and norm-zero elements, and compares those totals with independent square-residue convolution. Unit and nonzero-zero-divisor counts are derived only in composition dimensions, with the norm classification separately certified by test 6.
8. **`test_02_02_quadratic_identity_coefficient_certificate_through_dimension64`** checks the quadratic square identity on every basis element and sum of two distinct basis elements, for both signatures at dimensions 4,8,16,32,64 over F3. These checks cover the coefficients of the square map, explaining why idempotent and nilpotent counting can continue even where norm composition fails.

The supplied article contains no executable MATLAB source or complete algorithm listing, despite promising methods/pseudocode in its abstract. This package supplies new reproducible implementations; it does not claim to reproduce an unavailable original program.

### 3. Results and Discussions (pp. 4–6)

9. **`test_03_01_theorem1_octonion_idempotent_counts`** compares complete enumeration over F3/F5 for standard and split octonions with `p^6+p^3+2`. All four comparisons agree: 758 and 15,752 respectively.
10. **`test_03_02_theorem1_zorn_cases_and_missing_factors`** independently enumerates all Zorn elements over F3 and separates Cases I–IV of the proof. Correct counts are `4,52,52,650`, totaling 758. Printed Cases II and III each omit a factor of 2; their printed sum would be 706. A nonzero/nonzero-vector idempotent at the allegedly excluded endpoint `a=0` or `a=1` is exported. Thus the final theorem count agrees while the printed case argument needs repair.
11. **`test_03_03_corollary2_idempotents_in_higher_dimensions`** independently counts the quadratic conditions for `p=3,5,7,11`, dimensions 4–64, and both signatures. All 40 cases agree with `p^(d-2)+p^(d/2-1)+2`. This uses the quadratic identity, not a fabricated extension of the 3-dimensional Zorn cross product to arbitrary vector dimension.
12. **`test_03_04_figure1_formula_data`** recomputes every plotted prime's idempotent count by residue convolution and compares it with the printed formula. The CSV and regenerated Figure 1 contain those exact data. The chart has eleven prime inputs; it is not an independently measured experiment.
13. **`test_03_05_theorem3_nilpotents_and_nilpotency_index`** compares complete square-zero counts over F3/F5 with `p^6` for both signatures. It checks trace/norm conditions and powers of deterministic general elements, including nonnilpotent controls. Zero is included, and counts excluding zero are also recorded.
14. **`test_03_06_ordinary_matrix_nilpotent_probability_is_not_zorn_proof`** enumerates ordinary 2-by-2 scalar matrices over F3/F5/F7: there are `p^4` matrices and `p^2` nilpotents. This is different from the `p^8` Zorn space. A concrete octonion associator distinguishes Zorn/octonion multiplication from ordinary associative matrix multiplication. Theorem 3's final `p^6` count agrees, but citing an ordinary-matrix probability does not establish its Zorn conclusion.
15. **`test_03_07_corollary4_odd_prime_nilpotent_counts_and_characteristic2`** counts the higher-dimensional nilpotent conditions in the same 40 odd-prime/signature cases and agrees with `p^(d-2)`. It also exhaustively tests the prime 2, which Corollary 4's wording does not exclude: dimensions 4 and 8 have 8 and 128 square-zero elements, not the printed 4 and 64. The odd-prime hypothesis is necessary.
16. **`test_03_08_figure2_formula_data`** independently recomputes `p^6` for the eleven graph primes; the verified data match Figure 2's formula.
17. **`test_03_09_theorem5_common_nilpotents_exhaustive_coordinates`** squares every octonion coordinate vector in both valid signatures at p=3 and p=5 and intersects the actual zero-square masks. Counts are **297** and **3,625**, not the printed 33 and 145. It saves a common nilpotent with a nonzero first ternary block, illustrating the degrees of freedom dropped by the proof.
18. **`test_03_10_theorem5_factorization_and_figure3_data`** counts the independent ternary and quaternary norm-zero blocks at all eleven primes. Their counts are `p^2` and `p^3+p^2-p`; their product is the corrected coordinate intersection. Figure 3's printed series and the corrected series are exported and plotted together.
19. **`test_03_11_zero_divisor_definition_and_norm_zero_witnesses`** constructs exact nonzero conjugate partners and complete modular annihilator bases for 24 selected norm-zero elements across F3/F5, dimensions 4/8, and both signatures. Every kernel vector is checked by actual multiplication. The zero element is explicitly separated from the paper's nonzero-zero-divisor definition.
20. **`test_03_12_corollary6_printed_sign_and_zero_convention`** compares complete enumerations with the printed formula and the corrected composition-dimension counts. Including zero, the norm-zero count is `p^(d-1)+p^(d/2)-p^(d/2-1)`; nonzero zero divisors number one fewer. The printed plus sign before the last term matches neither convention.
21. **`test_03_13_theorem7_unit_counts_and_actual_two_sided_inverses`** compares complete finite-field counts with the printed unit formula in dimensions 4 and 8, and checks explicit two-sided inverses `conjugate(x)/N(x)` for general elements with nonzero norm. The correct composition-dimension unit count is `p^d-p^(d-1)-p^(d/2)+p^(d/2-1)`.
22. **`test_03_14_higher_dimensions_unit_and_zero_divisor_overlap`** exports sedenion witnesses at `p=3,5,7,11` that have nonzero norm and a verified two-sided inverse **and** annihilate a nonzero partner. Full annihilator bases and an associativity failure are checked. This disproves a general use of “units are the complement of zero divisors” beyond composition dimensions; it does not purport to enumerate every sedenion unit or zero divisor.

### 4. Conclusions (p. 7)

23. **`test_04_01_finite_field_signs_do_not_define_causal_types`** shows that the same finite-field element has integer representatives of opposite signs. Thus `N(x)>0` and `N(x)<0` do not define the paper's time-/space-like distinction over Fp. The valid real split norm is tested separately on examples with positive, negative and zero norm. Isotropy `N(x)=0` remains a meaningful condition over finite fields.

The data-availability statement and references on pp. 7–8 are recorded as source context; datasets were not requested from the authors and external references were not independently audited.

### CUDA checks ready for your normal pytest run

24. **`test_05_01_multiply_modp_basis_and_batched_products`** sends scaled basis batches and general coefficient batches through the real `multiply()` API, reduces the output modulo `p=3,5,7`, and compares every coefficient with independent recursive products in dimensions 4,8,16.
25. **`test_05_02_multiply_full_enumeration_mod3_octonion_squares`** sends all 6,561 F3 octonions through GPU squaring, compares every result with recursive multiplication, and independently obtains the 758 idempotents and 729 nilpotents.
26. **`test_05_03_multiply_split_products_via_verified_field_isomorphism`** converts split-octonion inputs to standard-octonion coordinates using the verified finite-field inverse map, calls `multiply()`, reduces modulo p, transports back, and compares against independent split products at p=3,5,7. This exercises split algebra through a supported standard-signature GPU computation.
27. **`test_05_04_multiply_sedenion_unit_and_zero_divisor_certificate`** batches the nonzero zero-product and both inverse products of the same sedenion at p=3,5,7. It checks the overlap with nonzero control outputs, not just a mask of vanishing products.

GPU tests use signed int64 coefficients with small bounds; reduction modulo p follows each integer product. No large-integer overflow or unsigned wraparound is part of the finite-field model. They are written but unexecuted locally.

## Mathematical interpretation and unimplemented parts

Every standard/split model satisfies `x^2-T(x)x+N(x)1=0`, where `T(x)=2*x0`. For odd p, a nontrivial idempotent therefore has `T=1,N=0`; a nonzero nilpotent has `T=N=0` and square zero. The subalgebra generated by one element is associative: its products reduce using the quadratic equation. These facts justify the counting conditions independently of the paper's problematic matrix arguments.

The norm-zero/unit classification is more restrictive. The nonsingular composition norm gives `L_x^T D L_x=N(x)D` in dimensions 4 and 8, so nonzero norm yields an invertible multiplication operator and a two-sided inverse; norm zero yields a nonzero conjugate annihilator. For dimension 16 and above the similitude identity fails, and the explicit overlap witnesses show why that argument cannot be reused.

| Requested coverage | What is implemented / what remains outside this package |
| --- | --- |
| Every numbered result | Theorem 1, Corollary 2, Theorem 3, Corollary 4, Theorem 5, Corollary 6, Theorem 7 each has explicit comparisons and documented outcomes. |
| All elements at small primes | Complete actual-product enumeration for dimensions 4/8 over F3/F5 in both models; extra F2 controls. |
| Larger idempotent/nilpotent counts | Exact quadratic-form convolution for listed primes through dimension 64, with structural checks. No exhaustive allocation of these huge algebras. |
| All higher-dimensional units/zero divisors | Not enumerated. Counterexamples refute the partition used by the paper; correcting one sign does not repair the classification in all dimensions. |
| Literal Table 2 as a valid split algebra | Not adopted: selected printed cells contradict composition. The valid split model is specified explicitly. |
| All real causal types from finite-field counts | Not meaningful without changing the field and specifying the real form. Separate real norm examples are provided. |
| Original MATLAB code and pseudocode | Not present in the supplied PDF; this package supplies its own implementation and CSV/SVG results. |
| The cited literature and historical claims | Not independently reviewed here. The supplied paper and its definitions are the verification target. |

The recorded JSON includes the PDF SHA256, all new Python source hashes, CayleyKernel module hash, seed, git HEAD, runtime and complete observations. Certificates contain field isomorphisms, annihilator bases, literal table entries and exact counterexamples so the findings can be reviewed independently of the graphs or pass count.

## Coefficient-type coverage

`GPUVerification.test_05_05_half_exact_modp_products` adds float16 device products for 32 seeded pairs in each p=3,5,7 and dimension 4,8,16 combination, checking output dtype and every residue against independent recursive finite-field multiplication. Integer outputs have magnitude at most 576 and are exactly representable in half. Larger-prime counting, rank, inverses and existing int64 GPU checks keep their original arithmetic. Native complex coefficients are not substituted for Fp.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **23 tests: 23 passed, 0 skipped, 0 failures, and 0 errors**. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
