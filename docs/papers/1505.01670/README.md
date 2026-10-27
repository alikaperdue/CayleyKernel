# Verification of Goldberg and Laffey, arXiv:1505.01670v1

Moshe Goldberg and Thomas J. Laffey, **On the Radius in Cayley-Dickson Algebras**, arXiv version 1, 7 May 2015, 14 PDF pages. Printed page numbers match PDF page numbers. Page 14 contains author addresses; the references begin on page 13.

Input: `/home/alika/Desktop/CayleyKernel/1505.01670v1.pdf`.
SHA-256: `29c5f289cf0a9c952be9851dca28aa518d2f4e30a4dcf2bc957c354755982e73`.
The source PDF is unchanged and is not duplicated here. The extracted text was read in full. Pages 2–4, 7–10 and 12–13 were additionally inspected as rendered images to confirm formulas, conjugation, angle conventions and inequalities. This review concerns the supplied file, without a search for later corrections or an audit of its cited papers.

## Recorded outcome

The paper runner on 22 October 2026 completed **36 tests: 33 passed and 3 skipped**, with no failures or errors. The three skips are direct GPU tests because CuPy is unavailable in the selected Python. Loading the shared library alone does not establish a working CUDA runtime.

The main radius result and power-norm identity have successful finite checks, including exact arithmetic beyond the octonions. Some passing tests deliberately reproduce problems with later statements: the folded quaternion angle construction, omitted continuity in Corollary 4.2, and an included equality boundary in the “stable but not strongly stable” condition. A passing counterexample regression is **not** verification of the printed claim. [FINDINGS.md](FINDINGS.md) gives the derivations and separates exact algebra from mathematical extension arguments.

[results/results.json](results/results.json) preserves observations, every pass and skip, PDF and source hashes, the seed, Python/NumPy/platform, git HEAD, runtime availability and elapsed time. [results/test-output.txt](results/test-output.txt) is the complete section-ordered log. The source hashes matter because the checkout has pre-existing uncommitted work; git HEAD alone does not identify the tested source.

## What the paper means

The “radius” here comes from the roots of the smallest real polynomial that an element satisfies. The main result says it is simply the ordinary length of the element's coordinate vector. Although general sedenion products can have zero divisors and lose norm composition, powers of **one fixed element** retain their expected lengths: `|a^k|=|a|^k`.

A subnorm is a positive, absolutely homogeneous size function; it need not obey the triangle inequality. The paper asks whether taking kth roots of the sizes of powers recovers the radius, whether these sizes satisfy a uniform growth bound (“stability”), and when the power equation `f(a^k)=f(a)^k` holds exactly. Continuity makes several statements much easier. The pathological constructions deliberately abandon continuity.

All Cayley-Dickson algebras in this review, including C and H, are treated as **real** algebras. Homogeneity therefore includes every negative real scalar. A nonconstant angle-weighted function on C would not be homogeneous under arbitrary complex scalars.

## Files and commands

- [algebra.py](../../../tests/papers/arxiv_1505_01670/algebra.py): exact module-based products, independent doubling oracle, powers, polynomials, roots, subnorms, square roots and a restricted symbolic angle model.
- [test_sections.py](../../../tests/papers/arxiv_1505_01670/test_sections.py): all tests, named in paper order.
- [run.py](../../../tests/papers/arxiv_1505_01670/run.py): standalone unittest runner and reproducible report writer.

From the repository root:

```bash
python3 -m tests.papers.arxiv_1505_01670.run
```

To regenerate the saved record:

```bash
python3 -m tests.papers.arxiv_1505_01670.run \
  --paper /home/alika/Desktop/CayleyKernel/1505.01670v1.pdf \
  --output docs/papers/1505.01670/results
```

Python and NumPy suffice for the CPU suite; pytest is optional. With pytest installed:

```bash
python3 -m pytest -v tests/papers/arxiv_1505_01670/test_sections.py
```

Both test classes use the repository's `cpu` marker to bypass its automatic GPU prerequisite fixture. The GPU class checks its own CuPy/library/device prerequisites. After those succeed, multiplication errors or mismatches fail instead of being silently replaced by reference products. The standalone runner executes the CPU class in section order, then the three GPU replays.

## Arithmetic, conventions and evidence

The paper numbers coordinates from **1**. Thus its sedenion `e4+e11` is kernel `e3+e10` with zero-based indices. There is no basis permutation. `twist(i,j)` takes zero-based indices; `basis(i+1,j+1)` returns a signed one-based label. `mul()` converts that label and extends the actual public basis product bilinearly. `basis()` calls `twist()` internally, so their agreement alone is not independent evidence.

`recursive_sign()` separately expands the four cases of the paper's exact doubling definition `(a,b)(c,d)=(ac-conj(d)b, da+b*conj(c))`. It never calls either sign API. It supplies independent expectations for complete basis sweeps and sampled dense products.

The common dimension set is **1,2,4,8,16,32,64**, corresponding to doubling levels 0–6. Unless otherwise stated, dense inputs comprise eight integer vectors per dimension, coordinates in [-2,2], from seed `150501670+dimension`. Tests use Python integers and `Fraction` whenever possible. Squared norms permit exact checks of norm power identities without introducing square roots. The module itself is not changed.

Numerical roots of minimal polynomials, sample continuity, spectra, angle coordinates and square roots use float64. Tolerances are stated in the code: root comparisons use 10–12 decimal places; ordinary square-root products use relative and absolute `2e-13`; angle-vector powers use absolute `2e-13` plus NumPy's default relative tolerance. Gelfand envelope checks allow `1e-11`; unit squared lengths are checked to 10 decimal places. A specially conditioned near-negative-real square-root test uses relative `2e-15` and absolute `1e-30`. Exact symbolic phi identities and integer counterexamples do not depend on floating-point residual tolerances.

### Restricted model for the discontinuous additive function

A computer cannot supply a general Hamel-basis oracle for all real numbers. The code therefore handles **only** angles `p*pi+r` with rational p,r, prescribing `phi(p*pi+r)=r`. Since pi is irrational, this prescription is well-defined and Q-linear on that subspace. It can be extended to a Q-linear map on R by extending a Hamel basis. Every such extension satisfies phi(pi)=0 and phi(1)=1, so it is pi-periodic and discontinuous.

`Angle` stores the two rational coefficients exactly. Floating point only chooses well-separated argument branches and checks the corresponding trigonometric vectors. No phi value is inferred from a rounded real number. Global conclusions about a permissible extension are mathematical arguments explained in FINDINGS, not an executable implementation of a discontinuous function on all R. The finite model must not be advertised as such an implementation.

## Complete test inventory in paper order

Identifiers below are method names in `PaperVerification`, omitting `test_`. “Pass” on a discrepancy means that the discrepancy was reproduced. Every coded CPU test is listed.

### 1. The Cayley-Dickson algebras, pages 1–3

| Test | Computation and finite scope | Interpretation |
| --- | --- | --- |
| `01_01_doubling_twist_signed_basis_and_embeddings` | All ordered basis pairs through dimension 64; `twist`, every input-sign combination of `basis`, bilinear product and independent recursive sign agree. Sample zero-padding embeddings from dimensions 1–32 into 2–64. | Pass; the paper and module conventions agree. |
| `01_02_distributivity_unit_and_conjugation` | Six consecutive dense triples per dimension: both distributive laws, two-sided identity, conjugation involution and reversal; dense products also match the independent oracle. | Pass. |
| `01_03_commutativity_associativity_and_alternativity_boundary` | Sample commutativity in dimensions 1,2; explicit quaternion noncommutativity; all 64 quaternion basis triples associative; octonion nonassociative triple; both alternative laws on eight dense octonion pairs. | Pass, with explicit expected failures beyond the relevant boundary. |
| `01_04_lemma_1_1_quadratic_trace_and_norm` | Eight dense vectors plus zero, 1 and -3 in each common dimension. Check both conjugate products, trace and exact annihilation by `t²-2*a0*t+N(a)`. | Pass. |
| `01_05_theorem_1_1_all_bracketings_and_recurrence` | Two dense inputs each in dimensions 2,4,8,16,32; enumerate every binary parenthesization for exponents 1–7 and compare with the independent quadratic recurrence. Also test both identities used in the proof. | Pass. Parenthesization counts are 1,1,2,5,14,42,132; duplicate values are retained so every tree is evaluated. |
| `01_06_printed_sedenion_nonalternative_example` | The exact printed pair `o1=e4+e11`, `o2=e7-e16` in paper numbering. Module and independent products give `o1*o2=0`, `o1*(o1*o2)=0`, `(o1²)*o2=-2*o2`. | Pass, reproducing the paper's example without a basis-convention adjustment beyond indexing. |

The paper's phrase referring to Theorem 1.1 at the end of its overview is consistent with the actual power-associativity theorem. General alternativity and power-associativity remain mathematical statements; finite samples do not prove them for all real vectors and dimensions.

### 2. Radius, pages 3–5

| Test | Computation and finite scope | Interpretation |
| --- | --- | --- |
| `02_01_theorem_2_1_minimal_polynomial_and_divisibility` | Eight dense inputs plus zero and -2 per dimension. Scalar elements use the linear minimal polynomial; nonreal elements use the quadratic. Evaluate these and two higher polynomial multiples, and check exact division with zero remainder. | Pass for the tested Cayley-Dickson elements. Independence of 1 and a rules out a linear annihilator in the nonreal case; arbitrary algebra/field classification is not implemented. |
| `02_02_theorems_2_2_2_3_radius_from_minimal_roots` | Numerically compute roots of the minimal polynomial for dense inputs, zero and -3; compare their maximum modulus with Euclidean length; check homogeneity at scalars -3/2, 0, 2/3. | Pass. Root error is recorded rather than defining the tested radius by the norm. |
| `02_03_power_norm_no_nilpotents_and_continuity_samples` | Eight inputs per common dimension, exponents 1–12: repeated module products equal quadratic-recurrence powers, exact `N(a^k)=N(a)^k`, nonzero inputs have nonzero tested powers. Radius changes under a 1/1000 real perturbation obey the length bound. | Pass. Bounded powers and perturbations are evidence, not proofs of nilpotent absence or continuity. |
| `02_04_radius_is_not_left_multiplication_spectral_radius` | Build the 16×16 left-multiplication matrix of the printed zero divisor from module products; compute rank and eigenvalues. | Supplement passes: element radius sqrt(2), multiplication-operator spectral radius 2, matrix rank 12. The paper's radius must not be replaced by the spectrum of this operator. |

### 3. Subnorms and the Gelfand formula, pages 5–8

| Test | Computation and finite scope | Interpretation |
| --- | --- | --- |
| `03_01_lp_subnorm_definitions_and_nontriangle_example` | Positivity and real homogeneity for p=1/2,1,3/2,2,4,infinity in dimensions 2,8,32; explicit triangle failure at p=1/2. Scalar A0 checks show every lp equals absolute value there. | Pass. The later p≤2 stability cutoff concerns dimension at least two; A0 is exceptional. |
| `03_02_gelfand_lp_and_continuous_angular_subnorm` | Dense input in each of dimensions 2,8,32; exponents 1,2,4,8,16,32,64,128,256. Compare all six lp root ratios with dimension-dependent norm-equivalence bounds. Also use the continuous size `|a|*(1+a0²/(2*|a|²))`, extended by zero at zero. | Pass. Exact integer powers are normalized after multiplication to avoid roundoff contamination. Raw repeated unit-float multiplication is separately recorded at 64,128,256. |
| `03_03_discontinuous_equivalent_bump_and_gelfand` | Equation (3.4), V=real line, kappa=3, base Euclidean norm. Off-line points at distances 1/10,1/100,1/1000 from 1, homogeneous samples, equivalence bounds, and powers of i through 128. | Pass. Values near 1 approach 1 while its assigned value is 3. Power-root discrepancy is bounded by `3^(1/k)`. |
| `03_04_additive_periodic_restriction_and_complex_argument` | Exact rational additivity/scaling and pi-periodicity on a 25-angle set in Q*pi+Q; complex argument reduction leaves phi unchanged. | Pass for the restricted model; global noncomputable extension is a separate argument. |
| `03_05_theorem_3_2_complex_log_gelfand_and_real_homogeneity` | Angles 1, pi-1, pi/3 and exponents 1,2,3,4,8,16,32,64. Actual module complex powers match the symbolic oriented argument; negative real scaling shifts it by pi; exact phi scaling and lp-dependent Gelfand error bounds are checked. | Pass. With f=Euclidean norm, root ratios are exactly exp(phi(angle)) in the symbolic model. With other lp functions, the finite directional factors obey the expected root bounds. |
| `03_06_theorem_3_3_quaternion_negative_homogeneity_counterexample` | q=cos(1)+i*sin(1); the printed folded argument of -q is pi-1. Symbolic log h(q)=1 and log h(-q)=-1, with numerical coordinate/branch confirmation. | Discrepancy reproduced: h is not absolutely homogeneous under negative real scalars. |
| `03_07_theorem_3_3_folded_power_root_counterexample` | Same q, eleven exponents through 64; actual quaternion powers agree with folded branch selection, but phi(arg(q^4))=-4 while phi(arg(q^7))=7. | Records the e^-1/e root-ratio behavior. The analytical failure of the claimed limit uses infinite returns of the irrational rotation, beyond this finite sample. |
| `03_08_dense_angle_families_and_large_log_values` | Four target angles, denominators 10,100,1000, prescribed phi=0 or 1; nearby angles with phi=±1000 and denominator 10000. | Pass for explicit dense-subspace approximants and large prescribed log values. Density/unboundedness on all intervals is supported by the displayed approximation bound, not finite enumeration alone. |

For the Gelfand experiments, `lp(a^k)^(1/k)/|a|` is bounded by the kth-root lp/Euclidean equivalence constants. This explains convergence without assuming that an individual trajectory's error decreases monotonically. The saved data contain every root ratio and the bound at k=256.

### 4. Stability and radial dominance, pages 8–11

| Test | Computation and finite scope | Interpretation |
| --- | --- | --- |
| `04_01_lp_stability_bounds_and_p_gt_2_instability` | For p=1/2,1,3/2,2, use sigma=`d^(1/p-1/2)` on actual powers through 32 in dimensions 2,8,32. For p=3,4,infinity use a=1+i and exponents 4,8,16,32,64, checking the exact log-growth formula. | Pass. For k divisible by 4 the unstable ratio is `2^(k*(1/2-1/p))`; its growth proves no uniform sigma exists for p>2. |
| `04_02_weighted_sup_stability_criterion` | Four rational weight pairs: (1,1),(2,2),(5/3,5/4),(3,1). Exact worst-direction witness (v,u); all coordinate pairs in [-3,3]² for passing criteria; powers through 12 on three inputs. | Pass for the criterion `u²v²>=u²+v²`, including an exact boundary case. Uniform bound sigma=max(u,v) is used for passing weights. |
| `04_03_weighted_l1_stability_criterion` | Four weight pairs including violations of each lower bound; test both coordinate axes, [-3,3]² for passing weights and powers through 12. | Pass for `u>=1 and v>=1`. Axes give the necessary radial-dominance obstructions; the analytic norm bounds give sufficiency. |
| `04_04_equivalent_discontinuous_bump_is_stable` | kappa=3 bump from Section 3 in dimensions 2,8,16, ten inputs per dimension and exponents 1–16. | Pass with sigma=3 and radial dominance. Confirms the example related to Proposition 4.1, not a universal converse for all discontinuous subnorms. |
| `04_05_corollary_4_2_missing_continuity_complex_witness` | Complex angle pi-1 has phi=-1; exact oriented argument identities through 32 preserve the power equation. | Discrepancy in the unrestricted corollary: a permissible global extension gives a strongly stable real subnorm with value e^-1 on a unit vector. Continuity is needed, as in Theorem 4.1. |
| `04_06_theorem_4_3_quaternion_stability_counterexample_family` | Symbolic angles `m*pi-M/2`, M=floor(2*m*pi)-4, m=1,2,10,100; verify branches with actual quaternion squares. | Fixed exponent k=2 already gives log stability ratios 4,16,116,1248. The exact family has no finite uniform bound, so the quaternion stability assertion fails for the displayed construction. |
| `04_07_strong_weighted_l1_criterion_and_boundary_typo` | Six weight pairs; all 625 pairs of inputs in [-2,2]² for strong cases, and exact scalar/i obstructions otherwise. Also a stable non-strong example (u,v)=(4,1). | Passes the preceding criterion `v²>=u>=1`. Reproduces the incorrect endpoint in the subsequent “stable but not strongly” sentence: (4,2) is strongly stable. The strict condition is `u>v²>=1`. |

These tests do not enumerate the set of all subnorms. They check concrete families and provide exact analytic witnesses where a printed universal assertion fails. Corollary 4.1 and the strongly stable radius example also use the norm-power checks in Sections 2 and 5.

### 5. The power equation, pages 11–13

| Test | Computation and finite scope | Interpretation |
| --- | --- | --- |
| `05_01_power_equation_constants_radius_and_unique_scaled_candidate` | Common dimensions, eight dense inputs plus zero, exponents 1–8: squared radius satisfies the power equation; constant 0 and 1 do too. Among positive scalar multiples c*radius, c=1/2,1,2, only c=1 passes at 1,k=2. | Pass. This illustrates Corollaries 5.1–5.2; it does not prove uniqueness among all continuous submoduli. |
| `05_02_nilpotent_obstruction_matrix_reference_only` | A 2×2 strictly upper triangular real matrix is nonzero but its square and spectral radius vanish. Positive candidate f(N) values contradict `f(N²)=f(N)²` when f(0)=0. | Reference-only illustration of Proposition 5.1 outside CayleyKernel's ordinary sequence. No such matrix is passed to the kernel as a Cayley-Dickson element. |
| `05_03_theorem_5_2_complex_power_and_quaternion_failure` | Exact restricted-phi identities on complex powers through 64, contrasted with the folded quaternion fourth power. | Complex model passes; quaternion formula has log defect -8 at k=4 and is not a submodulus as stated. |
| `05_04_lemma_5_1_exact_square_roots_generated_targets` | Eight dense root candidates b with positive scalar part in each dimension 2–64. Form a=b² with module multiplication and recover b by the paper's formula, using exact rational arithmetic; confirm with the independent oracle. Also scalars -9,-4,0,4,9. | Pass, including sedenions and higher algebras, negative real and zero branches. |
| `05_05_lemma_5_1_numerical_square_roots_and_cancellation` | Eight ordinary dense inputs per dimension 2–64; verify literal and reformulated float64 roots. Stress a=(-1,1e-12): literal formula divides by zero after rounding; the stable algebraic reformulation squares back correctly. | Pass as mathematical checks plus a reproduced numerical limitation, not a counterexample to Lemma 5.1. |
| `05_06_nonnegativity_square_argument_and_real_exception` | Sample square decompositions and nonnegative radius values in dimensions 2,8,16,32; signed real identity powers on A0; failed extension of real-part projection to C. | Pass. Illustrates Theorem 5.3's mechanism and why its n>=1 hypothesis matters; arbitrary real-valued solutions are not enumerable. |
| `05_07_closure_operations_on_power_solutions` | Exact families from t=N(a): t, t², max(t,t²), min(t,t²), t²*t³, and indicator(a!=0), tested through exponent 8 in dimensions 2,8,16,32, with inputs below/above unit length and zero. | Pass. These are power solutions; most are not absolutely homogeneous subnorms. |
| `05_08_composition_boundary_and_zero_divisor_embedding` | Sample full norm composition in dimensions 1,2,4,8; embed the printed zero divisors in 16,32,64. For each larger dimension also check each individual power through 8. | Pass: norm composition fails while the single-element power identity still holds. |

The references and author-contact page introduce no further computational claims. No historical priority or cited-paper verification is performed.

### Direct GPU replays

| Test in `GPUVerification` | What `multiply()` would execute |
| --- | --- |
| `test_01_01_multiply_basis_and_dense_doubling_oracle` | Every ordered basis pair and eight dense pairs in all seven common dimensions, float64, against the independent oracle. |
| `test_01_02_05_multiply_powers_norms_and_printed_zero_divisors` | Three dense inputs per dimension, repeated powers through 8 against exact quadratic recurrence; norm-power checks and the printed zero-divisor/nonalternativity products. Power coefficients are compared exactly; squared norms use relative `2e-13` because their float64 magnitudes can exceed exact integer range. |
| `test_05_multiply_square_roots` | Square a batch of eight numerical square roots per dimension 2–64 and compare to their targets with relative/absolute `2e-13`. |

All three were skipped in this run. CPU `mul()` is an explicit separate module-basis adapter, never a replacement hidden behind `multiply()`.

## Numerical results worth retaining

The exact checks produce zero residuals. The numerical radius-root and ordinary square-root residual maxima are saved in JSON; the ordinary square-root product error was approximately **4.0e-15**.

Repeated normalized float multiplication in dimension 32 produces a very different outcome from exact powers normalized afterward:

| Exponent | Raw CPU float squared-norm error | Maximum coordinate error against exact normalized power |
| --- | --- | --- |
| 64 | 1.55e-6 | 5.20e-4 |
| 128 | 7.41e21 | 3.59e10 |
| 256 | 1.68e77 | 1.71e38 |

Dimensions 2 and 8 stayed near rounding precision in this experiment. In higher algebras, multiplication by a fixed unit element need not preserve lengths on arbitrary other elements. Floating-point errors can leave the exact two-dimensional subalgebra of powers and be amplified. This is a numerical conditioning diagnostic of the CPU bilinear implementation; it is **not** a demonstrated GPU defect or a mathematical failure of the paper's power identity. The Gelfand table deliberately uses exact module powers before scaling and does not inherit this corruption.

## What cannot be concluded or has not been coded

| Paper content or extension | Limit of this implementation |
| --- | --- |
| Universal identities at arbitrary dimension, exponent and real input | Finite basis sweeps, dense samples and bounded tree enumerations are not proofs. No proof assistant or formal-variable verification in arbitrary dimension is included. |
| Theorems 2.1–2.2 for every finite-dimensional power-associative algebra over general fields | CayleyKernel implements the ordinary real doubling sequence. Only those elements and a clearly separate nilpotent matrix example are exercised. General-field minimal polynomial existence/uniqueness is not implemented. |
| General numerical minimal-polynomial discovery | For Cayley-Dickson elements it is known explicitly from the scalar/nonreal split. No arbitrary Krylov-rank or symbolic ideal algorithm is provided; the known polynomial is checked by actual multiplication and exact divisibility. |
| True infinite Gelfand limits | Computed exponents stop at 256, with rigorous finite-dimensional lp bounds used to interpret them. The general theorem for every continuous subnorm is not established by sampling. No attempt is made to enumerate all continuous functions. |
| Global continuity, density and discontinuity everywhere | Finite sequences and exact approximation bounds illustrate these properties. Topological assertions require the mathematical arguments; point samples cannot establish them. |
| An everywhere-defined discontinuous Cauchy solution | A Hamel-basis extension is nonconstructive. Only Q*pi+Q is implemented. No rounded-angle lookup, interpolation or measurable surrogate would be a valid global replacement. |
| Quaternion pathological-function results as printed | The explicit folded argument breaks negative-real homogeneity and power compatibility. Their failure is documented, rather than silently replacing the angle convention with a new definition. No corrected quaternion theorem is proposed. |
| All continuous stable subnorms, or all strongly stable subnorms | Concrete lp, weighted norms, an angular continuous subnorm and a line-bump family are tested. The open characterization raised on page 11 is not solved. |
| Theorem 5.1 / Corollary 5.2 uniqueness among all continuous submoduli | The scalar-multiple family is checked; arbitrary functions cannot be exhaustively represented by a finite suite. The paper's general uniqueness theorem is not computationally proved. |
| Theorem 5.3 for every solution of the power equation | Square-root decompositions and representative solutions illustrate the proof. No enumeration of all functions exists; general nonnegativity uses the square argument. |
| A numerically robust square-root library at every scale | A stable variant handles the tested negative-real cancellation case. No exhaustive subnormal/overflow analysis, all possible root branches, or general arbitrary-precision API is supplied. |
| Direct GPU verification in this environment | CuPy is missing. Tests are ready to run when CuPy, a CUDA device and the compiled kernel are available; no GPU run or timing claim is made. |

Existing unrelated work and the previous paper packages were preserved. A narrow `.gitignore` exception makes this paper's documentation and saved results visible to git, without unignoring all other documentation.

## Coefficient-type coverage

Basis/dense products, every intermediate power multiplication, printed zero divisors and square-root products also run typed device replays. Half replays rescale individual operands before quantization; they do not assert the entire eight-step power identity in half. Real spectral radius, positivity, norm composition and square-root existence are not inferred for arbitrary complexified elements.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.
