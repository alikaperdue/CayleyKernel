# Verification of Krausshar, arXiv:1912.01351v1

Rolf Sören Kraußhar, **Function Theories in Cayley-Dickson algebras and Number Theory**, version 1, 19 pages. The arXiv stamp is 3 December 2019; the title page says 4 December 2019. Printed and PDF page numbers agree.

Source: `/home/alika/Desktop/CayleyKernel/1912.01351v1.pdf`. SHA-256: `9acc2035adadf71050304ab673487741fed23f2711f418d25dde2647f3baacb2`. The original PDF is not copied into git. This review concerns that version only; it does not assess later revisions or corrections. Formulas on pages 4–6, 10, 12, 14 and 17 were also checked in rendered pages, since extracted text loses conjugation bars.

## Result and interpretation

The run on 21 October 2026 completed **25 tests: 23 passed, 2 skipped**, with no test failures or errors. Both skips are direct GPU tests: CuPy is unavailable in the selected Python environment. A shared library loads, but that does not establish working device execution. CPU tests use the actual `twist()` and `basis()` routines; no CPU result is presented as execution of `multiply()`.

**A passing counterexample test means that a printed claim failed on its stated input.** In particular, the run does not verify the main trace theorem. It reproduces an exact failure of Proposition 4.9, an exact failure of the two-sided assertion in Proposition 4.11, and a failure of the singularity cancellation used in Theorem 4.14. It also identifies a sign issue in the zeta series and a problem with the general higher-dimensional harmonicity assertion. See [FINDINGS.md](FINDINGS.md) for calculations and qualifications.

The [machine-readable results](results/results.json) contain all numerical observations, pass/skip lists, PDF and source hashes, Python/NumPy/platform information, git HEAD and elapsed time. [The complete test output](results/test-output.txt) lists every test. The checkout already had unrelated changes; git HEAD alone is therefore not the source identity. The saved source hashes identify the files actually run.

## What this paper is trying to do

A lattice is an integer grid, possibly stretched and tilted. A lattice with Cayley-Dickson multiplication permits certain non-real numbers to multiply grid points into other grid points. The paper uses such grids to build higher-dimensional versions of elliptic functions, functions that repeat along all lattice directions except at their poles.

The octonions let products stay inside an eight-dimensional algebra, but brackets matter: `lambda*(z*mu)` can differ from `(lambda*z)*mu`. The central proposed formula sums elliptic-function values over special division points in one lattice cell, expressing that sum using the lattice and certain quasi-period constants. This sum is the “trace” in Section 4.3; it is different from the elementary octonion trace `2*real_part(z)` in Sections 2–3.

The elementary algebra and lattice constructions give useful checks. The move from those constructions to the general two-sided function identity contains a bracket-sensitive failure. That is a mathematical finding about this printed version, rather than a reason to change CayleyKernel to force agreement.

## Code and reproduction

All new executable code is in the paper-specific test package, alongside the earlier paper package:

- [algebra.py](../../../tests/papers/arxiv_1912_01351/algebra.py): exact products, independent doubling, explicit paper table, Cauchy kernel and derivatives, finite lattice sums and numerical differential operators.
- [test_sections.py](../../../tests/papers/arxiv_1912_01351/test_sections.py): the complete tests, named in section order.
- [run.py](../../../tests/papers/arxiv_1912_01351/run.py): section-ordered standalone runner, provenance and observation recording.

From the repository root:

```bash
python3 -m tests.papers.arxiv_1912_01351.run
```

To regenerate the saved results:

```bash
python3 -m tests.papers.arxiv_1912_01351.run \
  --paper /home/alika/Desktop/CayleyKernel/1912.01351v1.pdf \
  --output docs/papers/1912.01351/results
```

Python and NumPy are required; no additional symbolic-math package or pytest is needed. If pytest is installed, the package also supports:

```bash
python3 -m pytest -v tests/papers/arxiv_1912_01351/test_sections.py
```

Both classes carry the repository's `cpu` marker to bypass its general GPU prerequisite fixtures. The two device tests have their own runtime/device/library checks. Once those prerequisites succeed, execution errors and mismatches fail the test rather than being silently replaced by a reference product. The unittest runner explicitly runs CPU checks in paper order, followed by GPU replays.

## Conventions and strength of the checks

Paper indices, kernel binary indices and signed API labels are different:

| Paper basis index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kernel zero-based index | 0 | 1 | 2 | 4 | 3 | 5 | 6 | 7 |

`twist(i,j)` accepts zero-based indices. `basis(i+1,j+1)` returns a signed, one-based label. A positive label `r` means `+e_(r-1)`; a negative label means its negative. This translation is isolated in `mul()`. The entire printed 7×7 imaginary multiplication table is a separate expectation, checked after the basis permutation above.

The independent `recursive()` routine uses the repository's standard doubling convention `(a,b)(c,d)=(ac-conj(d)b, da+b*conj(c))`. It calls neither public sign routine. It is an independent oracle for the module convention, not a literal implementation of the differently presented pair convention on page 4. Agreement with the independently transcribed paper table supplies the octonion convention check.

Most algebraic checks use Python integers and `Fraction`, without tolerances. `q(z)=conj(z)/N(z)^4` and its analytic first derivatives are exact on rational octonions. Numerical experiments use float64 NumPy arithmetic; their bilinear products use a tensor built from `basis()`. They are explicitly CPU experiments. Random seeds and finite bounds appear in the code and inventory below.

Finite-difference residual tolerances are absolute `1e-12` for the sampled Cauchy kernel, `1e-11` for its first derivatives, and `1e-9` for its sampled Laplacian. The finite-series derivative comparison uses relative `2e-8` and absolute `1e-3`, since the near-pole values are large. Parity checks use absolute `1e-8` or `1e-7`, plus NumPy's relative tolerance. Determinant/cofactor checks use small float64 matrices, not a general exact determinant algorithm. Exact counterexamples do not depend on these numerical tolerances.

## Test inventory in paper order

The identifiers below are method names in `PaperVerification`, with the common `test_` prefix omitted. Every coded test is listed, including numerical diagnostics and GPU replays.

### 1. Introduction, pages 1–3

This section motivates the distinction between Clifford paravectors and full Cayley-Dickson arguments. It introduces no separate finite numerical theorem. Its assertions about brackets, multiplication, lattices and failure of module closure are exercised at their later sections. No Clifford algebra implementation or literature-priority test is included.

### 2. Basic algebra, pages 4–5

| Test | What is checked | Scope and outcome |
| --- | --- | --- |
| `02_01_basis_twist_and_independent_doubling` | `twist`, signed `basis`, exact bilinear product and independent recursive doubling agree. | Every ordered basis pair in dimensions 1,2,4,8,16,32; all four input-sign combinations. Pass. |
| `02_02_printed_octonion_table` | Every entry in the printed table, using the explicit basis permutation. | All 49 entries. Pass. |
| `02_03_involution_quadratic_power_and_inverse` | Conjugation reverses multiplication; `a*conj(a)=N(a)`; `a²-S(a)a+N(a)=0`; two fourth-power bracketings; left and right products with `conj(a)/N(a)`. | Twelve seeded dense integer pairs per dimension 1–32 listed above. Pass. Having a two-sided inverse does not imply absence of zero divisors in sedenions. |
| `02_04_composition_alternativity_moufang_and_conjugate_rule` | Norm composition, both alternative laws, the printed Moufang identity, flexibility, and conjugate cancellation rules. | Twenty seeded dense triples each in dimensions 2,4,8. Pass. |
| `02_05_associativity_boundary_and_sedenion_failures` | Quaternion basis associativity; octonion nonassociativity; constructive sedenion zero divisor, norm-composition failure and alternativity failure. | All 64 quaternion basis triples; explicit octonion triple; deterministic sparse sedenion search with witness recorded. Pass. |

The norm is implemented as a sum of **squares**. The displayed component sum on page 5 appears to omit those squares. The alternative doubling chains mentioned at the end of the section are not implemented by CayleyKernel and are not tested here.

### 3. Integrality and CM lattices, pages 5–7

| Test | What is checked | Scope and outcome |
| --- | --- | --- |
| `03_01_trace_norm_polarization_and_brandt_conditions` | The norm-of-sum, product norm, and trace/dot-product identities used in Proposition 3.2; examples of integral individual elements whose sum need not be integral. | Integer and rational examples use exact arithmetic; a sqrt(2) integrality obstruction uses a numerical illustration. Checks supporting identities, not a general classification of Brandt algebras. Pass. |
| `03_02_exact_multiquadratic_lattice_structure_constants` | Canonical radical lattice closure: multiply square-root subset generators using XOR and multiply the overlapping radicands into an integer structure coefficient. | All 64 products for radicands 2,3,5, with kernel binary subset masks. Squared radical identities and destination/sign APIs are checked exactly; sample trace/norm integrality. Conjugation changes only integer coefficient signs. Pass. |
| `03_03_rational_coefficients_do_not_ensure_lattice_closure` | Rational generator scaling need not give a ring over Z; arbitrary independently chosen integer scales also need divisibility conditions; multiplication closure is different from commutativity. | `omega1=sqrt(2)e1/2` squares to `-1/2`. Independently scaling one radical basis generator by 2 gives a `3/2` structure coefficient. These are cautions about unrestricted readings of the lattice paragraph, not refutations of the canonical product-generated integer construction. Pass. |

The canonical lattice with weights `sqrt(1),sqrt(2),sqrt(3),sqrt(6),sqrt(5),sqrt(10),sqrt(15),sqrt(30)` is a concrete integral example. Arbitrary rational points in the surrounding number field are not automatically integer combinations of a chosen lattice basis. Likewise, distinct square-free radicands alone do not ensure field degree `2^k` (for example 2,3,6); the selected 2,3,5 avoid that issue.

### 4.1. Regular functions, pages 7–9

| Test | What is checked | Scope and outcome |
| --- | --- | --- |
| `04_01_01_cauchy_kernel_left_right_regular_and_derivatives` | Exact left and right Dirac residuals for q; numerical residuals for q and three of its partial derivatives; numerical harmonicity. | One explicit nonzero rational octonion, exact analytic first derivatives, central differences for further derivatives. Pass. |
| `04_01_02_fueter_linear_polynomials_and_module_obstruction` | The cancellation `D[(xj-x0*ej)*a]=0` for each basis coefficient; a linear regular polynomial whose product with a constant is not regular. | Seven linear Fueter variables × eight basis coefficients; deterministic search records an exact counterexample to module closure, confirming the paper's warning. Pass. |
| `04_01_03_cauchy_sphere_center_and_singularity_scaling` | The constant-function Cauchy integrand at the center of the unit sphere; area normalization; exact seventh-order Cauchy-kernel pole. | 64 seeded unit sphere points, inward normal for the displayed `q(z-w)` convention; exact scale factors 1/2,1/3,1/4. This is an integrand/normalization check, not general boundary integration. Pass. |
| `04_01_04_higher_algebra_regular_need_not_be_harmonic` | Explicit quadratic sedenion polynomial with `Df=0` coefficientwise and nonzero Laplacian. | Exact coefficient test and independent recursive confirmation of the nonzero Laplacian. Contradicts the unrestricted higher-dimensional sentence at the top of page 8; does not contradict octonion harmonicity. Pass as counterexample. |

### 4.2. Elliptic-function constructions, pages 9–11

| Test | What is checked | Scope and outcome |
| --- | --- | --- |
| `04_02_01_wp_even_zeta_odd_and_corrected_derivative` | Parity of finite symmetric wp/zeta sums; differentiation of a zeta sum using an explicitly proposed **minus** convergence correction gives the defined wp sum. | Cutoff 1 on the anisotropic canonical radical lattice, 6,560 nonzero points; three imaginary derivative directions. Both printed-plus and proposed-minus series are checked for oddness. Pass. |
| `04_02_02_printed_zeta_plus_sign_counterexample` | The printed plus correction gives the wrong derivative constant, and fails to remove the first-order Taylor term of an individual summand. | Exact rational single-summand mismatch; nonzero finite anisotropic aggregate mismatch; far-field norms at scales 16,32,64. The printed term decays approximately as radius^-8, the proposed correction as radius^-9 on this ray. Pass as discrepancy regression. |
| `04_02_03_finite_lattice_periodicity_and_quasiperiod_experiment` | Direct differences under an imaginary lattice period and comparison with `2*zeta(period/2)`. | Cubic cutoffs 1 and 2: 6,560 and 390,624 nonzero points. Defects are recorded, not asserted to vanish or certify convergence. Pass means the diagnostic completed with finite results. |

The recorded wp periodicity defects are about **21.20** and **3.09**; zeta quasi-period defects are about **4.84** and **1.12**. These are substantial truncation errors. They decreased for these two cutoffs, but this is not a bound for the infinite sums. The experiments deliberately retain the defects rather than declaring numerical periodicity “verified.”

The proposed minus correction is labelled in code. Changing that sign does not repair the separate bracket-sensitive Proposition 4.9 failure.

### 4.3. Division values and trace formula, pages 11–17

| Test | What is checked | Scope and outcome |
| --- | --- | --- |
| `04_03_01_kernel_conjugation_and_one_sided_transformations` | The bracketed conjugation factorization preceding Proposition 4.9 and both one-sided scaling identities. | Sixteen seeded dense integer triples; exact rational q values. Pass. |
| `04_03_02_proposition_4_9_exact_counterexample` | The two-sided kernel formula fails for three basis units. | Both module products and independent doubling give `+e4` rather than `-e4` in kernel numbering. Pass as counterexample. |
| `04_03_03_exhaust_basis_triples_and_associative_controls` | Exhaust the two-sided identity on octonion basis triples and compare with a quaternion subalgebra. | 512 triples: 344 satisfy the printed identity, 168 fail. All 64 triples in the selected quaternion subalgebra satisfy it. Pass as finite census and positive control. |
| `04_03_04_proposition_4_11_regular_counterexample` | Analytically differentiate the two-sided transformed kernel and evaluate both Dirac operators at 1. | Exact derivative residual `-8` on the same multiplier pair. Pass as counterexample to the general two-sided assertion. The one-sided claims remain supported. |
| `04_03_05_cm_map_inverse_period_coefficients_and_count` | Period matrix coefficients, conformal scaling, determinant index `N(lambda*mu)^4`, bracketed inverse map; Remark 4.13. | Three integral multiplier pairs; eight basis-point inverse checks each; all 256 half-coordinate division points for lambda=2, mu=1. Matrix determinant is float64; point inversions and half-coordinate enumeration are exact. Pass. |
| `04_03_06_parentheses_and_laurent_cancellation_counterexample` | The two parenthesizations differ; the leading singularity used by Theorem 4.14 does not cancel. | Unit nonreal multipliers on the integer octonion lattice; scales 1/2,1/4,1/8 give exact uncancelled `2*t^-7*e4`. The division set has only 0. Pass as counterexample to a proof step. |
| `04_03_07_cofactor_elimination_for_arbitrary_data_only` | Equation (8) and recovery of seven Ci constants from synthetic equation (7) residuals. | One nonsymmetric sheared matrix with first row representing 1, seeded integer constants, numerical cofactor solve. Pass. No actual Legendre constants or trace theorem are assumed here. |
| `04_03_08_trace_formula_unit_cm_numerical_diagnostic` | Compare the exact empty division trace with the theorem's predicted expression using truncated Legendre constants. | lambda=e1, mu=e2, W=I, division set {0}; lhs exactly 0. Cubic cutoffs 1,2 predict approximately `-9.0898*e4` and `-8.4374*e4`. This is an adverse numerical diagnostic without a certified infinite-series tail, not an exact standalone disproof of the final formula. Pass means the adverse diagnostic was reproduced. |

The unit-multiplier example obeys the stated non-real multiplicator requirements: both units multiply the integer octonion lattice into itself. Nontrivial multiplication in Definition 3.4 does not explicitly require index greater than one. The recorded `e4` trace component uses kernel numbering and is paper `e3`.

### 4.4. Outlook, pages 17–18

No additional finite assertion is introduced here. Algebraicity of the Legendre constants, constructing Galois extensions, higher Cayley-Dickson analogues and gauge-theory applications are research directions, not established outputs of this test suite. No numerical value is asserted to be algebraic, and no Hilbert-twelfth-problem claim is made.

### Optional GPU replays

| Test in `GPUVerification` | Direct `multiply()` calls |
| --- | --- |
| `test_02_multiply_basis_and_dense_independent_oracle` | Batch every basis pair and sixteen seeded dense pairs in dimensions 2,4,8,16,32; float64 expectations from independent recursive doubling. |
| `test_04_03_multiply_proposition_4_9_counterexample` | Evaluate all four bracketed products in the exact unit counterexample with `multiply()`, and compare the result with q(z). |

Both were skipped in the recorded environment. Neither a successful GPU run nor a GPU performance result is claimed.

## Coverage limits and further checks

| Content | What is absent, and what would be needed |
| --- | --- |
| Universal algebraic identities for all real vectors and dimensions | Finite basis sweeps and seeded examples are not proofs. A symbolic or formal proof would be needed for universal claims. |
| Full Brandt classification and arbitrary CM lattices | No enumeration of all integral orders or all real lattice bases. Arbitrary W would need exact number-field arithmetic, generator membership, all structure constants and divisibility checks. Only the specified canonical lattice and small integer CM maps are covered. |
| Clifford comparison in Section 1 | No Clifford paravector/geometric-product implementation; algebraic grades and comparisons with two-sided Clifford multiplication are not tested. |
| Cauchy integral theorem for arbitrary boundaries and functions (Theorem 4.2) | Only center/constant-function normalization is checked. General verification would require oriented surface quadrature, off-center evaluations and explicit regular test functions; even that would remain finite numerical evidence. The normal orientation matters for q(z-w). |
| Full Taylor/Fueter theory | Only degree-one Fueter cancellation and a quadratic sedenion counterexample are implemented. No all-multi-index symmetrizer, general Taylor/Laurent coefficient extractor, or radius-of-convergence verifier. These can be explored with truncated formal polynomials, but arbitrary-function convergence still requires analysis. |
| Liouville's theorem (Theorem 4.3) and absence of nonsingular nonconstant elliptic functions | Global boundedness and entire-space hypotheses cannot be verified by finite point sampling. No finite sample is offered as a proof. |
| Definition 4.4 and arbitrary singular sets | Only the Cauchy-kernel pole scaling is tested. The displayed isolated-point bound uses the fixed point a rather than distance z-a; interpreted literally it degenerates at a=0. No general meromorphic-function class is implemented. |
| Infinite series, normal convergence, full periodicity and Legendre relations | Only the listed finite boxes and ray decay experiment are computed. No rigorous tail estimates, adaptive certified summation, higher-derivative Eisenstein-series engine, or seven-dimensional cotangent comparison. Those would be useful further computational projects after resolving the printed sign issue. |
| Arbitrary division-point quotients | Half-coordinate points and three CM matrices are checked. General non-diagonal quotient enumeration would benefit from Smith/Hermite normal form over exact integers or a number-field lattice representation. |
| Theorem 4.14 as a general trace identity | The printed leading-singularity argument fails exactly. The final trace expression receives a limited numerical adverse diagnostic and an independent cofactor check, not successful verification. Arbitrary nonunit multipliers and nonempty division traces are not fully evaluated. A corrected two-sided transformation, hypotheses and proof are needed before treating the theorem as a reliable oracle. |
| Remark 4.15, the zeta trace constant C | No independent C formula is provided by the paper; evaluating one by rearranging the very identity under test would be circular. Its claimed derivation also depends on the failed cancellation. No independent zeta-division trace verification is claimed. |
| Other algebras and open number theory | No alternate doubling, generalized twisted group algebra, arbitrary number-field/Galois engine or gauge theory. CayleyKernel's public sign routines implement the ordinary real sequence. Numerical fitting cannot certify algebraicity. |
| Bibliography and version history | No independent literature audit, external source comparison or later-version search was requested or performed. Findings refer to the supplied PDF only. |

This package and its documentation were added without changing CayleyKernel's arithmetic or the earlier paper's tests. Existing unrelated working-tree changes remain in place.

## Coefficient-type coverage

Basis/dense products and the nested products in the Proposition 4.9 counterexample also run typed device replays. This checks complexified multiplication and individual half products; lattice convergence, harmonicity and the trace theorem retain the documented scope.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.
