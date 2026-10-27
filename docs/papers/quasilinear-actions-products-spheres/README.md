# Quasilinear actions on products of spheres

Özgün Ünlü and Ergün Yalçın, *Bulletin of the London Mathematical Society* **42** (2010), 981–990. DOI: `10.1112/blms/bdq072`.

Reviewed source: `/home/alika/Desktop/CayleyKernel/Quasilinear actions on products of spheres.pdf`, the ten-page published article, not an arXiv version. Printed page numbers below refer to 981–990. The complete text was extracted and the important formula pages visually checked against rendered pages.

The paper uses Cayley–Dickson algebras to construct free finite-group actions on products of spheres. A free action means that no nonidentity group element fixes a point. The representation's **fixity** is the largest dimension of the vectors fixed by a nonidentity element. The authors arrange enough mutually independent algebra-generated constraints that the last sphere has no fixed vectors left.

The algebraic part is a good fit for CayleyKernel. The smooth manifold and bundle conclusions require topology beyond the module. This review distinguishes exact algebra identities valid for arbitrary coefficients, explicit finite frame examples, arithmetic checks conditional on cited topological theorems, and results not computationally established.

## Files and reproduction

- [Paper-ordered executable tests](../../../tests/papers/quasilinear_spheres/test_sections.py).
- [Exact arithmetic and frame helpers](../../../tests/papers/quasilinear_spheres/algebra.py).
- [Report runner](../../../tests/papers/quasilinear_spheres/run.py).
- [Findings and qualifications](FINDINGS.md).
- [Results and provenance](results/results.json), [full test log](results/test-output.txt), and [exact certificates](results/certificates.json).

From the repository root:

```bash
python3 -m tests.papers.quasilinear_spheres.run \
  --paper '/home/alika/Desktop/CayleyKernel/Quasilinear actions on products of spheres.pdf' \
  --output docs/papers/quasilinear-actions-products-spheres/results
```

Without writing a report:

```bash
python3 -m unittest tests.papers.quasilinear_spheres.test_sections -v
```

NumPy and the local CayleyKernel module are needed. CPU tests require neither pytest nor CuPy. GPU tests use the actual `multiply()` function when CuPy, a CUDA device, and the compiled library are available; they skip explicitly otherwise. No GPU results are inferred from the CPU checks.

Recorded run: **24 tests: 22 passed, 2 GPU tests skipped, no failures or errors**. JSON records the PDF and Python source SHA256 hashes, git HEAD, runtime information, seed, all observations, passed test IDs, and skip reasons. The standalone runner orders CPU tests by section, then optional GPU tests. Ordinary unittest discovery may put the GPU class first.

## Conventions and exactness

The paper's chain is `A1=R`, `A2=C`, `A3=H`, `A4=O`, `A5=S`, with dimensions 1, 2, 4, 8, 16. Helpers use doubling level `n` for dimension `2**n`; this is different from the paper's representation dimension `n`.

Algebra vectors are sparse dictionaries `{i: coefficient}` with mathematical zero-based basis indices. Every product uses `CayleyKernel.basis(i+1,j+1)` and translates its signed **one-based** result back to a coefficient and index. `twist(i,j)` is compared separately against an independently recursive implementation of the printed doubling formula. These routines agree for every basis pair through dimension 16, including all four input sign combinations.

Tensor coordinates use index `row*d + algebra_index`, where `d` is the algebra dimension. The action in Section 3 is **right multiplication**, separately in each tensor row. No associativity is assumed. In Section 2 the printed Hermitian form is `sum(v_r conjugate(w_r))`; its classical spans use left coefficients. These conventions are kept separate explicitly.

Rank, nullspace, and orthogonality checks use integers and `Fraction`, with no numerical tolerance. General frame examples are stored as nonzero rational rays. Dividing each ray by its positive real norm gives an actual unit frame in `W_k`. This normalization leaves all tested orthogonality conditions, generated subspaces, and ranks unchanged. It does not provide a global choice of frame or a bundle trivialization.

## Tests in paper order

Each name below is the exact method name in `test_sections.py`. A passing diagnostic that exposes a missing step is not a verification of that literal sentence.

### 1. Introduction (pp. 981–982)

1. **`test_01_01_fixity_and_free_stiefel_dimension_obstruction`** constructs faithful `C2` representations `diag(I_f,-I_2)` for every `f=0,...,5`. It checks the order-two group law and computes the fixed dimension as `nullity(g-I)`. All determinants are +1. A fixed orthonormal `(f+1)`-frame cannot fit in that fixed subspace. These are explicit examples of the dimension obstruction behind freeness on Stiefel manifolds, not an enumeration of finite groups.
2. **`test_01_02_theorems_1_1_1_2_sphere_dimension_arithmetic`** checks the complex fixity-two bounds for `n=3,...,100`. The allowed multiplicity is `q=3` at `n=3` and `q=2` thereafter, with sphere dimensions `2n-1`, `4n-5`, `q(4n-8)-1`. At `n=4` it recovers `(7,11,15)`. It also records the real fixity-four construction at `n=12`, with dimensions `(11,21,39,71,255)` using `q=2` at the last stage.

**Not coded as a theorem:** the rank conjecture, the general existence problem in the introduction, all finite subgroups of `SU(4)`, and the full assertions of Theorems 1.1/1.2 and Corollary 1.3. They provide context and targets for the later checks. The review does not assert that the paper's 2010 descriptions of open problems remain current.

### 2. Proof of Theorem 1.1 (pp. 982–986)

3. **`test_02_01_conjugation_hermitian_and_real_group_invariance`** checks Hermitian symmetry, the norm formula, and invariance under the rational orthogonal row rotation with entries `3/5`, `4/5` for `R`, `C`, and `H`. This tests a concrete family of coordinate actions and the conventions used for the classical Stiefel construction.
4. **`test_02_02_stiefel_fibres_and_lemma_2_1_stable_complement`** constructs nontrivial unit two-frames over each of `R`, `C`, and `H` in four coordinates. From actual products it forms their classical coefficient spans, computes the orthogonal complement, verifies its dimension `d(n-k)`, and verifies that adjoining the trivial frame span restores all `dn` real dimensions. The sphere fibre has dimension `d(n-k)-1`. This is a fibrewise model of the stable-complement argument in Lemma 2.1.
5. **`test_02_03_lemma_2_2_exact_integer_stable_range_boundaries`** checks the stable-range inequality `base_dimension <= c(q*rank+1)-2` for `c=1,2,4`, ranks 1–11, and base dimensions 0–59. It verifies that the computed positive Whitney multiplicity is minimal, including the failure at `q-1`. This checks use of the cited theorem, not the theorem's homotopy-theoretic proof.
6. **`test_02_04_theorem_1_1_complex_to_quaternionic_fixity`** uses `C2` acting on `C^4` by `diag(1,1,-1,-1)`. Its fixed part after quaternionic extension has real dimension 8, hence quaternionic dimension 2. The coefficient span of two fixed quaternionic frame vectors exhausts that part; the fixed last fibre is zero. This is an explicit realization of the freeness argument, with final sphere dimensions `(7,11,15)` conditional on the bundle argument.
7. **`test_02_05_real_fixity_three_whitney_bounds`** checks `7n-13 <= q(4n-12)+2` for `n=4,...,100`. At `n=4`, minimum `q=4`; `q=2` suffices once `n>=9`. It records every resulting four-sphere dimension list and checks the lowest boundary of the preceding connectivity inequality `3n-4 <= 4n-6`.
8. **`test_02_06_omitting_complex_orthogonality_creates_fixed_points`** implements the warning on p. 986. A real fixity-two involution fixes a unit tuple satisfying the weaker real-pair constraints. The omitted imaginary constraint has inner product 1, so the tuple is excluded by the correct construction. This explains concretely why the extra complex orthogonality condition is necessary.
9. **`test_02_07_remark_2_4_orientation_sign_arithmetic`** checks determinant signs of diagonal orthogonal actions and that both possible signs become +1 modulo 2. This is only the elementary orientation arithmetic relevant to Remark 2.4; it does not compute the induced homology of a constructed sphere bundle.

**Not computationally established:** the global Stiefel/coset homeomorphism and fibration sequence; continuity of global stable bundle splittings in Lemma 2.1; the classifying-space, connectivity and compression arguments in Lemma 2.2; smooth replacement of trivializations in Remark 2.3; the pullback bundle trivializations; all induced homology actions in Remark 2.4. No numerical samples could certify these global statements. The tests check their finite algebraic ingredients and dimension hypotheses.

### 3. General construction and proof of Theorem 1.2 (pp. 986–990)

The PDF prints the heading “Proof of Theorem 1.1” again here, although the introduction and this section's main proof identify its target as Theorem 1.2.

10. **`test_03_01_full_doubling_table_twist_and_signed_basis_labels`** checks all **341** basis pairs across dimensions 1, 2, 4, 8, and 16 against the independent recursive doubling formula. All four signed input-label combinations are checked. It also checks general integer products and conjugation reversal. Equality on the entire basis table certifies equality of the bilinear multiplications for arbitrary real coefficients in these dimensions.
11. **`test_03_02_module_warning_bilinearity_and_nonassociativity`** checks distributivity and supplies explicit associator witnesses in `O` and `S`. Section 3's “module” language must be read as a bilinear multiplication operation where appropriate, as the authors themselves warn. Ordinary associative module identities cannot be silently used.
12. **`test_03_03_lemma_3_1_uniformity_polynomial_certificate`** verifies an identity for **every coefficient**, not random values: for each strictly earlier/later pair of the five algebras, an earlier element `a` embedded in the later algebra satisfies `|ab|^2=|a|^2|b|^2`. If `L_i` is the multiplication matrix of an earlier basis element, the checks certify `L_i^T L_i=I` and `L_i^T L_j+L_j^T L_i=0` for `i!=j`. Therefore `L_a^T L_a=|a|^2 I` for arbitrary real `a`. A nonzero earlier coefficient has zero annihilator. The same holds for a nonzero tensor vector because at least one row is nonzero. This establishes the zero-annihilator algebraic ingredient of uniformity for the whole five-stage tensor chain; it does not certify the subsequent global bundle conclusion.
13. **`test_03_04_sedenion_zero_divisors_and_sixth_stage_obstruction`** searches for two nonzero two-term sedenions with zero product, verifies the product exactly, and embeds them into dimension 32. The embedded zero divisor has nonzero annihilator while the identity has zero annihilator. Thus merely appending a sixth Cayley–Dickson stage would lose the stated constant-annihilator condition. This is a concrete obstruction to that extension, not an impossibility theorem for every alternative construction.
14. **`test_03_05_definition_3_2_separability_and_full_direct_sum`** constructs exact admissible four-frame rays for `(n,seed)=(5,2010),(5,981),(12,2010)`. At every stage `k=1,...,4`, it checks each generated block's pairwise independence and, separately, the rank of the entire sum: `k*2^k`. Full joint independence is stronger than the printed pairwise-intersection condition. These finite examples support the construction but do not prove separability for every frame.
15. **`test_03_06_lemma_3_3_pairwise_condition_alone_is_insufficient`** records an elementary warning: three lines in `R^2` can have all pairwise intersections zero while their total sum is not direct. This does **not** provide a counterexample satisfying every hypothesis of Lemma 3.3 or a counterexample to the Cayley–Dickson construction. It explains why the previous test checks full joint rank rather than inferring it from pairs. The abstract lemma's full generality remains outside this verification.
16. **`test_03_07_proposition_3_4_exact_complement_and_stabilization`** uses a seeded frame at `n=5` and independently computes all four successive fibre kernels from the multiplication constraints. Their dimensions are `(n-k)2^k`: 8, 12, 16, 16. It verifies every kernel vector's orthogonality, kernel independence, and that constraint span plus kernel restores the entire ambient space. All frame rays and kernel bases are exported as exact certificates.
17. **`test_03_08_proposition_3_5_equation_3_1_thresholds`** evaluates the four displayed inequalities, obtaining minimum representation dimensions **2, 5, 12, 27**. Thus `n>=12` handles the first three; the last needs `n>=27` without extra copies. This is dimension arithmetic, conditional on the splitting proposition's hypotheses and cited stable-range theorem.
18. **`test_03_09_whitney_double_rank_and_sphere_off_by_one`** checks all `n=12,...,100`: the final vector rank is `16n-64`, its `q`-fold Whitney sum has rank `q(16n-64)`, and its unit sphere has dimension one less. `q=2` suffices for all these `n`; `q=1` suffices at `n>=27`. The printed p. 989 “vector bundle ... dimension ... -1” conflates vector rank with sphere dimension.
19. **`test_03_10_group_equivariance_and_frame_preservation`** constructs an admissible frame at `n=12` and applies the faithful oriented involution `diag(I4,-I8)`. It checks preservation of lengths and all actual frame constraints, and checks that the row action commutes with right multiplication. This is an explicit quasilinear group action before any choice of product trivialization.
20. **`test_03_11_fixity_four_final_fibre_has_no_fixed_vectors`** builds a nontrivial admissible `W4` frame in the four-dimensional fixed real subspace of that involution. Its four sedenion-generated blocks have joint rank **64**, exhausting the fixed tensor space of dimension `16*4`. The exact fixed last fibre is zero, as is its two-copy version. This tests the central fixed-point obstruction for a specific frame and faithful group representation, not every possible group or frame.
21. **`test_03_12_corollary_coordinate_extraction_requires_orthogonalization`** implements an explicit admissible five-frame whose raw nonzero real components offer **no** orthogonal five-tuple by direct selection. Gram–Schmidt nevertheless yields five nonzero orthogonal fixed rays. This qualifies the literal coordinate-selection sentence on p. 990: selecting raw coordinates alone need not give a Stiefel point, even after normalization. The certificate supplies both the original frame and corrected orthogonal rays; see the detailed example in [FINDINGS.md](FINDINGS.md).
22. **`test_03_13_corollary_fixity_five_next_frame_dimension`** constructs another admissible five-frame, checks that its real components span five dimensions, and orthogonalizes them. A six-frame cannot be fixed in a five-dimensional fixed space. This tests the dimension ingredient in the real argument for Corollary 1.3.

**Not computationally established:** Lemmas 3.1/3.3 and Proposition 3.4 as global bundle statements for every abstract filtration; separability/full joint independence for every admissible Cayley–Dickson frame; Proposition 3.5's global smooth product conclusion; the homology-trivial conclusion of Theorem 1.2; the periodic cohomology/Smith theory and periodic-complex construction cited in Corollary 1.3. The code does not build a finite CW complex, compute a classifying-space homotopy, construct a diffeomorphism, or audit the seven cited references. Tests of exact fibres should not be read as those results.

### Optional GPU execution of the algebra used above

23. **`test_04_01_multiply_all_basis_pairs_through_sedenions`** batches every basis pair in dimensions 1–16 through the real GPU `multiply()` API and compares the output with the independent recursive doubling table. It is skipped in the recorded environment.
24. **`test_04_02_multiply_uniformity_and_zero_divisor_witness`** batches integer octonion/sedenion products against the independent oracle, checks norm composition for embedded earlier elements, and checks the explicit zero divisor `(e1+e10)(e4-e15)=0`. It is skipped in the recorded environment.

## Interpretation of the results

The strongest result of this computation is the complete coefficient certificate for the earlier-to-later norm identity. It applies to arbitrary real coefficients in the specified inclusions. Full exact fibre bases and concrete fixed-space obstructions give additional reproducible examples of how the construction works.

The topological conclusions remain dependent on the mathematical arguments and cited theorems. Passing all coded tests is evidence for the checked ingredients, not a computational proof of free smooth actions on products of spheres for all groups covered by the article. The wording qualifications and exact counterexample to literal raw-coordinate selection are recorded explicitly rather than hidden by the passing regression tests.

## Coefficient-type coverage

Basis products through sedenions, uniformity product fixtures, embedded products and zero-divisor witnesses also run typed device replays. Complexification checks multiplication only: positive real norm composition, injectivity of real frame maps and free group actions remain separate real-algebra statements.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.
