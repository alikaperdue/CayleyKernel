# Verification of Zhilina, arXiv:2608.28176v1

Svetlana Zhilina, **Orthogonality graphs of real Cayley–Dickson algebras. Part I: Doubly alternative zero divisors and their hexagons**, version 1, dated 28 August 2026, 28 PDF pages. Page numbers below are the paper's printed page numbers, which match the PDF page numbers.

The reviewed input was `2608.28176v1.pdf`. Its SHA-256 is `750e708c0284c4e4b45daaea101c39db0256cf9dec659dc899311f237ac7a490`. The input PDF is not duplicated in this directory.

## Recorded result

On 20 October 2026, the paper-specific runner completed **31 tests: 29 passed and 2 skipped**, with no failures or errors. The two skips are the direct GPU checks because CuPy is missing from the selected Python environment. The 29 passes include two deliberate regressions reproducing discrepancies in the printed statements. The saved run took approximately two seconds on this environment; this is suite duration, not a kernel benchmark.

## What the paper means, in practical terms

A zero divisor is a nonzero element that gives zero when multiplied by another nonzero element. The paper connects such elements in a graph. Here **orthogonal means both products are zero**, `xy = yx = 0`. That differs from two ordinary coordinate vectors having dot product zero. The code checks both meanings separately.

A graph vertex represents a whole line: multiplying every coefficient of an element by the same nonzero number gives the same vertex. A hexagon therefore needs six distinct lines, not just six differently written vectors. An annihilator is the vector space of all partners giving a zero product.

The paper studies when a zero product forces other zero products. Its main constructions produce a six-cycle, then a twelve-vertex double hexagon. The latter has 24 undirected edges: six copies of the four-edge graph K2,2. Section 5 builds quaternion and octonion subalgebras under specific assumptions. Section 6 classifies zero divisors made from one basis unit in each half of a doubled algebra.

## Run it

From the repository root, with Python 3.10 or later:

```bash
python3 -m tests.papers.arxiv_2608_28176.run
```

Save an auditable run, optionally hashing the input PDF:

```bash
python3 -m tests.papers.arxiv_2608_28176.run \
  --paper /path/to/2608.28176v1.pdf \
  --output /tmp/cayley-paper-2608.28176/results
```

The standard library `unittest` runner needs no pytest, NumPy, or CuPy for CPU checks. The tests also work with the repository's pytest setup:

```bash
python3 -m pytest -v tests/papers/arxiv_2608_28176/test_sections.py
```

The `cpu` marker bypasses the repository's general GPU prerequisite fixture. The three GPU methods do their own prerequisite checks. When CuPy, a CUDA device, and the compiled kernel are available, the same commands run the `multiply()` checks automatically. Missing prerequisites produce explicit skips. A multiplication mismatch or a kernel execution error after prerequisites succeed is a failure.

The saved [results](results/results.json) record the historical run. Raw console logs are local intermediates and are not required to interpret that JSON. See [findings](FINDINGS.md) for the two discrepancies found in the paper. A passing counterexample regression means that the discrepancy was reproduced; it does **not** mean the printed statement passed.

## Files and arithmetic

- [test_sections.py](../../../tests/papers/arxiv_2608_28176/test_sections.py): tests arranged by paper section and statement number, followed by optional device checks.
- [algebra.py](../../../tests/papers/arxiv_2608_28176/algebra.py): exact sparse arithmetic, independent recursive oracle, projective lines, graphs, and exact matrix rank.
- [tables.py](../../../tests/papers/arxiv_2608_28176/tables.py): separate expectations for all four printed multiplication tables. Table 1 was checked visually on page 16.
- [run.py](../../../tests/papers/arxiv_2608_28176/run.py): standalone runner and report writer.

Sparse elements are dictionaries `{zero_based_basis_index: coefficient}`. Coefficients are Python integers or exact fractions, so the CPU assertions have no floating point tolerance. A zero product is an empty dictionary after coefficient cancellation. The main multiplication adapter calls `CayleyKernel.basis(i+1, j+1)` for every pair of nonzero terms. The `+1` matters: this API uses signed, one-based labels, while the paper and `twist()` use zero-based indices. All four choices of input signs are tested independently.

The independent oracle implements Definition 2.4 by following the four doubling cases. It never calls `twist()` or `basis()`. Agreement between the two public routines alone would not be independent evidence, since `basis()` calls `twist()` internally. The iterative high-index oracle also follows doubling cases, so it avoids Python's recursion limit.

`multiply()` is exercised by three optional GPU methods. The original two retain batched float64 products with exact CPU expectations and add half and complex product replays. The third checks complex coordinate embeddings of mixed signatures. See the coefficient-type coverage below for input scaling, tolerances and signature transport. The tests do not replace `multiply()` with a CPU fallback and do not modify CayleyKernel.

## Coverage in paper order

In the inventory below, a range of dimensions describes the **algebra dimension**, not the number of doubling steps. Test method suffixes identify the implementation in `test_sections.py`. Every coded test is listed, including supplements and GPU checks.

### 1. Introduction, pages 1–2

This section states the research motivation and points forward to later results. It contributes no separate numerical identity. The hexagons, multiplication table, subalgebras, and classification claims are checked at their later statements. Historical priority and literature claims are not computational tests.

### 2. Overview, pages 2–6

| Statements | Coded test | What is checked and how far |
| --- | --- | --- |
| Definitions 2.1–2.3; Notation 2.2 | `02_01_graph_definitions_and_projective_scaling` | A genuine zero-divisor pair, scalar invariance of its projective lines and adjacency, rejection of zero as a vertex, and a dot-perpendicular basis pair whose product is nonzero. |
| Definitions 2.4–2.6; basis convention | `02_02_definition_2_4_twist_basis_and_oracle` | Exhaust every ordered basis pair in dimensions 1,2,4,8,16,32,64 against independent doubling; test `twist`, sparse multiplication, and all four signed `basis` inputs. Independently sample 24 pairs at each of 32-,128-,1024-bit index widths with seed 260828176. |
| Proposition 2.7; Notation 2.8; Proposition 2.9; Lemma 2.10; Definitions 2.11–2.15 | `02_03_01_norm_involution_bilinear_form_and_flexibility` | Twelve seeded dense integer triples per dimension 2–32: conjugation reverses multiplication; norm is scalar, additive across the two halves, and equals sum of squares; real part and polarization identities; both adjoint identities; flexibility and associator reversal. Dimensions 2–8 additionally satisfy alternativity and multiplicative norm. A sedenion zero divisor supplies explicit negative controls for both global claims. |
| Lemma 2.14(1),(3), plus (2) in a reference split algebra | `02_03_02_lemma_2_14_anticentralizers` | Exact rank of `b -> ab+ba` gives the anticentralizer dimension for three sedenion examples. Pure coordinate tests check its dot-product description. A nonpure split-quaternion example of norm zero has a one-dimensional anticentralizer spanned by its conjugate. |
| Definition 2.15; Lemma 2.16 | `02_03_03_lemma_2_16_conditional_norm` | Every ordered basis pair in dimensions 8,16,32 strongly alternates and has multiplicative norm. This is conditional testing, not an assertion that every element is alternative above dimension eight. |

Definitions of pure, doubly pure, and four times pure elements become coordinate exclusions in later tests. Examples 2.13 identify the standard algebras rather than adding new formulas. Split norm examples are tested in the reference supplements below.

### 3. Hexagons arising from alternativity, pages 7–11

| Statements | Coded test | What is checked and how far |
| --- | --- | --- |
| Notation 3.1; Lemmas 3.2–3.3; Remarks 3.4,3.6; Corollary 3.7; Proposition 3.8; Description 3.9 | `03_01_lemmas_3_2_to_3_8_hexagon` | Twelve deterministic zero-product pairs in each of dimensions 16 and 32. Verify the component strong-alternativity hypotheses, strong alternativity of the derived components, equal nonzero component norms, all six directed zero products, six distinct projective lines, and both alternate expressions in Proposition 3.8. In the ordinary sequence gamma=chi=−1. **The first derived component is conjugate(ac)**; PDF text extraction drops its overbar. |
| Corollary 3.7; Lemma 3.3; Remark 3.4 | `03_01_reference_only_degenerate_cycle_and_zero_norm_branch` | Reference-only split-quaternion counterexample to six distinct vertices, fully described in FINDINGS. Also checks the zero-component-norm branch of Lemma 3.3 at four values of chi, including zero and a fraction. That last example has a zero second component and zero derived vertex; it checks the algebraic equality, not a graph cycle. |
| Definition 3.12; Lemmas 3.13–3.14; Remark 3.15 | `03_02_lemmas_3_13_3_14_full_annihilator_spaces` | Four example pairs in each of dimensions 16 and 32. Construct both annihilator formulas from every component basis vector; check the conditions in both directions, the second parametrization of the orthogonalizer, and compare span ranks to the **full multiplication-matrix nullities**. This verifies the whole annihilator spaces for these examples, not just a few partners. |
| Proposition 3.10; Notation 3.11; Corollary 3.16 | `03_02_proposition_3_10_and_corollary_3_16` | Twelve ordinary sedenion hexagons: every ordered pair of vertices is tested; exactly the six cycle edges and their reverses vanish, with no chords or loops. Separate reference split examples check a nonpure norm-zero conjugate pair and pure self-orthogonality (Proposition 3.5). This does not exhaust Proposition 3.10 in arbitrary algebras. |
| Sections 3–4, supplement | `03_to_04_nonbasis_rational_witness` | Rotate an annihilator through rational coefficients 3/5 and 4/5. Check the hypotheses and unit component norms; verify the zero product, all 196 entries of Table 1, and rank 14. This exercises genuine sums of basis units. |

### 4. Main-sequence hexagons, pages 11–16

| Statements | Coded test | What is checked and how far |
| --- | --- | --- |
| Lemma 4.1; Corollary 4.2; Lemma 4.3; Corollary 4.4 | `04_01_lemmas_4_1_4_3_corollaries_4_2_4_4` | Twelve example pairs in each of dimensions 16 and 32: reverse products vanish, tilde preserves annihilation, all four components are pure, paired norms match, the component product relations hold, all four associator and eight anti-associator equalities hold, and the required Euclidean perpendicularity conditions hold. |
| Lemmas 4.5,4.6,4.10; Theorem 4.11; Remarks 4.12–4.13 | `04_02_01_lemmas_4_5_4_6_4_10_and_table_1` | For the same 24 examples: sign-flipped products from Lemma 4.5; seven-vector orthonormality; every one of the 14×14 entries in Table 1, including auxiliary f7,f8,f9 and their tildes; the Gram matrix and rank 14. A nonzero associator also confirms the final observation that span(f0,f1,f2,f3) is not associative. Lemma 4.10's twelve products are included in this full table. |
| Corollary 4.7; Description 4.8; Remark 4.9; Remarks 4.12–4.13 | `04_02_02_description_4_8_double_hexagon_graph` | Construct the twelve vertices in corner order f1,f5,f3,f6,f2,f4, with each f paired with −tilde(f). Check twelve distinct lines, all pairs of vertices, exactly 24 undirected edges, the six K2,2 blocks, no extra edges, and rank 12. Reversal preserves the undirected edge set. |

### 5. Alternative subalgebras, pages 17–20

| Statements | Coded test | What is checked and how far |
| --- | --- | --- |
| Lemma 5.1; Corollary 5.2 | `05_01_lemma_5_1_corollary_5_2` | Dimensions 8,16,32: every basis tilde sign/index; twelve seeded doubly pure vector pairs per dimension; tilde squared, product/tilde identity, and both dot-product equivalences. |
| Lemma 5.3; Table 2 | `05_02_lemma_5_3_table_2` | Three doubly pure examples per dimension 8,16,32, including sums and nonunit norms. Check every Table 2 entry and all 4³ basis associators. Bilinearity/trilinearity then establish closure and associativity for each tested span. |
| Corollary 5.5 | `05_03_corollary_5_5_strong_alternativity_polarization` | In dimensions 8,16,32 check both alternation directions against every basis vector; polarize the quadratic direction using every unordered distinct pair. This establishes the claim for **all real vectors in each of these fixed dimensions**, not just basis partners. |
| Lemma 5.6; Table 3 | `05_04_lemma_5_6_table_3` | Three strongly alternating perpendicular pure pairs per dimension 8,16,32, with a scaled generator. Check every Table 3 entry and all basis associators of each four-dimensional span. |
| Lemma 5.8; Table 4; equations (5.1)–(5.4) | `05_05_lemma_5_8_table_4_alternativity_and_moufang` | In dimensions 8,16,32, verify hypotheses, rank eight, all Table 4 entries, and the homomorphism against a separately calculated source algebra with parameters (−norm(a),−norm(b),−1). Basis polarization checks alternativity throughout each tested span. Twelve seeded combinations per span check all four printed Moufang identities. |
| Corollary 5.9 | `05_06_corollary_5_9` | Every ordered basis pair in the preceding algebra, embedded into dimensions 8,16,32: verify all three identities with explicit conjugations and signs. |
| Remarks 5.4,5.7; degenerate cases of Lemmas 5.3,5.6 | `05_07_reference_only_split_degenerate_homomorphisms` | Split-octonion examples with nonzero norm-zero generators: tilde(a)=a, rank-three image of Table 2, and a nonzero strongly alternating perpendicular pair with ab=0 satisfying Table 3. These realize the nontrivial kernels mentioned in the paper. |

### 6. Basis-pair zero divisors, pages 20–27

Here n counts the **component algebra's** doublings; the full algebra is dimension 2^(n+1). All pure candidates are `(e_i, s e_j)` with i>0 and s=±1. A candidate is retained as a graph vertex only when it has an annihilator. Self-products are stored separately from graph edges: a graph has distinct endpoints even where the paper discusses an element being orthogonal to itself.

| Statements | Coded test | What is checked and how far |
| --- | --- | --- |
| Notations 6.1,6.4; Lemma 6.2; Corollary 6.3; Proposition 6.5 | `06_01_01_lemmas_6_2_6_3_proposition_6_5_components` | Build the complete finite pure basis-pair graphs for component n=1–5, full dimensions 4–64. Traverse connected components; all their component products have the same unsigned basis label, and their characteristics match the norm formula. Both multiplication orders are checked for each neighbor. For the sedenions obtain 84 active vertices, degree four, and 168 edges. |
| Remark 6.6 | `06_01_02_remark_6_6_four_types` | Exhaust pure basis-pair supports at component n=2,3,4; check that every pair belongs to exactly the printed type I,II,III,IV according to its component product label and that the displayed forms cover it. Signs do not change the product's line. |
| Proposition 6.7 | `06_02_01_proposition_6_7_all_pure_basis_pairs` | At component n=2,3,4, every pure support quadruple that can cancel by XOR, with both relative signs in both factors: replacing (c,d) by (d,−c) preserves orthogonality. Includes positive and negative cases. XOR-incompatible supports are excluded because neither side can vanish. |
| Proposition 6.8 | `06_02_02_proposition_6_8_independent_eight_units` | In the 32-dimensional algebra, all 840 ordered distinct quadruples from seven lower-half pure basis units; their four tildes make eight independent units. Check both equivalences, and require nonzero numbers of positive and negative cases. |
| Lemmas 6.9–6.12,6.15, ordinary branch | `06_02_03_lemmas_6_9_to_6_12_and_6_15_neighbors` | Component n=2,3,4: exclude diagonal and second-component-scalar candidates; verify exact neighbor sets for tilde-related pairs; verify the full orthogonality criterion and shared neighbors of (c,tilde(1)) and (tilde(1),−c); verify both claimed neighbors for every pair with four independent tilde-related supports. |
| Notation 6.13; Lemma 6.14 | `06_02_04_lemma_6_14_reduction` | Exhaust ordered independent pure lower-half quadruples at component n=4 (n=3 has no four such units). Check all 4×4 combinations for part (2), using an independent reference split algebra for the reduced positive last parameter. For part (1), check a **proposed corrected** Proposition 6.7 orbit, clearly identified in code and FINDINGS. This is not verification of the printed list. |
| Lemma 6.14(1), printed version | `06_02_05_lemma_6_14_1_printed_counterexample` | Exact concrete counterexample, checked with both CayleyKernel and independent doubling. The regression passes only if the advertised equivalence fails on this example. |
| Theorem 6.16 | `06_02_06_theorem_6_16_exact_kernel_classification` | Exhaust every candidate at component n=1–4 (full dimensions 4–32). Compare the theorem's exclusions to **both full left and right multiplication-matrix nullities**, as well as graph neighbors. At n=5 also exhaust candidate neighbor existence, without claiming full rank verification there. |
| Section 6 split/mixed branches, supplement | `06_reference_only_all_signatures_classification_and_neighbors` | Component n=1,2,3, every ±1 parameter signature: full dimensions 4,8,16 with 4,8,16 signatures respectively. Check Theorem 6.16 with exact right multiplication rank and neighbor existence; component-product invariants, characteristic, self-orthogonality for chi=+1, Lemma 6.9's required neighbors, the full iff of Lemma 6.11, and the positive constructions in Lemmas 6.10,6.12. These are explicitly reference-only checks. |

Graph construction does not assume the classification theorem it tests. Expanding a four-term product shows that cancellation requires `i XOR j = k XOR l`, because terms on different basis supports cannot cancel. For every first candidate and every possible pure first component of a second candidate, the remaining component index is determined by that equation; **both** signs and multiplication orders are then evaluated. For dimensions 4–32, exact matrix rank additionally rules out arbitrary dense annihilators for the candidates classified as nonzero divisors. Searching only basis-pair partners would not justify that conclusion by itself.

### GPU replays of Sections 2–6

| Coded test | Direct calls to `multiply()` |
| --- | --- |
| `GPUVerification.test_02_multiply_all_basiss_and_dense_oracle` | In dimensions 2,4,8,16,32, batch every ordered basis pair and sixteen seeded dense integer pairs; compare against independent Definition 2.4 results. |
| `GPUVerification.test_03_to_06_multiply_hexagon_tables_and_all_sedenion_candidates` | In dimensions 16 and 32, replay all 196 table products for one double hexagon against the sparse module path. Then multiply every ordered pair among the 112 pure signed sedenion basis-pair candidates: 12,544 products, including both orders and XOR-incompatible negative cases. |

## What has not been established or implemented

| Paper content or extension | Limitation |
| --- | --- |
| Claims for arbitrary dimension and arbitrary real vectors | Finite computation cannot prove the universal statements. Exhaustive claims here always name a finite basis, graph, parameter set, or span. Seeded vector examples are samples, not real-vector enumeration. |
| Direct CayleyKernel verification of split/mixed signatures | The public routines have no doubling-parameter argument. Their signs implement gamma=−1 at every step. Reference tests cover selected split statements and all signatures through full dimension 16. The complex GPU supplement additionally checks transported basis/dense products through dimension 16; it is not a native split-signature kernel. |
| General nonunit gamma normalization/isomorphism in Section 2.3 | The reference supports nonunit parameters for Section 5 source tables, but no arbitrary-real normalization/isomorphism test is implemented. Such a test would require explicit rescaling maps and a policy for irrational square roots. |
| General-field graph definitions in Section 2.1 | Tests use real algebras and exact rational coordinates. They do not model arbitrary fields, characteristic two, or all abstract algebras. |
| Entire real zero-divisor graphs and their diameter | Only the finite projective basis-pair subgraphs are built. General vertices have arbitrary real coefficients and are not finitely enumerable. The introductory graph-distance notation supplies no numerical diameter theorem to test here. |
| Section 3 arbitrary-signature and isotropic branches | Ordinary examples receive broad testing. Reference supplements cover specific split self-orthogonality, a norm-zero branch, and a repeated-vertex issue. Lemmas 3.13–3.16 are not exhaustively exercised for all mixed parameters or arbitrary doubly alternative components. |
| Section 4 all normalized strongly alternating component choices | Twelve deterministic basis examples per tested dimension plus one rational example are covered. This does not exhaust every normalized real quadruple. |
| Lemma 5.8 all degenerate homomorphisms | Nondegenerate eight-dimensional images are tested; lower-rank examples are included for Lemmas 5.3 and 5.6. There is no separate general rank-deficient eight-generator verification for Lemma 5.8. |
| Lemmas 6.7–6.15 every signature and sign variation | Ordinary tests exhaust specified finite families. The mixed supplement checks classification and selected neighbor formulas, but does not exhaust both implications of all these lemmas, all larger signatures, or every sign pattern in the eight-independent-unit tests. Lemma 6.15's chi=+1 behavior is partly exercised by the all-candidate self-orthogonality checks; its complete positive neighbor rule is not separately exhausted. |
| Corollary 3.7 and Lemma 6.14(1) exactly as printed | Concrete discrepancies are preserved as counterexample tests. They must not be summarized as successfully verified theorems. |
| GPU execution in the recorded environment | Two tests skipped because CuPy was unavailable. A compiled library was found, but that alone does not demonstrate a working GPU product. No GPU execution or timing result is claimed. |
| Formal proofs and claims of novelty | No proof assistant development, independent literature review, or validation of cited papers is included. Computations provide evidence and finite counterexamples. |

The code adds a paper-specific test package and documentation rather than changing the module's mathematical routines. Existing unrelated working-tree changes were left in place.

## Coefficient-type coverage

The basis/dense oracle and complete hexagon/candidate batches also run typed device replays. `GPUVerification.test_06_complexified_all_signatures` adds both complex precisions for every +/- signature in dimensions 2,4,8,16, checking all basis products and eight dense pairs per signature through the generator-phase embedding. Exact real rank, isotropy and graph classification remain reference computations.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **31 tests: 29 passed, 2 skipped, 0 failures, and 0 errors**. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
