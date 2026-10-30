# Follow-up experiments: XOR octonions, basis-unit centralizers and particle irreps

This is a **separate follow-up** to [the original paper verification](../README.md), with its own [test suite](../../../../tests/papers/physletb_141078/test_extensions.py), [runner](../../../../tests/papers/physletb_141078/run_extensions.py), data and analysis. It addresses the three further investigations requested after the initial review.

The original paper is N. Furey, *Standard model symmetries and the nested embeddings of R ⊂ C ⊂ H ⊂ O*, Physics Letters B 882 (2026) 141078, [DOI](https://doi.org/10.1016/j.physletb.2026.141078). Relevant starting points are Sections 3-6 (multiplication algebras and centralizers), 8 (symmetries), and 9.3/9.6 (operator blocks and Hopf geometry).

Reference [7] is now available and was used: N. Furey, *A Superalgebra Within*, [arXiv:2505.07923v4](https://arxiv.org/abs/2505.07923v4), dated 9 June 2026, particularly Sections VI-VIII, Eqs. (23)-(26),(30),(37)-(44), and the particle tables on pages 10-12. The reviewed PDF was saved as `2505.07923v4.pdf`, SHA-256 `11aa3c8f5888f742ec3f64f657eb8974551a31759d817c9567d89b5271f11957`. Page 6 and pages 9-12 were inspected visually as well as through text extraction. PDFs remain outside Git.

## Recorded outcome

**14 tests: all 14 passed, with no skips, failures or errors, in 38.71 seconds.** The saved run enabled GPU coverage through level 20. It completed 202 products in 2D batches, including full arrays with 1,048,576 components. CUDA device access required running outside the execution sandbox; an initial sandbox run had skipped that test. The new GPU test is separate from the two original GPU tests that passed in the user's earlier run. The [saved follow-up report](results/results.json) records the configuration, environment, source hashes, PDF hashes and exact matrices.

The follow-up establishes three useful results:

1. Exhaustively classified **898,567** eight-dimensional XOR subspaces through level 9; **124,843** retain octonion norm composition. Every passing subspace is listed in a machine-readable CSV.
2. Compared **2,052 complete basis-unit profiles**, including every imaginary basis unit through level 10. The selected basis unit did not change the joint-centralizer size in any tested case.
3. Reconstructed explicit internal irreducible particle-block representations from reference [7], with a necessary, explicitly documented correction to one printed color-generator sign. The literal printed formula is retained as a discrepancy regression, not quietly replaced or reported as verified.

These are computational results and certificates for the scopes stated below. We do not claim they are new to the mathematical literature.

## Run the separate script

Use Python 3.10 or later with NumPy and SymPy, from the Git project root:

```bash
python3 -m tests.papers.physletb_141078.run_extensions
```

Save a new run without overwriting existing evidence:

```bash
python3 -m tests.papers.physletb_141078.run_extensions \
  --paper /path/to/CERN-main.pdf \
  --reference /path/to/2505.07923v4.pdf \
  --output /tmp/cayley-furey-followup-new
```

To enable batched GPU checks through level 20:

```bash
python3 -m tests.papers.physletb_141078.run_extensions \
  --gpu-max-level 20 --output /tmp/cayley-furey-followup-gpu
```

Default GPU coverage stops at level 12. The level-20 option submits full arrays with 1,048,576 components, using batch size two to bound memory and work. Lower levels use larger batches: 64 at levels 4,8,12 and eight at level 16. Each level includes both a composition subspace and a noncomposition subspace. At level 4 there are only 15 distinct subspaces, so a batch may repeat a subspace with different coefficients. No GPU batch substitutes a CPU fallback for `multiply()`.

For a quicker finite survey, use `--max-exhaustive-level 6`. The default is 9. Level 10 is an optional much larger exhaustive run and was **not** executed for the saved report. `--high-samples` controls distinct sampled subspaces per high level; `--max-unit-level` controls the highest level at which every imaginary unit receives a complete column scan.

Classify your own three XOR-independent basis indices without running the survey:

```bash
python3 -m tests.papers.physletb_141078.run_extensions --classify 1,2,4
python3 -m tests.papers.physletb_141078.run_extensions --classify 1,2,12
```

The first is an octonion example. The second returns a concrete zero-divisor counterexample. Decimal and `0x` hexadecimal indices are accepted. The criterion works for arbitrary-width indices; it does not allocate an ambient vector of length 2^level.

Repository integration:

```bash
python3 -m pytest -q tests/papers/physletb_141078/test_extensions.py
```

## 1. Which eight-dimensional XOR subspaces retain norm composition?

Here a subspace means a **unital coordinate subalgebra**

```text
W = span_R { E_a : a in U },  U = span_F2 { b1,b2,b3 }.
```

The three integer indices are XOR independent, so U contains eight indices including zero. Closure follows from `E_a E_b = twist(a,b) E_(a XOR b)`. This class of subspaces does not include every arbitrarily rotated eight-dimensional subspace, nor affine XOR cosets that omit zero.

For each W, the code builds its full 8-by-8 sign table with CayleyKernel's `twist_many()`. Let L_i be the restricted multiplication matrix of its ith imaginary basis element. Signed permutation matrices give `L_i^T=-L_i` and `L_i^T L_i=I`. The remaining exact Hurwitz equations are

```text
L_i L_j + L_j L_i = 0,  1 <= i < j <= 7.
```

The code checks all eight columns of all 21 equations. These polarized equations are necessary and sufficient for `N(x*y)=N(x)*N(y)` for **every pair of real coefficient vectors in that fixed subalgebra**. Thus classification is not based on random coefficient tests. Independent symbolic norm expansions validate positive and negative representatives; signed algebra isometries certify all observed table types.

### Complete finite classification

Each binary subspace is enumerated once using reduced row echelon form. Enumeration totals match the Gaussian binomial coefficient `[n choose 3]_2`.

| Level n | Ambient dimension | All XOR subspaces | Composition | Noncomposition |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 8 | 1 | 1 | 0 |
| 4 | 16 | 15 | 8 | 7 |
| 5 | 32 | 155 | 50 | 105 |
| 6 | 64 | 1,395 | 310 | 1,085 |
| 7 | 128 | 11,811 | 2,046 | 9,765 |
| 8 | 256 | 97,155 | 14,478 | 82,677 |
| 9 | 512 | 788,035 | 107,950 | 680,085 |

[composition_subspaces.csv](results/composition_subspaces.csv) lists **every passing subspace** at these levels, with its level and three canonical generator indices. The convention is low-bit-to-high-bit binary columns, with each row's pivot its least set bit. Rows with free columns greater than their pivot give a unique representative. Alternative generating triples describe the same subspace; the classifier accepts them directly.

All observed multiplication tables in the complete survey and high-level samples fall into **64 coordinate sign patterns**, comprising eight composition patterns and 56 noncomposition patterns. Explicit signed XOR-linear isometries identify the eight composition patterns with one standard octonion table, and the 56 others with one noncomposition table. The certificates check **all 64 basis products**, so they preserve products and the Euclidean norm. This is a two-type classification of the observed tables, not an unproved claim that sampling exhausts every possible higher-level table.

### A failure that is stronger than a small norm error

The subspace generated by indices `1,2,12` contains

```text
U = {0,1,2,3,12,13,14,15}.
x = E1 + E12
y = E2 + E15
x*y = 0,  N(x)=N(y)=2.
```

It is eight-dimensional and closed, yet has zero divisors and cannot be a normed division octonion algebra. All observed noncomposition table patterns admit such a two-term zero-divisor witness. This shows why dimension eight and XOR closure alone are insufficient.

### High levels

At levels 12,20,32,64,128,256,1024, the report records 256 distinct sampled subspaces per level. Their generators use genuine high-bit indices rather than just the low embedded octonions. Each sampled subspace is classified exactly for all coefficients within it. The sampling does **not** classify every subspace of the ambient high-dimensional algebra.

The first eight sampled tables per level also have every product compared with the independent iterative Cayley-Dickson doubling oracle. Level 1024 means 1024-bit indices; no array of length 2^1024 is constructed.

## 2. Do other basis units change the enlarged centralizer?

For every unit under comparison, the code checks all columns of L_u, R_u, their squares and their products. Coverage is every imaginary basis unit at levels 3-10, plus ten selected units at each of levels 11 and 12. The choices include small indices, sums of bits, units adjacent to the doubling generator and the highest index.

All 2,052 complete profiles satisfy

```text
L_u^2 = R_u^2 = -I
L_u R_u = R_u L_u
rank((I-L_u R_u)/2) = 2
rank((I+L_u R_u)/2) = d-2.
```

The two simultaneous complex blocks therefore have complex dimensions 1 and `d/2-1`. Their full joint centralizer in End_R(A) has real dimension

```text
2 + (d-2)^2/2,
```

and its skew part has half that dimension. These are centralizers in the **full vector-space endomorphism algebra**, not a claim about a smaller selected multiplication subalgebra.

| Ambient dimension d | Full joint centralizer | Skew joint centralizer |
| ---: | ---: | ---: |
| 8 | 20 | 10 |
| 16 | 100 | 50 |
| 32 | 452 | 226 |
| 64 | 1,924 | 962 |
| 128 | 7,940 | 3,970 |
| 256 | 32,260 | 16,130 |
| 512 | 130,052 | 65,026 |
| 1,024 | 522,244 | 261,122 |

An independent signed-graph solver imposes the commutator equations on **every matrix entry**, including the transpose constraints for skew matrices. It verifies the dimensions for every imaginary basis unit in dimensions 8,16,32. General rational matrix elimination provides an additional independent comparison for units E1,E3,E8,E15 in dimension 16.

The [basis-unit CSV](results/basis_units.csv) lists every complete profile and dimension. At still higher levels, 16 selected units per level 20,32,64,128,256,1024 are checked against 68 sampled columns each, including independent recursive signs. Those are samples, not complete profiles or full enormous matrix-equation solves.

**Interpretation:** choosing a different basis unit did not reduce the sedenion centralizer back to the octonion size. The block enlargement is persistent across the tested basis choices. Keeping the original embedded octonion sector is different from treating the entire larger algebra as the color sector.

## 3. Explicit internal particle representations

See [PARTICLES.md](PARTICLES.md) for the reconstruction, the source discrepancy, exact representation labels, and what the calculation does and does not establish.

Unlike the earlier dimension-only Peirce calculation, this work constructs the reference's octonion ladder operators, vacuum, five projectors, color and weak generators, hypercharge operator, quaternionic fourth ladder operator, and the actual particle operator bases. It exports explicit matrices to [results.json](results/results.json), including the restricted generator action for every internal irreducible block.

## Tests actually implemented

All names below follow `test_` in [test_extensions.py](../../../../tests/papers/physletb_141078/test_extensions.py).

| Test | Verification |
| --- | --- |
| `01_enumeration_and_exact_composition_criterion` | Unique finite enumeration; canonical basis independent of generator choice; rejection of impossible distinct sample sizes; symbolic norm polynomial for both table types; required skew/orthogonality identities. |
| `02_exhaustive_xor_classification` | Every rank-three binary subspace at levels 3 through the selected maximum (9 in the saved run); exact Hurwitz classification and Gaussian enumeration totals; retain every passing generator triple. |
| `03_high_bit_xor_classification_and_oracle` | Distinct high-bit samples at seven high levels; exact composition criterion; independent doubling on eight complete tables per level; CPU validation of positive/negative GPU batch selection and batch sizes. |
| `04_signed_isometry_classes_and_counterexamples` | Signed relabeling certificates for every observed table pattern; all 64 products per certificate; explicit zero divisors for every observed noncomposition pattern. |
| `05_all_basis_units_complete_columns` | Complete-column unit profiles, projector ranks and commuting complex structures for 2,052 unit/level combinations. |
| `06_independent_matrix_entry_centralizer_solve` | Full and skew matrix-entry constraints for every basis unit in dimensions 8,16,32; rational-matrix comparisons in dimension 16. |
| `07_high_bit_nondoubling_units` | Independent recursive signs and complex-structure identities at sampled columns for high-bit units through level 1024. |
| `08_reference_literal_color_generator_discrepancy` | Reproduce the printed nonzero trace and nine-dimensional generated color algebra; verify the separate minimal sign repair restores a traceless eight-dimensional algebra. A passing discrepancy regression means the literal statement failed its mathematical requirement. |
| `09_reference_generators_match_original_paper` | Repaired reference color span equals the projection of the original paper's independent Eq. (8); all generator/projector commutators, Casimirs, color/weak mutual commutation and explicit charge-convention relation. |
| `10_clifford_ladders_vacuum_and_peirce_projectors` | Full four-mode canonical anticommutation relations, Hermitian rank-one vacuum and orthogonal projectors with complex ranks 1,3,2,1,1. |
| `11_explicit_particle_ladder_bases` | Construct all ten off-diagonal families with four beta left/right excitations; full independent spans, correct source/target support and Hermitian conjugates; 192 real off-diagonal degrees of freedom. |
| `12_certified_internal_irreducible_representations` | Exact restricted color/weak/Y/Q actions, electric charges, intertwining with actual matrix commutators, scalar commutants, quadratic Casimirs and color cubic trace distinguishing 3 from 3bar. Export exact matrices and bases. |
| `13_diagonal_adjoint_representations_and_missing_top` | Diagonal color adjoint 8 and weak adjoint 3 with singlet complements; 64 diagonal real dimensions; location and absence of the top-containing internal irreps in the inner sector. |
| `14_batched_composition_and_noncomposition_products` | Actual 2D-array GPU `multiply()` batches replay independent products from both types; configured through level 12 by default or level 20 by option. Passed through level 20 in the saved follow-up, using CUDA access outside the sandbox. |

## Saved artifacts and reproducibility

- [results.json](results/results.json): source/PDF hashes, settings, all outcomes, sign-table classes and isometries, counterexamples, sampled classifications, centralizer checks, explicit particle states/operators and restricted irrep matrices.
- [composition_subspaces.csv](results/composition_subspaces.csv): 124,843 passing canonical subspaces in the complete finite survey.
- [basis_units.csv](results/basis_units.csv): 2,052 complete basis-unit profiles.
- [extension_algebra.py](../../../../tests/papers/physletb_141078/extension_algebra.py): classification criterion, enumeration, isometries and independent signed-graph centralizer solver.
- [particle_reconstruction.py](../../../../tests/papers/physletb_141078/particle_reconstruction.py): explicit reference formulas and representation matrices.

The original review's historical results are preserved. The new reference discrepancy concerns reference [7]'s printed Eq. (26); it does not contradict the main paper's independently verified Eq. (8). No module implementation was changed for this follow-up.
