# 0704.0112v3: Placeholder Substructures III

Robert P. C. de Marrais, *Placeholder Substructures III: A Bit-String-Driven “Recipe Theory” for Infinite-Dimensional Zero-Divisor Spaces*.

Reviewed source: `/home/alika/Desktop/CayleyKernel/0704.0112v3.pdf`, 32 pages. The arXiv banner says v3, 22 November 2007; the document's title-page date and PDF creation metadata say 29 October 2018. This review identifies the actual supplied bytes by SHA256 rather than resolving that date discrepancy. Page numbers below are the printed page numbers. The full text was read; Figure 1 and the central counting/cookbook formulas were also inspected in rendered pages.

The computational core is unusually well suited to CayleyKernel: a table painted by bit rules can be compared independently with a table obtained from actual Cayley–Dickson products. The agreement is extensive, but there are qualifications to the printed preparation and unrestricted off-switch wording. See [FINDINGS.md](FINDINGS.md) for exact examples.

**Suite: 32 tests. Recorded CPU run: 29 passed; the 3 CUDA tests were not selected.** No `multiply()` call or CUDA-device probe was made locally. Your normal pytest command runs the GPU tests automatically when CUDA is available; there is no environment-variable opt-in.

## Code, data and reproduction

- [Paper-ordered tests](../../../tests/papers/arxiv_0704_0112/test_sections.py).
- [Exact product/table/recipe helpers](../../../tests/papers/arxiv_0704_0112/emanation.py).
- [Report and table-rendering runner](../../../tests/papers/arxiv_0704_0112/run.py).
- [Results with provenance](results/results.json), [complete test log](results/test-output.txt), [exact certificates](results/certificates.json), [six full signed tables](results/tables.json), and [count CSV](results/counts.csv).
- [Recomputed balloon-ride diagram](results/balloon-ride.svg) and [hide/fill diagram](results/hide-fill.svg). These are new renderings of computed tables, not copied paper images.

Run the CPU checks and save their results from the repository root:

```bash
python3 -m tests.papers.arxiv_0704_0112.run \
  --paper /home/alika/Desktop/CayleyKernel/0704.0112v3.pdf \
  --output docs/papers/0704.0112/results
```

On your CUDA machine, run all 32 tests normally:

```bash
pytest -v tests/papers/arxiv_0704_0112/test_sections.py
```

Or run only the three GPU tests:

```bash
python3 -m unittest \
  tests.papers.arxiv_0704_0112.test_sections.GPUVerification -v
```

Or run the whole suite with GPU opt-in and save a separate result set:

```bash
python3 -m tests.papers.arxiv_0704_0112.run --cuda \
  --paper /path/to/0704.0112v3.pdf \
  --output docs/papers/0704.0112/results-cuda
```

CPU checks use Python's standard library and the local CayleyKernel package. NumPy, CuPy, the compiled kernel, and a CUDA device are needed for GPU execution. The report runner selects only CPU tests unless `--cuda` is supplied. Ordinary unittest/pytest runs the GPU methods automatically, skipping only if CuPy, the compiled kernel, or a CUDA device is missing. `cpu` pytest markers prevent the repository's GPU fixtures from suppressing these exact CPU tests.

## What the terminology means here

For dimension `2**N`, the generator is `G=2**(N-1)`. A strut constant is an integer `0<S<G`. An **Assessor** at label `r` is the plane with basis `e_r` and `e_(G+(r XOR S))`. The two primitive diagonal lines have representatives

`e_r + e_(G+(r XOR S))` and `e_r - e_(G+(r XOR S))`.

A **DMZ** is a pair of these lines with zero product in both orders. An **Emanation Table (ET)** has one row/column for each `r=1,...,G-1` except `S`. A filled cell records a DMZ relationship, with content `P=r XOR c`. The main and opposite-strut diagonals are blank.

A **Box-Kite** has six Assessors, grouped into three pairs of strut opposites. Its twelve non-strut edges are DMZs. We enumerate the XOR-compatible three-strut configurations and verify all edges; we do not simply assume every 24 filled directed cells constitute a Box-Kite. The XOR-sail requirement matters: arbitrary induced octahedral subgraphs need not be the paper's structures.

The label order uses increasing smaller members of XOR-S pairs, then their partners in reverse order. This makes the other empty diagonal run from the upper right to the lower left. Signed table entries use:

| Entry | Meaning |
| --- | --- |
| `0` | blank |
| `-P` | marked cell; the two zero-dividing dyads have the same slope |
| `+P` | unmarked cell; opposite slopes |

All algebra indices in saved dictionaries are mathematical, zero-based indices. Only the product adapter translates to `basis()`'s signed **one-based** labels. `twist()` takes zero-based indices. Coefficients are exact integers; zero detection never uses floating point tolerance.

There are three separate computational paths:

1. An independently recursive doubling oracle, without `twist()` or `basis()`.
2. Actual sparse multiplication with `basis()`, plus a faster four-coefficient cell decision using `twist()`. The latter is checked against full products, not assumed correct.
3. The bit recipe, which calls neither product implementation. Literal preparation and our proposed correction are kept separate.

## Tests in the paper's order

Names below are exact executable method names. A passing counterexample regression means the printed statement needs qualification; it does not mean that statement has been verified.

### 1. The Argument So Far (pp. 1–7)

1. **`test_01_01_xor_signs_and_signed_labels_against_doubling`** checks all **87,380** basis pairs through dimension 256 against independent recursive doubling, including signed input-label combinations through dimension 32. This verifies the XOR index rule and sign convention, rather than only checking known zero products.
2. **`test_01_02_rules_zero_one_two_and_cyclic_positive_trips`** checks inherited cyclically positive triples, the generator rule `e_u e_G=e_(u+G)`, and all three Rule 2 transformations for generated triples at levels `N=3,...,7`. It checks each transformed triple cyclically, not only the first product.
3. **`test_01_03_assessor_diagonals_and_unsigned_viziers`** checks the three unsigned XOR relations for all valid Assessors at `N=4,...,7`. For the small strut constants it also verifies that an Assessor's two diagonals do not zero-divide each other and that a primitive diagonal squares to `-2`. The unsigned qualification follows the paper's explicit restriction on Viziers 1 and 3.
4. **`test_01_04_seven_sedenion_box_kites_and_edge_signs`** constructs the seven sedenion Box-Kites. Each has six vertices, twelve edges, six marked and six unmarked undirected edges, eight triangular faces, four XOR sails, and two faces consisting entirely of negative edges. It verifies the edge partition rather than only counting cells.
5. **`test_01_05_emanation_table_construction_and_actual_two_way_products`** checks every cell for every `S` at `N=4,5,6`: **31,092 cells**, including blanks. It verifies label completeness, strut-mirror order, table symmetry, the fast sign decision against full sparse products, both multiplication orders, and both simultaneous slope choices for filled cells. Explicit zero-product pairs are exported.
6. **`test_01_06_trip_count_two_step_and_number_hub`** checks full-table counts `24*Trip_(N-2)` for `S<=8` and higher powers of two through `N=7`. For the power-of-two Number Hub, it checks that the upper-left quadrant's absolute entries are the XOR multiplication table with the main diagonal blank.

**Scope not covered:** the recap refers to earlier papers, a slide deck, physical interpretations and more general composite zero divisors. This suite checks the primitive two-term family explicitly defined here. It does not re-prove Parts I/II in full, classify arbitrary dense zero divisors, or interpret zero divisors as physical singularities. Euclidean norms still exist; what a nonzero zero product obstructs is multiplicative norm composition.

### 2. Recursive Balloon Rides: Theorems 11–13 (pp. 7–16)

7. **`test_02_01_theorem_11_central_embedding_preserves_marks_and_blanks`** checks ten specified transitions, including `S=9,15,17,24,25,31,48,57,73`. For every old cell it compares `(r,c)` at level `N` with `(g+r,g+c)` at `N+1`, where `g` is the old generator. Signed content and blanks agree.
8. **`test_02_02_theorem_11_label_lines_become_dmz_contents`** checks that the new rows headed `g` and `g+S` contain the prior labels `u` and their partners `u XOR S` at columns `g+u`. It checks all old labels in four transitions. As the proof notes, label lines do not inherit an old edge mark; here their actual new signs come from products.
9. **`test_02_03_theorem_12_four_corners_zero_padding`** checks every prior-label cell in five transitions without shifting its low indices. The signed contents are unchanged. In paired label order these old labels occupy the four corners.
10. **`test_02_04_theorem_13_french_window_shutters_and_diagonals`** checks both one-sided shifts in six transitions. Filled entries gain `g` in magnitude with the same mark; main-diagonal blanks become marked `g`; strut-diagonal blanks become unmarked `g+S`; rule-hidden blanks remain blank. These four cases are counted separately in the results.
11. **`test_02_05_theorem_13_label_line_extensions_reverse_marks`** checks the predicted reversal between `cell(g,u)` and `cell(g,g+u)` for every old label in four transitions, including their expected magnitudes.

[The regenerated diagram](results/balloon-ride.svg) shows the finite `S=15`, `N=5,6,7` stages corresponding to Figure 1. It does not render an asserted infinite-dimensional limit. The external slides and the reference's Cesàro-fractal construction were not imported or independently audited.

### 3. Singleton High Bits and Modularity (pp. 16–20)

12. **`test_03_01_whorfian_formula_and_printed_sequence`** reproduces the formal sequence `0,3,19,91,395,1643,6699` for `N=4,...,10`, then checks every directed cell of `S=15` tables at valid levels `N=5,...,10` against 24 times those counts. It checks that adding `4**(N-4)` restores the maximal count. The `N=4` zero is a formal formula term; an ET with `8<S<16` is not admissible at that level.
13. **`test_03_02_muntin_and_diagonal_contributions_units`** constructs the near-solid-line labels for `S=15` at `N=5,...,8`, checks their count `2*mu`, and counts the non-overlap line contribution directly. It verifies `omega=24*mu*(mu+1)` **cells**, the remaining `delta=24*Trip_(N-3)` cells, and `(omega+delta)/24` Box-Kites. This exposes a factor-of-24 units issue in the displayed equation on p. 17.
14. **`test_03_03_explicit_box_kites_partition_edges`** enumerates actual XOR-sail Box-Kites in nine tables, including `S=15,N=7` and `S=57,N=7`. Every filled undirected edge belongs to exactly one enumerated Box-Kite. The complete list of **63** Box-Kites for `S=57,N=7` is saved, supplying the count left uncomputed on p. 25.
15. **`test_03_04_theorem_14_singleton_high_bit_and_tandem_shifts`** checks maximal singleton-high-bit strut constants `S=g+s`, `s=1,...,8`, at `N=5,...,9`, omitting the invalid endpoint `S=G`. Every cell agrees with the `(R or C or P) in {g,s}` rule. Each table has `6*(G-4)` filled cells and `G/4-1` Box-Kites.
16. **`test_03_05_theorem_14_recursive_harmonics`** checks all cells in three consecutive levels for each of `S=9,15,17,24,33,40`, including the rule's repeated multiples/residue targets at higher levels.
17. **`test_03_06_density_limit_and_finite_degree_histograms`** computes exact finite densities from the established count formula for `N=5,...,14` and their gaps to `5/8`. It also computes the actual assessor-graph degree distributions at `N=5,...,8`. Those finite graphs each have only two degree values. Neither density nor visual self-similarity proves a power-law degree distribution or supplies a fractal dimension.

The tournament-dice suggestion is not coded as a result: the paper offers numerical resemblance, without a dice construction to test. No conclusion about a new tournament solution follows from matching the Box-Kite sequence.

### 4. Hide/Fill Involution: Theorems 15–16 (pp. 20–25)

18. **`test_04_01_chingon_hide_fill_counts`** checks all 31 strut constants in dimension 64. It reproduces 35 Box-Kites for full tables, 19 for `9<=S<=15`, 7 for `17<=S<=24`, and 23 for `25<=S<=31`.
19. **`test_04_02_theorem_15_restricted_two_bit_off_switch`** checks four examples where all old labels are below the newly added bit. Every old filled cell then becomes blank. This is the range in which the proof's additions to upper indices agree with XOR toggling.
20. **`test_04_03_theorem_15_unrestricted_wording_counterexample`** saves exact DMZ pairs for `r=2,c=8` at `N=5`, both before and after `S=1` becomes `S=9`. The cell remains filled. The unrestricted reading of Theorem 15 needs a label-range qualification; the next-bit operation does not turn off every old cell.
21. **`test_04_04_hidden_box_octonion_clones`** constructs four hidden-triangle examples at `S=9,N=5`. Each eight-basis-index set is XOR-closed. For each it constructs an explicit signed basis map from the standard octonions and verifies **every multiplication-table entry**. It separately verifies every coefficient of the norm-composition identity on that eight-dimensional subspace. These are genuine zero-divisor-free subalgebras for arbitrary real coefficients, not merely basis units whose pairwise products happen to be nonzero.
22. **`test_04_05_theorem_16_hat_trick_restoration`** checks three explicit low-label families with `S<L<M<G`: appending `L` hides old filled cells; appending `M` restores them. The examples include separated bits, not only adjacent bits. Restoration is checked as filled status; no unproved general sign-invariance claim is added.
23. **`test_04_06_rule_priority_and_paper_examples_25_41_57_73`** verifies the printed rule targets and first-rule precedence for all cells of the four highlighted tables. It records rule overlaps, which rule first decides each cell, and final counts. In particular, `S=57,N=7` has **1,512 filled directed cells**, 756 edges and 63 Box-Kites. The paper's 3,720 cells there are the admissible off-diagonal positions, not the filled count.

The stated band extension for every multiple of 8 needs care beyond the worked Chingon bands: exact products give different counts at `S=47` and `S=48` in dimension 128. The finite discrepancies and a proposed preparation repair are tested next. No links to E8, a theory of everything, or many-worlds physics are inferred from the verified octonion clones.

### 5. Fundamental Theorem and Cookbook (pp. 25–30; references pp. 31–32)

24. **`test_05_01_proposition_inner_skybox_and_613_bit_preparation`** checks the unique finite dimension bracket for eligible integers from 9 through 4096. It reproduces the example `613=0b1001100101`, inner level `N=11`, and high-bit exponents `[9,6,5]`.
25. **`test_05_02_cookbook_literal_preparation_exhaustive_finite_audit`** compares the printed preparation, interpreted with ordinary modular residues, against every valid cell for every `S` at `N=4,...,8`: **2,289,516 comparisons**. It records all mismatch counts and first witnesses rather than requiring the printed recipe to pass. Non-power-of-two failures include `S=48,80,96,112`; power-of-two controls are recorded separately by their explicit values. Both products for the false filled cell `S=48,r=1,c=2` are saved. It also checks the `S=47` versus `S=48` band-boundary counts.
26. **`test_05_03_proposed_trailing_bit_correction_exhaustive_finite_audit`** repeats that complete finite audit with our explicitly separate preparation: ignore bits below exponent 3, and also ignore the actual least significant set bit if that bit is 8 or higher. There are **zero mismatches** across the same 2,289,516 cells. This is a finite verification of a proposed correction, not a proof of the full infinite recipe or a silent attribution of our rule to the author.
27. **`test_05_04_recipe_613_seeded_large_bit_queries`** checks 9,000 deterministic sample cells: `S=613` at `N=11,12`, multiple-of-16 controls at `N=10`, and a 201-bit strut constant at `N=203`. Only individual sparse queries are evaluated at that enormous formal dimension; no `2**203` array is allocated. Ten sampled cells per case also receive full sparse-product and independent recursive-sign controls for both column slopes.
28. **`test_05_05_literal_mutual_orthogonality_and_repeated_lines`** shows that one assessor diagonal participates in four different filled cells in a sedenion ET. Cells encode relationships, not a fresh algebra direction for every cell. This limits a literal Euclidean reading of “one mutually orthogonal line per cell” in the final proposition; an enlarged space with a specified inner product would be needed for that interpretation.
29. **`test_05_06_triadic_relation_join_example`** implements the explicit finite relation operation on p. 29: `(a,x,z) in R`, `(x,b,y) in S`, `(z,y,c) in T` produces `(a,b,c)`. It checks a four-result example, empty input, monotonicity, and closure as triples. This tests the mathematical definition actually given, not the proposed database architecture.

### CUDA tests written for you, not run here

30. **`test_06_01_multiply_all_basis_pairs`** batches every basis pair at `N=1,...,7` through `multiply()` and compares each coefficient against recursive doubling.
31. **`test_06_02_multiply_complete_emanation_tables`** checks every dyad product for both column slopes in four representative complete tables, including `S=57,N=7`. Both multiplication orders, filled cells, hidden cells, and diagonal products are compared with independent recursive products. Expectations include complete nonzero outputs, not only a zero/nonzero mask.
32. **`test_06_03_multiply_zero_divisor_scaling_and_recipe_counterexample`** checks a scaled zero-product witness and the two nonzero products in the falsely painted `S=48` cell. This guards against both false negatives and false positives in the kernel. All coefficients are small exactly representable float64 integers.

## What cannot be established by these computations

| Topic in the paper | Status and reason |
| --- | --- |
| Theorems 11–16 for all levels and all labels | Finite complete tables and explicit ranges are tested. An induction or symbolic proof for the unrestricted domains is not supplied. |
| Fundamental proposition's infinite “Sky” | No completed infinite table, topology, measure, or limit construction is implemented. Sparse queries at large bit lengths remain finite calculations. |
| Identification of a particular classical fractal, fractal dimension | The finite masks are regenerated. A precise limiting metric/normalization and a reference construction are needed for a rigorous identification. |
| Scale-free networks | Degree histograms are recorded; a scaling law for the specified graph is not established. Visual recursion alone does not prove one. |
| Wolfram Class 4 complexity, changing chaotic attractors | No transition dynamics, initial conditions, or operational classification criterion is specified in the article. |
| Neuroscience, semiotics, physics/E8, adaptive parsing, database performance | These are motivating comparisons or future work here. No falsifiable executable model or benchmark is supplied; the finite trijoin definition is the exception coded above. |
| Referenced slides, earlier papers, tournament dice and external literature | Not fetched or independently audited. The supplied PDF is the source under review. |
| GPU correctness | Tests are provided for your machine. No CUDA execution has been performed or claimed. |

The PDF, module and test source hashes are saved with the data. Complete signed tables, explicit product witnesses, signed octonion isomorphisms, and the 63-Box-Kite list make the substantive finite results reviewable without relying on the rendered pictures or a “tests passed” statement alone.

## Coefficient-type coverage

All basis-pair, complete emanation-table, reverse-order, scaled zero-divisor and recipe-counterexample device products also run the typed replays. Exact integer table construction and combinatorial counts keep their original arithmetic.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.
