# Structure of the Cayley-Dickson algebras

G. P. Wilmot, arXiv:2505.11747v3, 6 February 2026.
The paper classifies basis triples by three associator orderings, groups
them into cycles, counts eight-dimensional subalgebras and finite two-term
zero products, and studies split signatures. These are direct module tests.
The finite zero-divisor counts describe particular **basis-pair families**,
not all zero divisors with arbitrary real coefficients.

```bash
python3 -m unittest tests.papers.desktop_batch.test_2505_11747v3 -v
```

Code: `tests/papers/desktop_batch/test_2505_11747v3.py`; exact count tables,
split counts, outcomes and provenance in `results/`.

## Every coded test in section order

| Section | Test suffix | Scope |
| --- | --- | --- |
| 1 | None | Motivation and terminology; no historical/physics validation. |
| 2 graded construction, blade laws | `02_graded_basis_and_blade_laws` | Levels 1-8: every imaginary basis square and ascending-generator extension; all distinct basis pairs verify anticommutation and both repeated-basis associativity directions. |
| 3 Theorem 5 / Table 2 | `03_1_theorem_5_and_table_2_triads` | Every naturally ordered distinct positive-basis triple at levels 2-7 (through 333,375 triples in dimension 128). Only H,A,B,C,X patterns occur; associative counts match (d-1)(d-2)/6 and totals match binomial(d-1,3). All type distributions saved. |
| 3 Theorem 7 / Table 4 | `03_2_theorem_7_cycle_silos` | Levels 3-6: enumerate exactly b<c<b XOR c<d and classify all three cycle members. Complete agreement with the eight allowed silos and the first four printed columns. |
| 3 Table 5 | `03_3_table_5_subalgebra_counts` | All three-dimensional XOR subspaces at levels 3-6; classify every contained triad. Counts O/P4/P12/P14 reproduce (1,0,0,0),(8,7,0,0),(50,63,42,0),(310,413,504,168). No reliance on the claimed recurrence to generate the counts. |
| 3 Malcev discussion | `03_4_actual_malcev_identity_vs_associative_exclusion` | Actual normalized-commutator Malcev identity for all quaternion basis triples. It holds, so the statement that associative algebras are not Malcev cannot be applied to the usual commutator definition. Simplified equation (10) has narrower triad assumptions. |
| 4 Theorem 10 / Table 14 | `04_1_theorem_10_counts_and_complete_expanded_products` | Levels 3-7: enumerate all distinct pure two-term supports compatible with XOR cancellation, test **all four** relative-sign combinations. Support-pair counts 0,84,1260,13020,117180 agree with the formula. All signed products through level 5 replay via `basis()` in both orders. |
| 4 Table 8 / modes | `04_2_table_8_all_prime_and_mode_products` | All 21 printed prime rows and all four mode constructions, both multiplication orders. Graded digits map to bitmask basis labels. Verify a=(bc)d under this table's convention. |
| 5 Theorem 12 / selected split counts | `05_split_trace_and_selected_pair_counts` | All signatures through level 5: pure basis-square trace is 1 whenever any parameter is +1. Independent reference counts for all-positive split octonions/sedenions and last-positive sedenions. Split counting allows opposite-square supports; see distinction below. |
| Extension | `06_extension_high_index_type_patterns` | 128 seeded triples per index width 20,32,128,1024,4096: no forbidden associator pattern. These are sparse high-bit samples, not exhaustive high-dimensional triad counts. |
| Device replay | `07_gpu_table_8_example` | Direct `multiply()` of the first Table 8 zero-product pair; explicit dependency/device skip. |

## Counting conventions and discrepancies

The paper's ordinary counts quotient the two admissible relative-sign
patterns on each unordered pair of two-term supports. If one instead counts
signed projective vertices and undirected edges, the ordinary counts double:
dimension 16 has 84 support pairs but 168 signed edges. The runner saves both
numbers. Zero support/index cancellation is necessary but never assumed
sufficient: both scalar coefficients of the expanded product are tested.

The prose below Table 9 says sedenion types are H=35,A=84,C=112,X=224.
Direct enumeration gives H=35,A=84,C=84,X=252, consistent with combining
the cycle and non-cycle entries of Table 4. A regression preserves this
disagreement rather than marking that sentence as verified.

For all-positive split octonions the enumeration gives **12 support pairs
and 24 signed edges**, while Table 14 prints 24 and the Section 5 prose
mentions 12. The split sedenion entries 180 and 84 agree as support counts
(360 and 168 signed edges). Thus the table does not consistently use the
same quotient in its split-octonion row. Section 4's same-square requirement
is unsuitable for all split examples: in a composition split algebra a
two-term isotropic vector needs opposite-square components. The split
enumeration explicitly allows that, while keeping all four basis labels
distinct. It does not count scalar-containing idempotents or identical-support
conjugate pairs.

## Not coded / limits

No infinite induction proof of the cardinality/subalgebra recurrence is
claimed. Theorems 2-4/6 and every Table 1 associator equivalence, Moufang
equivalence, simplified Malcev rule and Section 4 cycle/mode completeness
proof are not individually transcribed. The finite type/cycle/subspace
enumerations check selected consequences independently. Table 8's prime and
mode products are checked, but the db/ab text columns, Tables 9-13 and
Table 15's complete split-mode catalogue are not separately transcribed.

No general real zero-divisor classification, arbitrary annihilator dimension,
all split isomorphisms of Table 16, Lie/G2/particle-physics claim, or naming
proposal is validated. Split calculations use parameterized independent
doubling; the module has no signature parameter. Counts at the paper's
larger levels 8-10 are not exhaustively rerun. The extension instead checks
individual high-bit basis triples through level 4096. No huge dense vector
allocation or GPU timing is claimed. The mathematical oracle remains the
module's standard convention; independent twist/doubling comparisons are in
the Ren-Zhao companion and existing project suites.

## Coefficient-type coverage

The Table 8 zero-product device fixture also runs typed replays. `test_08_gpu_complex_split_signatures` adds all +/- signatures through dimension 16 in both complex precisions, comparing all basis products and eight dense pairs per signature via the generator-phase embedding. Triad/graph counts remain counts of the original signed basis families, not all complex zero divisors.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **10 tests: 10 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:39.151542+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
