# An alternate Cayley-Dickson product

John W. Bales, Missouri Journal of Mathematical Sciences 28(1), 2016,
printed pp.88-96 (nine PDF pages). Input name: `1474295358.pdf`.

The important distinction is the multiplication convention: this paper uses
P2 with a shuffle basis. CayleyKernel implements P3 transpose with a block
basis. Their signs are not expected to agree at identical integer labels.
The paper's closed sign formula is implemented separately and checked against
its own doubling definition and its twist tree. The module is checked against
an independent P3-transpose doubling oracle. These comparisons do not supply
the paper's asserted isomorphism between all eight products.

## Reproduce

```bash
python3 -m pytest tests/papers/desktop_batch/test_1474295358.py -v
```

Complete outcomes/provenance are in `results/`. Code is
`tests/papers/desktop_batch/test_1474295358.py`.

## Section-ordered inventory of every coded test

| Section | Test suffix | What is computed |
| --- | --- | --- |
| 1-2, printed pp.88-90 | `01_02_standard_product_and_shuffle_conventions` | All 4096 module basis products in dimension 64 against independent block doubling, including signed one-based basis-label conversion and twist. Require actual sign differences from P2; test P2 identity and all imaginary squares through dimension 128. |
| 3, p.90 | `03_equations_3_1_3_2_all_pairs_through_1024` | All 1,048,576 ordered index pairs in dimension 1024: closed formula vs independent P2 doubling using least-significant-bit shuffle recursion. Covers diagonal, zero, reversed, same leading-bit block and different-block cases. |
| 4, pp.91-93 | `04_tree_and_printed_examples` | Full 128x128 twist tree vs independent doubling; all four worked products (3,14),(35,55),(87,340),(51,12), including XOR output indices and signs. |
| 5, p.94, extension | `05_theorem_high_indices_and_fixed_bit_reversal` | Six index widths 16,20,32,128,1024,4096, 16 seeded pairs each: closed formula, tree and doubling agree. Also all 128x128 pairs check XOR compatibility of fixed-width bit reversal, the coordinate reindexing from shuffle to block. This last identity is not a P2-to-P3 sign isomorphism. |
| Device check | `06_gpu_standard_convention` | Direct `multiply()` for every ordered basis pair in dimension 16, compared with exact module sparse expectations. It tests the standard convention, not P2. |

`omega2` encodes (3.1)-(3.2); `doubling2` follows the four P2 pair branches
and does not call the formula/tree/module. `tree2` follows Figure 2 from most
significant doublets. Thus the central test has independent expectations.
High index tests allocate only Python integers, not vectors of dimension
2^4096. This is deeper sign-law testing, not a dense multiplication benchmark.

## Limits and unimplemented material

- The eight products are displayed in Section 1, but only P2 and the module's
P3-transpose convention are independently implemented. Exhaustive tests of all
eight, their transpose pairs and explicit signed-coordinate isomorphisms are
not present. Bit reversal alone must not be mistaken for the full isomorphism.
- Figures 1,3,4 are finite sign tables whose entries are covered at dimensions
128,32,1024. Pixel-by-pixel comparison to the printed images is not performed.
Figure 5 is P3 (not its transpose); no reproduction of that bitmap is claimed.
- Section 4's text attributes the same-block (35,55) example to (3.2), though
its conditions are those of (3.1); the worked product itself checks correctly.
- Theorems at unbounded level need proof, not a finite test. CPU samples
through level 4096 support the implemented formula, with no measured claim
about GPU cost at those levels. No assertion about speed or historical
priority is tested.
- GPU tests explicitly skip if prerequisites are missing. `basis_product` is
the shared alias for this checkout's actual public `basis()` routine.

## Later-version qualification

The supplied 1707.07318v4 companion distinguishes P1/P2 and their transposes
as pseudo products, with zero divisors at dimension eight. Its finite
alternativity and zero-product checks support that distinction. Thus the
earlier paper's general claim that all eight give isomorphic Cayley-Dickson
algebras must not be inferred from the sign/periodicity checks here. See
[the later paper report](../1707.07318v4/README.md).

## Coefficient-type coverage

All standard-convention device basis-pair fixtures also run typed replays. This does not supply a missing P2/P3 coordinate isomorphism; the alternate product and twist-tree tests retain their original exact convention.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **5 tests: 5 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:24.041411+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
