# The twisted group algebra structure of the Cayley-Dickson algebra

Guangbin Ren and Xin Zhao, arXiv:2205.07986v2, 13 March 2023, 13 PDF pages.
This directly tests the module's central sign law. The paper writes a basis
product as (-1)^sigma(A,B) e_(A XOR B). Its degree means the **least**
significant set-bit position, not the usual highest-bit degree.

```bash
python3 -m unittest tests.papers.desktop_batch.test_2205_07986v2 -v
```

Code: `tests/papers/desktop_batch/test_2205_07986v2.py`. Outcomes, witnesses,
equation mismatch pairs, high-index coverage, source/PDF hashes: `results/`.

## Tests in section order

| Section / statement | Test suffix | What is checked |
| --- | --- | --- |
| 1 introduction | Later Theorem 1 tests | Formula (1.1) is restated as (4.1); no separate history/priority test. |
| 2 (2.1)-(2.14), Lemma 1 | `02_doubling_generator_relations_and_involution` | Levels 1-6, eight seeded pairs of dense integer component vectors per level: generator movement and five identities (2.7),(2.9)-(2.12). Every left-associated ascending-generator monomial matches its basis label. All arithmetic uses module basis products with exact integers. |
| 3 Proposition 1 (3.3) | `03_2_proposition_1_recursive_parity` | Lower levels 0-6, every lower basis pair and all four top-bit choices; recursion agrees with independent explicit formula. |
| 3 twisted algebra structure, extension | `03_1_twisted_group_is_not_associative_cocycle` | All octonion triples find nonzero associator parity; all 4096 quadruples check the associator's coboundary identity. Prevents incorrectly assuming the twist is an associative group-algebra 2-cocycle. |
| 4 Lemma 4 / Theorem 1 | `04_1_lemma_4_and_theorem_1_exhaustive` | **All 1,048,576 ordered basis pairs in dimension 1024**: literal bit-by-bit (4.1) vs `twist()`, signed one-based `basis()` and independent iterative doubling; Lemma 4 bit conditions checked wherever A,B are positive and distinct. |
| 4 Example 1 / (4.11) | `04_2_example_1_and_equation_4_11_discrepancy` | Correct degrees and product for A=5,B=6; exhaust all 42 distinct positive octonion pairs and save those where printed simplified (4.11) disagrees with (4.1). Passing means the discrepancy was reproduced. |
| 5 Proposition 2 / Theorem 2 (5.7) | `05_split_last_parameter_and_proposition_2` | Every pair in dimensions 2-64 against independent parameterized doubling with only the last parameter +1. The correction flips signs exactly when both top bits are set. Split-complex (1+g)(1-g)=0 supplies a negative norm/division control. Reference-only: the module does not accept split parameters. |
| Beyond paper | `06_extension_high_indices_and_padding` | 24 seeded pairs at 20,32,128,1024,4096-bit index widths: formula, independent iterative doubling, module twist and negative signed basis labels agree. Sparse integers, not dense 2^4096-dimensional vectors. |
| GPU supplement | `07_gpu_basis_and_dense_doubling` | Direct `multiply()` on all dimension-16 ordered basis pairs plus 16 seeded dense integer pairs, compared with exact sparse module expectations. Explicit skip if CuPy/device/library is absent. |

## Findings

The central formula (4.1) agrees with the module and independent doubling
through the finite exhaustive range and high-index samples above.

Example 1 incorrectly states deg(5)=deg(6)=2 and deg(5 XOR 6)=1. Its own
definition gives 0,1,0, hence l=1. The final product -e3 is correct even though
the explanation is not. Formula (4.11) uses the bit at position 2 and includes
all three phi terms regardless of l; it does not reproduce every octonion
basis product. The report saves concrete mismatch pairs rather than treating
this display as verified. The corrected implementation uses (4.1) throughout.

Section 3 Lemma 2 writes k=Z2 while describing real CD algebras; over a
characteristic-two field the signs +1 and -1 coincide. This cannot literally
describe the real noncommutative algebras without a scalar-field correction.
Our tests work over real/rational coefficients and F2 **exponents**.
Definition 3's first power 2^n is inconsistent with the stated XOR definition;
the code uses the standard bitwise XOR specified in the text and elsewhere.

## Not coded / limits

No proof of unbounded Theorems 1-2, arbitrary-field construction or historical
novelty is claimed. Lemma 1 identities (2.4)-(2.6) and all elementary trace/norm
axioms are not each separately transcribed; existing project companions
already exercise standard involution/norm. Formula (4.10) is not tested as a
separate implementation. Split coverage changes only the **last** doubling
parameter, as in this paper; no claim about arbitrary signature follows from
(5.7). No split implementation or new public routine was added to CayleyKernel.
The GPU comparisons use sparse module expectations; CPU twist correctness
has its independent oracle in the exhaustive test. Finite passing examples
are evidence, not a formal proof. High-index checks do not benchmark dense
GPU multiplication. `basis_product` is the adapter for public `basis()`.

## Coefficient-type coverage

All basis/dense device fixtures also run typed replays. `test_08_gpu_complex_split_last_parameter` additionally transports the last-positive split signature in dimensions 2,4,8,16,32,64 into each complex kernel, checking all basis products and eight dense pairs per dimension. The kernel still implements the standard signature; twist bits and F2 exponents remain integer tests.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **8 tests: 8 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:32.858117+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
