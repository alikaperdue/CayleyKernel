# The Cayley-Dickson doubling products

John W. Bales, arXiv:1707.07318v4, 28 August 2023, 24 PDF pages.
The v4 text makes an important distinction: four of the eight catalogued
products give octonions at the third doubling; four are **pseudo** products
with zero divisors already in dimension eight. This corrects any impression
from the earlier papers that all eight must be octonion conventions.

## Run

```bash
python3 -m pytest tests/papers/desktop_batch/test_1707_07318v4.py -v
```

Code: `tests/papers/desktop_batch/test_1707_07318v4.py`; provenance and
individual results: `results/`. Arithmetic is exact integer sparse arithmetic.
The generic oracle implements all 32 (f_i,g_j) combinations, separately from
the eight-product implementation used in the periodicity companion.

## Tests in section order

| Sections | Test suffix | Scope |
| --- | --- | --- |
| 1 | None | Motivation only. |
| 2-5 definitions, 32 candidates, XOR | `02_05_basic_properties_all_32` | All candidate products, all 16x16 basis-index pairs: identity, squares, anticommutation; on constructed nonbasis vectors also norm sum-of-squares and conjugation reversal. Finite checks of (2.1)-(2.6), not a proof that no other polynomial forms exist. |
| 6-7 initial points, Table 7.1 | `06_07_initial_elimination_16` | Initial interior triples for s=1-31; exactly 16 candidates satisfy cyclic quaternion signs. |
| 8-9 Table 8.1 and eight survivors | `08_09_full_elimination_8` | All distinct nonzero p,q<32: exactly the listed eight survive the quaternion property. Independently compare all eight against the periodicity companion on every 64x64 pair. |
| 10-11 octonion tables and Fano plane | `10_11_fano_triples_and_pseudo_zero_divisors` | Every printed triple for all eight, all cyclic rotations and XOR supports. The printed (e2+e5)(e3+e4) vanishes exactly for P1,P2 and their transposes, not the four octonion products. |
| 11 octonion distinction | `11_octonion_vs_pseudo_alternativity_polarization` | Exhaust coefficients of left/right alternativity in dimension eight using every basis diagonal and pair sum. Only P0,P3 and their transposes satisfy the identities throughout R^8 by polarization. This is stronger than merely testing basis triples. |
| 12-15 recurrences / parity blocks / twist tables | `12_15_module_and_twist_tables` | Every module-converted 128x128 pair agrees with independent P3-transpose doubling. All 2x2 parity blocks for lower indices<16 agree between two oracle implementations, all eight products. Does not independently transcribe every printed table cell. |
| 16 tree and (16.2)-(16.4) | `16_tree_example_and_extension_families` | Printed (93,37) example; all n<m through m=7 with r,s<min(2^n,8), plus (n,m)=(20,32),(128,1024),(1024,4096). All three sign families check via independent P2 tree; this is a reference-only pseudo-product extension. |
| 17 / device replay | `17_gpu_standard_octonions` | Direct module `multiply()` for the printed pseudo-zero-divisor factors, in the module's standard block convention. Expected from sparse standard multiplication, not a claim that the module implements P2. |

## Interpretation and omissions

- Section 2(3b) calls the signed basis elements a group. For genuine octonions
they are closed with identity and inverses but are not associative, so
"loop" is needed. No group associativity is assumed by these tests.
- Sections 4,7,8's exclusion of other possible polynomial forms is a proof
claim. The code enumerates exactly the displayed 32 candidates, not every
conceivable doubling law. The 12 displayed induction relations and the
successor formulas (9.4)-(9.7) are not individually transcribed; the complete
finite sign/quaternion tests cover their underlying finite multiplication.
- The Section 11 alternativity classification tests all polynomial
coefficients in dimension eight; it is not a general proof of isomorphism,
nor a separate exhaustive verification of all Moufang identities.
- The pictures' coordinates, arrow graphics and pixel patterns are not
compared. The signed Fano triples are checked, not the drawing aesthetics.
- Several Section 12 scalar recurrences appear to have transcription issues
(e.g. (12.13) applies a conjugation condition to s where the displayed product
conjugates r). The code follows the explicit doubling products rather than
silently accepting each printed scalar recurrence. A complete equation-level
erratum audit remains unimplemented.
- Equation (16.1) is a suggested ratio-region description; its printed
inequality layout is not used as a computational theorem. Only the explicit
tree example and integer families (16.2)-(16.4) are tested here.
- The Appendix bc/LaTeX program is not executed; its sign-generation purpose
is covered by Python oracles, not its rendering instructions.
- High levels mean sparse integer bit widths, not dense dimensions allocated.
The pseudo P2 extensions are independent-reference checks; only the
P3-transpose comparisons exercise `basis()`/`twist()`. GPU execution skips
explicitly if dependencies are absent. No module arithmetic was changed.

## Coefficient-type coverage

The standard-octonion device fixtures also run typed replays. Complexification does not turn the pseudo products into the standard product or establish a coordinate map between the catalogued conventions.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **8 tests: 8 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:26.901622+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
