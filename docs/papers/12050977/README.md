# Representation of an ordered group of symbols by hypervectors

Karunaratne et al., US 12,050,977 B2, 30 July 2024. The supplied 27-page PDF
is an image scan: cover/references, 15 drawing sheets, then description and
claims in columns 1-20 (PDF pp.18-27). Description/claims and Figures 6A-C,
7A-C were read from rendered pages; there is no OCR dependency.

The method repeatedly circularly shifts a binary vector by a symbol-dependent
amount and applies a fixed shuffle. Numerical inputs instead select a mask.
This is a permutation encoder, not Cayley-Dickson multiplication.

## Run and recorded evidence

```bash
python3 -m pytest tests/papers/desktop_batch/test_12050977.py -v
```

The adjacent results directory records test outcomes and PDF/source hashes.
All tests are in `tests/papers/desktop_batch/test_12050977.py`.

## Coverage in document order

| Portion | Test suffix | Scope |
| --- | --- | --- |
| Background and detailed description, cols.1-4 | `01_background_permutation_preserves_density_and_distance` | Seed 12050977; vector dimensions 8,1024,10000,65536,1048576: circular permutation preserves binary density and Hamming distance. The million-component calculation is an actual CPU vector operation, not a dense CD multiplication. |
| Character encoder, cols.4,7-10; Fig.6A-C | `02_detailed_description_character_shift_stages_fig6` | Exact printed eight-bit shift inputs/outputs for c=2,a=0,b=1, before the shuffle. Left shift matches the drawings. |
| Fig.5 recurrence, cols.9-10 | `03_flowchart_sequential_recurrence` | All 64 three-symbol words over four shift amounts, independent composition of coordinate maps vs iterative vector implementation. Uses an explicitly specified shuffle permitted by the description, not a transcription of the Fig.6 wiring. |
| Bundling / associative memory, cols.5-6,9,12; Fig.8-9 | `04_bundling_distance_and_finite_collision` | All 256 four-symbol words with D=8: count distinct encodings, check additive bundle mass and self-distance. Collisions are unavoidable since weight-four permutations offer at most 70 outputs. This limits literal universal uniqueness language; the patent's large-D statistical behavior is not refuted. |
| Beyond patent: CD substitution | `05_extension_basis_action_is_signed_xor_not_cyclic_shift` | Sparse basis actions through level 4096: output index is XOR, signs can be negative, applying the same nonreal basis action twice gives -identity on tested basis inputs. A cyclic +1 coordinate shift has a different map. Thus basis multiplication is not a drop-in implementation of the patent. |
| Direct GPU replay | `06_gpu_signed_action_negative_control` | Two integer products in dimension 8 through `multiply()`, including the negative square. Explicit dependency/device skip supported. |

## Not implemented and proposed follow-ups

- Background cognitive, classification, energy, area and crossbar hardware claims require measurements/data/hardware absent here. Bibliography, cover classifications and Fig.10 generic computer architecture add no module-verifiable numerical identity.
- Numerical masks, cols.5,8,11-12 and Fig.7A-C: no general masked-shift specification is reconstructed. A naive coordinate-wise masked roll can duplicate/drop bits; a rotation among selected positions is a different algorithm. Figure 7A visibly changes the bit count in its intermediate vector, so the density-preserving character tests must not be extended to this branch. Claims 4-6/16 are not computationally verified.
- Figure 9: original random seed, generated words, full shuffle maps, baseline implementations (HCUB_D2/D4, RR), and raw distributions are not supplied. No claim to reproduce the plots or statistical classification accuracy. Next work could specify these choices and save a fresh Hamming-distance study; it would be a new experiment.
- Claims 1-3,7-10,14-15,17-20 have software ingredients covered as above, not a legal claim-scope assessment. Claims 11-13 concern systems/parallel devices not exercised here.
- A separate CD-based signed encoder could be studied for collision rates and nonassociativity. It must be named a new encoding scheme, with explicit parentheses and signed coefficients. This companion tests only its most basic obstruction to binary-permutation substitution.

`basis_product` in the shared adapter calls this checkout's signed one-based
`basis()`. The high-bit extension does not allocate dimension 2^4096 and does
not claim GPU performance. No patent rights or validity conclusion is made.

## Coefficient-type coverage

The existing signed-XOR-action negative-control device fixtures also run typed replays. Half and complex coefficients exercise the CD comparison only; the patent encoder remains a binary permutation algorithm.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **6 tests: 6 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:22.659587+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
