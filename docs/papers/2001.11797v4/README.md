# A comparison of vector symbolic architectures

Schlegel, Neubert and Protzel, arXiv:2001.11797v4, 16 December 2021,
32 PDF pages. A VSA stores symbols in vectors, overlays them by bundling,
and associates roles with fillers by binding. This paper compares eleven
existing designs; Cayley-Dickson multiplication is not one of them.
The companion checks selected baseline ingredients, then runs a clearly
separate CD binding experiment, including negative controls for invertibility.

```bash
python3 -m pytest tests/papers/desktop_batch/test_2001_11797v4.py -v
```

Code: `tests/papers/desktop_batch/test_2001_11797v4.py`. Recorded numerical
scores, capacity cells, witnesses, outcomes and source/PDF hashes: `results/`.
NumPy needed for baseline arrays; exact sparse CD uses signed public `basis()`.

## Every coded test, in paper order

| Section | Test suffix | Scope |
| --- | --- | --- |
| 1 introduction | None | Motivation covered by the later retrieval experiment. |
| 2.1-2.4 spaces, similarity, binding | `02_1_similarity_bundling_and_binding_baselines` | Seeded D=128,1024,65536: bipolar self-inverse MAP binding, BSC-XOR equivalence, cosine/Hamming relation, exact FHRR phase unbinding (floating tolerance). MAP multiplication is elementwise; it is not module `multiply()`. |
| 2.5 analogy, (10)-(14) | `02_5_analogy_map_i` | D=65536, seeded unnormalized MAP-I records for USA/Mexico. Nearest candidate to Dollar times the record mapping is Peso; all six scores saved. One trial, not a probability estimate. |
| 3.1 capacity | `03_1_bundling_capacity_small_reproduction` | N=1000, D=128,1024,4096, k=2,8,32. Save all nine MAP-I top-k recall cells and independently check scalar scores. One trial per cell; neither original heatmap nor all eleven VSAs replicated. |
| 3.2 inverse chains / CD extension | `03_2_cd_binding_exact_for_basis_roles_not_all_roles` | Basis-role bind/unbind exact through sparse level 4096. A concrete nonzero sedenion role/filler pair binds to zero, demonstrating that arbitrary CD roles cannot support lossless unbinding. This is a new experiment, not a finding about the paper's listed VSAs. |
| 3.3 bundled pairs / CD extension | `03_3_cd_role_filler_retrieval` | D=128, three basis roles, 20 bipolar fillers: save retrieval outcomes; verify the exact signal-plus-cross-terms decomposition. Does not assert a universal statistical success rate. |
| 4.1 language n-grams / CD extension | `04_1_cd_ngram_parentheses_are_required` | An octonion basis triple gives different products under the two parenthesizations. CD n-grams need an explicit evaluation order. Both results saved. |
| 5 / device replay | `05_gpu_binding_counterexample` | Direct `multiply()` of the nonzero sedenion zero-divisor pair, exact sparse expectation; skipped explicitly without GPU prerequisites. |

## Analysis and uncovered material

CD binding can be exactly undone for selected basis roles even at very high
levels, but it is neither generally associative nor a globally invertible
binding. The zero-divisor witness is a decisive obstruction to using arbitrary
sedenions as lossless keys. Its failure is expected mathematics, not a kernel
failure. It would be interesting to compare restricted role families against
MAP/FHRR under noise; the saved small experiment is only a starting point.

Sections 2.1-2.4 catalogue MAP-C/B/I, BSC, BSDC-CDT/S/SEG, HRR, FHRR,
VTB, MBAT. Only bipolar/MAP-I, XOR and FHRR ingredient identities are coded;
no complete eleven-implementation library, HRR approximate correlation,
VTB matrix binding, sparse thinning or MBAT inversion is delivered. Section
2.5's non-self-inverse two-step analogy is not separately replicated.

Sections 3.1-3.3's original grid, ten-repeat statistics, inverse chains of
length 40, eleven-VSA comparisons, normalized curves and plots are not
replicated. The new capacity grid has only one trial per cell; its assertions
check bookkeeping, not a required accuracy threshold. Scores are saved for
analysis rather than turned into artificial pass/fail claims about accuracy.

Section 4.1's 21-language, 1000-training/100-test-sentence corpus and Section
4.2's image datasets, descriptors, preprocessing, ground truth, sLSBH,
SeqSLAM and AUC tables are not present in this repository. No language or
place-recognition result is claimed. The n-gram check is an algebraic
extension only. Section 5's empirical ranking and resource-cost discussion
cannot be verified from module arithmetic. No dense CD vector of dimension
2^4096 is allocated, and no GPU timing is claimed.

## Coefficient-type coverage

The sedenion binding-counterexample device fixture also runs typed replays, including phase-scaled zero divisors. Native complex CD coefficients do not implement FHRR elementwise binding, and do not make arbitrary sedenion roles invertible.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **7 tests: 7 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:27.386631+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
