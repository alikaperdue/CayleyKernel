# Circuits, methods, and articles of manufacture for hyper-dimensional computing systems and related applications

Rosing et al., US 2022/0019441 A1, 20 January 2022, 109 PDF pages.
This is a nine-part patent disclosure, not a single CD algebra theorem.
The supplied image-only PDF was OCRed with temporary Tesseract 5.3.4 and
Poppler; mathematical formulas used below were checked against their
surrounding definitions. OCR is a reading aid, not the test oracle.

```bash
python3 -m pytest tests/papers/desktop_batch/test_20220019441.py -v
```

Code: `tests/papers/desktop_batch/test_20220019441.py`; raw finite experiment
measurements and historical outcome/PDF/source provenance: `results/`.
OCR was performed with external Poppler/Tesseract tools; no OCR helper is
required to run the committed tests.
Full copyrighted OCR text is not added to the repository.

## Every implemented test in disclosure order

| Part / paragraphs | Test suffix | What is checked |
| --- | --- | --- |
| Front matter, background, summary, drawings; [0001]-[0093] | None | Architecture/motivation; figures are addressed at their corresponding parts. |
| 1 PriveHD [0094]-[0137] | `01_privehd_linear_encoding_reconstruction_and_quantization` | Exact Hadamard example of (1-2a) linear encoding and inversion. Two different inputs have the same sign-quantized encoding. Demonstrates information loss, **not differential privacy**. |
| 2 HyperRec [0138]-[0179] | `02_hyperrec_xor_majority` | All 65,536 byte pairs: XOR bind/unbind; all bit triples: majority agrees with at least two inputs. These are Boolean ingredients, not a trained recommender. |
| 3 HPU [0180]-[0249], especially [0208] | `03_hpu_blend_dot_product_has_cross_term` | D=4096, prefix blend of length d=1000. Dot with first input equals d plus an explicit suffix cross term; the saved nonzero term shows dot=d is generally an expectation/approximation, not an exact identity for arbitrary random draws. |
| 4 SearcHD [0250]-[0307] | `04_searchhd_bit_substitution_moves_toward_sample` | Replacing 200 selected bits of a 1024-bit model by a sample never increases its Hamming distance to that sample. Selected stochastic-training ingredient only. |
| 5 associative search [0308]-[0348] | `05_associative_search_normalization_and_safe_bound` | Normalized class vectors give same cosine/dot argmax. Verify an exact Cauchy-Schwarz upper bound for suffix score contributions. This is a safe software search ingredient, not a replication of analog search circuitry or approximate hierarchy. |
| 6 GenieHD [0349]-[0389], especially [0363]-[0366] | `06_geniehd_rolling_window_vs_naive` | Exact bipolar rolling-window encoding vs direct n-gram recomputation for widths 1,3,6,10 on a fixed DNA string, D=4096; every window checked. Uses elementwise multiplication and reversible rotations exactly as specified, not CD multiplication. |
| 7 RAPID [0390]-[0449] | `07_rapid_unit_edit_distance_against_exhaustive_paths` | Restricted unit-cost Needleman-Wunsch/edit-distance recurrence checked against exhaustive paths for all pairs of A/C words of length<=3; example ATGTTATA/ATCGTCC score saved. No claim to reproduce the patent's affine-gap scores. |
| 8 workload-aware FPGA [0450]-[0496] | `08_workload_markov_probability` | Illustrative three-state row-stochastic model, 20 forecast steps: probability mass/nonnegativity. Matrix and states are chosen here, not inferred hardware workload statistics. |
| 9 SCRIMP [0497]-[0548] | `09_scrimp_stochastic_multiply_requires_independence` | Exact independent Cartesian bipolar streams satisfy E[XY]=E[X]E[Y]. Reusing the same stream violates the product-of-means rule. A software statistical ingredient, not a stochastic circuit emulation. |
| Beyond disclosure | `10_cd_is_not_elementwise_binding` | The module imaginary basis square is -1, while the componentwise square would preserve the coordinate. Prevents silently conflating HDC multiplication with CD multiplication. |
| Device supplement | `11_gpu_distinction` | Direct module `multiply()` replay of the distinction, explicit dependency/device skip. |

## Analysis / not implemented

The useful connection is an experimental comparison of representations and
operations, not replacing every multiply symbol with CayleyKernel. Most
products here are componentwise Boolean/bipolar operations. The tests above
use NumPy for those operations; the module is exercised only where the
operation really is a CD product. Extending the DNA rolling identity to
arbitrary CD vectors would require associativity/inverse laws that fail; it
is not a valid automatic higher-level generalization.

For each part, substantial content remains outside this companion:

1. No DP mechanism, sensitivity calibration, epsilon/delta proof, PSNR,
   retraining or privacy guarantee; deterministic quantization is insufficient
   to establish DP. No privacy-sensitive deployment is provided.
2. No rating-level encoding pipeline, prediction formula, MovieLens/Amazon
   benchmark, cold-start analysis or reported MSE/memory savings replication.
3. No general regression/reinforcement learning suite, compiler, ISA simulator,
   FIFO/HPU scheduling, DBlink, analog precision, ADC/DAC or endurance model.
4. No complete multi-prototype SearcHD classifier or in-memory sense amplifiers,
   CAM, stochastic update distribution or classification/energy curves.
5. No full multi-stage clustering, two-power quantization/retraining, approximate
   search error or evaluation on ISOLET/UCIHAR/IMAGE datasets.
6. No multi-length Method 1 reference aggregation, refinement, statistical
   false-positive independence proof, genome-scale dataset or GPU/FPGA/ASIC
   throughput replication. Overlapping n-grams can introduce dependencies.
7. No affine-gap alignment/traceback, ReRAM logic, chromosome-scale DP or
   hardware scalability/power comparison.
8. No fitted voltage-delay/power curves, QoS optimizer, FPGA control, workload
   trace fitting, misprediction feedback or measured energy savings.
9. No ReRAM bitline simulation, stochastic number generator circuit, segmented
   switches, nonlinear accumulators, DNN/Sobel accuracy or energy/endurance data.

The concluding generic implementation paragraphs and claims 1-16 (PDF
pp.107-109) are not legal or patent-validity assessments. Claims involving
software primitives have only the finite examples above; hardware claims
are untested. The nine-part paragraph ranges are navigation guides, not a
claim that every paragraph contains an independently reproduced experiment.
Where exact output inputs/seeds/traces/layouts are unavailable, no reproduction
of a published figure or numerical performance result is claimed.

## Coefficient-type coverage

The CD-versus-elementwise-binding device negative control also runs typed replays. Boolean encoders, elementwise binding, stochastic streams and hardware claims retain their documented scope.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **11 tests: 11 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:28.206094+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
