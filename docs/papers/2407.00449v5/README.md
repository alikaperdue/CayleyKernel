# Fully tensorial approach to hypercomplex-valued neural networks

Niemczynowicz and Kycia, arXiv:2407.00449v5, 26 January 2026.
This is relevant because its multiplication tensor can represent the exact
CayleyKernel product. Its dense and convolution algorithms can then be
checked against sums of module products, including in nonassociative and
nonalternative algebras. No training dataset is needed for these identities.

```bash
python3 -m unittest tests.papers.desktop_batch.test_2407_00449v5 -v
```

Code: `tests/papers/desktop_batch/test_2407_00449v5.py`; measurements and
provenance in `results/`. NumPy implements the real tensor contractions;
independent recursive doubling supplies structure constants. The comparison
path calls the module's signed basis multiplication.

## Every test in section order

| Section | Test suffix | Finite coverage |
| --- | --- | --- |
| 1 | None | Motivation; existing software/training claims not replicated. |
| 2.1-2.2, (5)-(7) | `02_structure_tensor_complex_example_and_cd_products` | Dimensions 1,2,4,8,16,32,64; eight seeded dense integer products per dimension. Tensor contraction vs module sparse product; both printed complex tensor slices independently checked. |
| 3.1 Algorithm 1 | `03_1_dense_contraction_permute_reshape_bias_activation` | Batch=2, input multiplicity=3, outputs=4, algebra dimensions 2-32: exact contraction/permutation/reshape against explicit sums of x*w. Bias and tanh post-processing checked. |
| 3.2 Algorithm 2 | `03_2_convolution_1d_2d_3d` | All three spatial ranks, algebra dimensions 4 and 16, batch=2, two input/output channels, valid unit-stride kernels of width 2. Per-component real cross-correlations vs every explicit window's module products. Checks output component order as well as values. |
| 3.3 / 4 Lemma 2, CD case | `04_1_cd_component_functional_reconstruction` | Dimensions 2-64, every output component: each scalar-projection multiplication form is a signed permutation and nonsingular; reconstruct an arbitrary seeded real functional and verify its evaluation. No inference from absence of a global annihilator is needed. |
| 4 Definition 1 / Lemma 2, general scope | `04_2_nondegeneracy_definition_does_not_imply_lemma_for_all_algebras` | Explicit dual-number counterexample: unit guarantees no vector annihilates every partner, but the scalar-component bilinear form has rank 1 instead of 2. Saved component ranks expose the gap in the printed general implication. This reference-only algebra is not a CD module feature. |
| 5 / Appendix B extension | `05_appendix_B_tensor_free_high_index_product` | Sparse products through 4096-bit indices against independent doubling; saved dense-tensor entry/nonzero counts show why materializing d^3 entries is inappropriate at high d. |
| Device replay | `06_gpu_tensor_replay` | Direct `multiply()` of a dimension-16 integer pair, explicit dependency/device skip. |

## Findings / boundaries

The concrete tensor layer formulas agree in the tested CD algebras; they
need bilinearity, not associativity. The generalized Lemma 2 uses a stronger
condition than Definition 1 supplies: every scalar-component pairing must
be nondegenerate, whereas a unital algebra automatically satisfies the
printed no-global-annihilator condition. The dual-number example separates
these claims. For the tested CD dimensions, the stronger component condition
is verified directly. No conclusion that CD universal approximation fails
is drawn from the general lemma's counterexample.

No complete neural-network package, TensorFlow/PyTorch/Keras comparison,
autodiff/gradient audit, training/accuracy benchmark or arbitrary activation
implementation is delivered. Convolution coverage is valid, unit-stride
cross-correlation only; padding, dilation, nonunit strides and all bias/activation
combinations in convolution remain untested. Sections 2.1/Appendix A's
coordinate-free tensor quotient/universal-property/functoriality statements
are not formally proved. Section 4's compact-domain sigmoid approximation
theorem is not numerically proved; no finite grid could establish it.

The full multiplication tensor has d^3 storage entries but only d^2 nonzeros
for CD basis products. The high-level extension evaluates only nonzero
supports, preserving the multiplication; it does not claim dense O(1) work
or a benchmark of the paper's materialized-tensor implementation. High index
widths are not dense allocated dimensions. GPU tests skip explicitly if
prerequisites are absent.

## Coefficient-type coverage

The structure-tensor device fixture also runs typed replays, including independent imaginary coordinates. `test_07_gpu_typed_dense_layer` additionally checks Algorithm 1 in dimensions 2,4,8,16,32 for float16, complex64 and complex128: a two-sample, three-input, four-output layer is evaluated with broadcast GPU products and compared with the independent tensor contraction. Small integer real/imaginary coefficients keep the half products and channel sums exact. The real component-reconstruction and layer/convolution reference experiments retain their stated scope; new multiplication dtypes alone do not certify a complex-valued training pipeline.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **7 tests: 7 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:37.252288+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
