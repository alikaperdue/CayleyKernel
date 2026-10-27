# Theoretical properties for neural networks with weight matrices of low displacement rank

Zhao et al., arXiv:1703.00144v4, 22 September 2017, 13 PDF pages.
This is a matrix/neural-network paper. It is relevant to CayleyKernel through
**linear operators**: left multiplication by a fixed basis unit gives a real
signed permutation matrix A with A squared = -I. Such A satisfies the
paper's potency hypothesis with q=2,a=-1. The tests use this bridge; they do
not identify ordinary matrix multiplication with a Cayley-Dickson product.

## Run / results

```bash
python3 -m pytest tests/papers/desktop_batch/test_1703_00144v4.py -v
```

NumPy is required for these matrix experiments. Outcomes, dependency skips,
source hashes and PDF hash are recorded under `results/`. All coded tests
are in `tests/papers/desktop_batch/test_1703_00144v4.py`.

## Tests in section order

| Section | Test suffix | Scope |
| --- | --- | --- |
| 1-2 introduction/related work | None | Historical and empirical motivation, no new finite formula. |
| 3.1 Lemma 3.1, (1)-(3) | `03_2_lemma_3_1_reconstruction_with_kernel_operators` | Dimensions 2,4,8,16,32,64; independent seeded integer M, A from module basis products, B diagonal with distinct entries in (0,1/2]. Check A^2=-I and M=(D+ADB)(I+B^2)^-1, D=M-AMB. |
| 3.1 Table 1 | `03_1_table_1_cauchy_stein_counterexample` | For ordinary Cauchy M_ij=1/(s_i-t_j), s=(2,3,5,7), t=(11,13,17,19), check Sylvester displacement has rank 1 while printed Stein displacement has rank greater than 1. Passing means this discrepancy was reproduced. |
| 3.2-4, Theorems 4.1-4.2, (4),(7),(17) | `04_theorem_4_1_and_4_2_single_ridge_embedding` | Dimensions 2,4,8,16: solve for generator g so first column of reconstructed rank-one-displacement M equals arbitrary seeded v. Check displacement rank and equality of selected tanh ridge outputs on 20 inputs. A is real diagonalizable over C, nonsingular; B has distinct absolute eigenvalues and I+B^2 is invertible. |
| 5 Lemma 5.1, (22)-(23) | `05_lemma_5_1_preserves_finite_network_output` | Embed five arbitrary eight-input ridges in five LDR blocks; select one output per block and zero the dummy output coefficients. Check original/expanded network outputs agree on 32 inputs. This verifies finite embedding, not the asymptotic bound. |
| 6 (26)-(32) | `06_gradient_finite_differences_and_printed_transpose` | Dimension 4, generator rank 2, nonsymmetric upstream gradient. Check every g,h coordinate against central finite differences, and reproduce disagreement with the printed g-gradient transpose. Absolute tolerance 2e-8. |
| Extension | `07_extension_basis_operators_through_dimension_256` | Exact-valued matrix operations verify orthogonality and A^2=-I in dimensions 32,64,128,256. These are actual full matrices. |
| GPU supplement | `08_gpu_operator_replay` | `multiply()` replays the 16 columns of left multiplication by e15; explicit skip without device prerequisites. |

## Findings

Table 1 appears to mix Stein and Sylvester operators: the standard Cauchy
rank-one property applies to AM-MB with diagonal operators, not generally
M-AMB. The regression preserves this distinction; it does not mark the
printed table as verified.

For W=sum A^k G H^T B^k T and upstream D=dO/dW, ordinary matrix calculus
gives dO/dG=sum (A^k)^T D (B^k T)^T H and
dO/dH=sum B^k T D^T A^k G. The printed (31) uses D^T in the G formula.
The finite difference experiment confirms the corrected formula and detects
the printed one on the chosen nonsymmetric example. The H expression (32)
agrees with this derivation. This is a formula-level finding, not a verdict
on the approximation theorems.

## Not coded / limits

Sections 4-5 contain infinite-dimensional functional-analysis arguments:
discriminatory activations, Hahn-Banach/Riesz representation, arbitrary
continuous functions/measures, Barron error estimates and deep approximation.
Finite samples cannot prove universal approximation or O(1/k) error rates.
The embeddings check selected algebraic ingredients only. No training on
ImageNet/CIFAR, compression ratio, classification accuracy or wall-time
improvement is claimed. No complete replication of all structured-matrix
rows in Table 1 or of the deep construction in Theorem 5.3 is implemented.

The q=2 reconstruction uses a nonzero negative scalar a, avoiding a
restriction to a=1. It tests a useful family of operators, not arbitrary
diagonalizable A,B. Theorem 4.1's general column-existence proof and its
printed g-construction are not fully validated. The code solves the column
map directly rather than assuming that construction. Section 6's bias/input
gradients and nonlinear multi-layer backpropagation are not separately tested.
Higher-dimensional basis operators remain useful, but arbitrary sums may not
square to a scalar in nonalternative algebras; this companion makes no such
generalization. `basis_product` in the shared code aliases public `basis()`.

## Coefficient-type coverage

The left-multiplication-operator device fixture also runs typed replays. This checks coefficient storage and complexification of the CD operator action, without changing the displacement-rank matrix theorems or neural-network experiments.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **7 tests: 7 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:26.247792+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
