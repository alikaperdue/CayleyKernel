# Efficient Riemannian optimization on the Stiefel manifold via the Cayley transform

Li, Li Fuxin and Todorovic, arXiv:2002.01113v1, ICLR 2020, 15 PDF pages.
Here "Cayley" means a matrix transform, not Cayley-Dickson doubling.
There is a useful bridge: a module basis unit induces a skew-symmetric real
matrix J with J^2=-I, on which the matrix transform has a simple exact formula.

```bash
python3 -m pytest tests/papers/desktop_batch/test_2002_01113v1.py -v
```

Code: `tests/papers/desktop_batch/test_2002_01113v1.py`; measured errors,
orthogonality defect, source/PDF hashes and outcomes in `results/`.

## All tests in section order

| Section | Test suffix | Coverage |
| --- | --- | --- |
| 1-2 | None | Motivation/related work; no training reproduction. |
| 3.1-3.2 (1),(2),(6) | `03_1_projection_tangent_and_transport_linearity` | Stiefel shapes (4,2),(8,8),(16,5), seeded QR frames: W skew symmetry, tangent constraint, independent Euclidean projection formula, linear combination identity underlying implicit transport. |
| 3.2.1 (3)-(5), extension | `03_2_closed_cayley_and_module_complex_plane` | Full operators at dimensions 4,16,64,128 and three step sizes. Solve the closed transform, check orthonormality and compare with ((1-t^2)I+2tJ)/(1+t^2), t=alpha/2. Sparse module products reproduce transformed columns. |
| 4 Algorithm 1 | `04_algorithm_1_two_iterations_are_not_exactly_orthonormal` | Two fixed-point steps on a 4x2 frame with alpha=.8 produce a measurable nonzero orthogonality defect. Saved as an approximation limitation, not a kernel failure. |
| 5 Theorem 1 / Appendix B | `05_1_theorem_1_contraction` | Eight iterations on an 8x3 frame satisfy error contraction factor alpha/2=.2; errors saved. |
| 5 / Appendix B (16)-(17) | `05_2_appendix_B_big_O_not_little_o_counterexample` | Analytic two-dimensional example, checked against direct solves: error/alpha^(s+2) tends to a nonzero constant for s=0,1,2. Supports big-O, contradicts literal little-o notation. |
| 6 / device supplement | `06_gpu_rotation_replay` | Direct `multiply()` of an integer complex-plane rotation numerator on every dimension-16 basis column. Explicit prerequisite skip. |

## Findings and omissions

Finite truncation of the fixed-point series need not land exactly on the
Stiefel manifold, even though its limit does. The companion records the
two-step defect rather than treating approximate orthogonality as exact.
Theorem 1's contraction is observed, while Appendix B's Taylor estimate
uses little-o where this example gives big-O with nonzero leading constant.
The analytic normalized-error limit is 1/2^(s+1), not zero.

No new high-dimensional optimizer is delivered. Only selected Algorithm 1
ingredients and a two-step update are tested; repeated SGD training,
full momentum dynamics, Algorithm 2 ADAM bias corrections, stochastic
Theorem 2 and Appendix C convergence arguments are not implemented.
Section 6 CNN/RNN/CIFAR and temporal-data results, wall times, classification
accuracy, gradient stability and the appendix experimental settings require
training data and infrastructure absent here. No speed comparison is claimed.
Projection consistency does not prove all differential-geometric retraction
or transport axioms. The exact operator extension reaches dimension 128;
larger dense matrix solves have cubic cost even if basis signs are cheap.
Module basis arithmetic and ordinary matrix multiplication are kept distinct.

## Coefficient-type coverage

All dimension-16 rotation-numerator device products also run typed replays. The Stiefel matrix solve, convergence bounds and approximate orthogonality use their original real numerical protocol; they are not asserted exactly in half.

See [coefficient types and numerical scope](../COEFFICIENT_TYPES.md) for the
shared float16/complex64/complex128 fixtures, tolerances and reproduction.
Existing saved results describe their recorded runs; these device additions
require a working CUDA environment and are not represented by those logs.

## Saved evidence and current reproduction

The [saved JSON snapshot](results/results.json) reports **6 tests: 6 passed, 0 skipped, 0 failures, and 0 errors**. Its recorded timestamp is `2026-10-26T06:32:27.781761+00:00`. These are historical results for the source hashes and environment in that file, not a fresh run of the current checkout. Hash entries for retired helpers are provenance only; those helpers are not required by the commands above. Later test additions may change the test count. See the [saved-evidence policy](../README.md) for reproduction and log handling.
