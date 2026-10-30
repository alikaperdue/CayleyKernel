# Interpretation of the user's GPU eigenvalue exploration

The run dated October 28, 2026 used `--experiment eigen --backend gpu
--levels 3,4,5,6,7,8 --samples 10`, seed 23613 and tolerance 1e-9.
All 60 expected cases are present. The script hash matches the current runner;
all saved vector-file hashes were checked. The original results are preserved
in [gpu-eigen-exploration.json](results/gpu-eigen-exploration.json).

## Main result

Every transformed monic quadratic converged. In octonions, all ten converted
candidates also solved the original matrix equation. Beyond octonions, none
of the 50 converted candidates met the original-equation tolerance. Direct
Newton/GMRES solves recovered eigenpairs in 43 of those 50 cases.

| Level | Dimension | Transformed quadratic converged | Converted candidate valid | Final eigenpair verified | Unresolved |
| --- | --- | --- | --- | --- | --- |
| 3 | 8 | 10/10 | 10/10 | 10/10 | 0 |
| 4 | 16 | 10/10 | 0/10 | 10/10 | 0 |
| 5 | 32 | 10/10 | 0/10 | 9/10 | 1 |
| 6 | 64 | 10/10 | 0/10 | 9/10 | 1 |
| 7 | 128 | 10/10 | 0/10 | 7/10 | 3 |
| 8 | 256 | 10/10 | 0/10 | 8/10 | 2 |

This identifies a practical boundary of the octonion transformation, rather
than a boundary where eigenvalues necessarily cease to exist. It is consistent
with the loss of alternativity and inverse properties already examined in the
companion. No new general theorem is established.

## Independent verification

All 60 saved final candidates were re-evaluated with independent recursive
NumPy Cayley-Dickson doubling, without calling the kernel or its twist formula.
Both rows of `A*(1,x)=(1,x)*lambda` were checked, including `a+b*x=lambda`.
The combined Euclidean residual reproduced the same classification: 53 valid
pairs and seven unresolved candidates at absolute tolerance 1e-8. The maximum
residual among accepted pairs was 3.18e-10. See
[gpu-eigen-audit.json](results/gpu-eigen-audit.json).

The seven unsuccessful searches are:

| Level | Seed | Final reported eigenpair residual |
| --- | --- | --- |
| 5 | 28621 | 1.7201 |
| 6 | 29620 | 0.03705 |
| 7 | 30613 | 0.27324 |
| 7 | 30614 | 0.25407 |
| 7 | 30621 | 0.44835 |
| 8 | 31616 | 0.68685 |
| 8 | 31620 | 0.03843 |

These are solver stalls/budget exhaustion, not certified rootless matrices.
The method searches eigenvectors with first coordinate 1 and uses one
transformed starting candidate. It neither searches all vectors nor proves
nonexistence. Ten sparse-entry matrices per level do not estimate a universal
success probability or establish a monotonic dimension trend.

## Timing and the next useful investigation

Recorded case times sum to approximately 71.0 seconds. This includes continuation,
Krylov iteration, transfers, verification and vector saving; it is not a pure
kernel benchmark. The largest two levels dominate this particular run.

The most informative next step is to revisit these seven specific matrices
with alternate starting candidates, both square-root continuation branches,
and larger or preconditioned linear solves. Recovering a pair would identify
a numerical search limitation in that case. Repeated failure would still not
prove nonexistence; it would motivate a different formulation or a certified
analysis. This follow-up has not yet been run.

NPZ vectors remain in the original `/tmp/cayley-eigen-exploration` output
folder. The committed-style JSON snapshots preserve their hashes, coefficients,
seeds and run metadata; binary vectors are local artifacts.
