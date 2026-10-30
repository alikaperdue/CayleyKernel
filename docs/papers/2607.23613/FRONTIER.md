# Beyond the paper's worked examples

The original suite checks finite instances of the paper. It does not prove its
universal theorems. These additional experiments investigate larger inputs and
the boundary of the octonion eigenvalue argument. They make no novelty claim.

Your reported CUDA run passed all 22 original tests in 8.12 seconds. That is
user-reported suite time, not an independently recorded kernel benchmark. The
new experiments below have separate local CPU evidence and an optional GPU test.

## Run the new experiments

Run these commands from the repository root in a Python environment with NumPy and SciPy. Install the
additional dependencies if needed:

```bash
python -m pip install -r tests/papers/arxiv_2607_23613/requirements-frontier.txt
pytest -q tests/papers/arxiv_2607_23613/test_frontier.py
```

The runner is [scripts/paper_quadratic_frontier.py](../../../scripts/paper_quadratic_frontier.py).
All output paths must be new directories; existing evidence is never overwritten.

Start with non-scalar quadratic roots on the GPU:

```bash
python -m scripts.paper_quadratic_frontier \
  --experiment roots --backend gpu --levels 3,4,5,6,8,10 \
  --samples 5 --output /tmp/cayley-frontier-gpu-roots
```

Then push dimension and linear-term strength independently:

```bash
python -m scripts.paper_quadratic_frontier \
  --experiment roots --backend gpu --levels 12,14,16 \
  --samples 2 --output /tmp/cayley-frontier-gpu-deep

python -m scripts.paper_quadratic_frontier \
  --experiment roots --backend gpu --levels 4,6,8 \
  --samples 10 --strength 2 --output /tmp/cayley-frontier-gpu-strong
```

Levels 1-20 are accepted, including level 20 (1,048,576 coordinates). Each
nonlinear solve can require many products. A fast individual multiplication
does not make an entire continuation solve equally fast. Start with the first
command and use its recorded call counts and wall times to choose larger runs.
The solver stores about 40 Krylov vectors plus its other vectors; it never
builds a dimension-by-dimension Jacobian. GPU mode currently uses host NumPy
vectors for SciPy solves, so transfers and CPU work are included in timings.

Probe the eigenvalue construction beyond octonions:

```bash
python -m scripts.paper_quadratic_frontier \
  --experiment eigen --backend gpu --levels 3,4,5,6,7,8 \
  --samples 10 --output /tmp/cayley-frontier-gpu-eigen
```

A quick CPU comparison uses `--backend cpu`. CPU root mode precomputes signed
permutations from CayleyKernel `twist_many()` for the sparse fixed coefficients,
then applies them to dense vectors. This is a specialized sparse-coefficient
algorithm, not dense kernel timing. CPU eigen mode uses independent recursive
full products and is limited to level 8. GPU mode uses actual `multiply()` for
all fixed-coefficient and full products, including the final original-equation
checks. It never silently falls back to CPU multiplication.

## Experiment 1: unplanted general quadratics in full dimension

Solve

```
f(x) = x*x + strength * (a*(x*b) + c*x) + r = 0.
```

The coefficients `a,b,c` have three randomly positioned nonzero entries each.
Their indices range across the selected algebra. The constant `r` is a dense
independent Gaussian vector, normalized in scale by sqrt(dimension). It is
not constructed from a desired root. These experiments are more general than
the earlier deep `x*x+2*x+r` family, whose linear coefficient was scalar.

The method starts at the two square roots of `-r`, then increases a continuation
parameter from 0 to 1 in `x*x+s*L(x)+r`. At each step damped Newton iteration
uses GMRES to solve Jacobian-vector equations. The derivative of the square
is computed from the quadratic identity, preserving linear memory usage.
Failed steps reduce the parameter increment. Reaching the target still requires
checking the original residual; in GPU mode that check uses `multiply(x,x)`.

Things to inspect:

- **Root separation:** do the two starting branches reach distinct roots, or
  approach the same one? Neither outcome counts the entire zero set.
- **Continuation trace:** which parameter values require smaller steps?
- **Effect of dimension and strength:** does stronger noncommutative coupling
  cause more iteration or branch difficulty?
- **Residual and direct-product residual:** do the coordinate identity and
  actual kernel evaluation agree?

The basic experiment has only two starting paths. It can miss other roots and
singular root spheres. A stalled continuation is **unresolved**, not a rootless
polynomial or a contradiction of the paper's existence theorem. This is an
exploratory numerical solver, not a certified or exhaustive algebraic solver.

## Experiment 2: can a failed transformation still lead to an eigenpair?

For independently sampled matrix entries `[[a,b],[c,d]]`, first run the paper's
transformed monic quadratic. Then measure separately:

1. Its own polynomial residual.
2. The residual in the original equation after setting `x=b_inverse*y`.
3. The inverse-property defect `b*(b_inverse*y)-y`.
4. The Artin-identity defect `b*(x*(b*x))-(b*x)*(b*x)`.

Beyond octonions these defects can be nonzero. The script then tries damped
Newton directly on

```
x*(b*x) + x*a - d*x - c = 0.
```

This equation is equivalent by distributivity to the right-eigenvector equation
for `(1,x)` at every level; no reassociation or inverse identity is assumed.
Its derivative keeps both bracketed terms separately. A candidate is accepted
only after checking `c+d*x = x*(a+b*x)` with full products. The first eigenvector
row is satisfied by defining `lambda=a+b*x`.

This separates “the octonion proof stops working” from “an eigenpair cannot be
found.” Success in sampled higher algebras does not prove existence for every
matrix. Failure does not rule out another root or an eigenvector outside the
form `(1,x)`.

## Recorded local observations

The JSON snapshots below record CPU experiments using NumPy 1.26.2 / SciPy as
specified in each file; consult their version fields for exact environment
values. They preserve seeds, coefficients, solver histories and source hashes.

| Experiment | Completed local scope | Observation |
| --- | --- | --- |
| General roots, strength 0.5 | 2 polynomials each at levels 3,4,5,6,8,10,12 | All 28 tracked branches produced accepted roots. Largest coordinate residual below 1e-10. Full independent recursive-product checks ran through level 8; larger CPU levels used the quadratic identity. |
| Stronger linear terms, strength 2 | 1 polynomial each at levels 4,5,6 | All six branches produced accepted roots. |
| Right eigenvalues | 2 matrices each at levels 3,4,5 | All six ultimately produced verified eigenpairs. The four level-4/5 transformed candidates had substantial original-equation errors before the direct solve. |
| Regression tests | New companion only | 8 passed, 1 skipped locally; the skip is the GPU root/eigen pipeline because this execution environment has no CUDA device. |

For the four higher-level matrix cases:

| Level | Seed | Original-equation error after transformation | Error after direct solve |
| --- | --- | --- | --- |
| 4 | 27613 | 0.6351 | 9.84e-11 |
| 4 | 27614 | 0.01968 | 7.70e-11 |
| 5 | 28613 | 1.7117 | 3.58e-11 |
| 5 | 28614 | 0.8897 | 8.48e-11 |

The second level-4 sample is especially informative: its inverse-property defect
is near roundoff, while its Artin-identity defect is about 0.01968. A correct
inverse step alone does not make the octonion transformation valid.

## Saved data and independent inspection

- [Root run](results/frontier-roots.json)
- [Stronger-coupling run](results/frontier-strong.json)
- [Eigenvalue run](results/frontier-eigen.json)
- [Portable eigenpair vectors and coefficients](results/frontier-eigen-witnesses.json)

Each new run writes `results.json` and compressed NumPy files containing the
computed vectors. Root files include the independent constant; eigen files
include `x` and `lambda`. JSON records vector-file hashes. Load files with
`numpy.load` using the default `allow_pickle=False`.

The committed JSON snapshots retain the original `/tmp` output locations.
Their `vector_file` fields are relative to those locations. Binary NPZ files
are local run artifacts, not Git fixtures; regenerate a run with its recorded
configuration to obtain them. The small eigenpair witnesses are additionally
preserved as portable JSON so their original matrix equations can be checked
without those local binaries. Seeds and environment versions aid reproduction,
but numerical last bits may differ across library versions and hardware.

Regression checks cover signed-permutation convention versus independent
recursive doubling, stable square roots near the negative real axis, the
Jacobian formula, unplanted sedenion roots, octonion eigenpairs, failed sedenion
transformation followed by direct recovery, stalled-solver classification,
optional GPU execution, and saved-vector reload into the original equation.

Promising next investigations are more continuation starting paths, singular
root-sphere detection, conditioning estimates near stalls, and GPU-resident
Krylov solves to reduce host transfers. None is claimed implemented here.

## Coordinated batched eigen searches

See [BATCHED_EIGEN.md](BATCHED_EIGEN.md) for the batched Newton/GMRES
solver, correctness tests, archival guarantees, and identical-workload CPU/GPU
benchmark commands. Eigen searches and retries default to batching; the individual solver remains
available with `--solver individual`. Root experiments retain their existing default.
