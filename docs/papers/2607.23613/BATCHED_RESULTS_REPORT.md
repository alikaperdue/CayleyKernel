# CayleyKernel-assisted eigenpair searches: results and interpretation

**29 October 2026 · Real Cayley–Dickson algebras · GPU numerical experiments**

## Executive summary

CayleyKernel made it practical to search thousands of independent matrices for
numerical eigenpairs. Coordinating those searches into genuine GPU batches
produced measured solver speedups of **10.72× at dimension 64** and **17.89× at
dimension 256** on the recorded screening workloads. These are end-to-end
solver comparisons, not estimates based solely on fewer multiplication calls.

A survey of **8,192 sparse-coefficient matrices** verified at least one eigenpair
for **7,620 matrices (93.02%)**. Continuing only the unsuccessful cases recovered
another **270**, increasing coverage to **7,890 (96.31%)**. A separate survey with
denser coefficients verified **1,007 of 1,024 matrices (98.34%)** under the
original screening budget. Continuing its 17 unresolved saved candidates recovered
**8 more in 15.1 seconds**, increasing denser coverage to **1,015/1,024 (99.12%)**.
A subsequent alternative-start trial recovered **all nine remaining matrices in
8.6 seconds**, giving **1,024/1,024 verified matrices (100%)** in this sample.

The continuation and denser survey took approximately **76.7 and 82.5 seconds**,
respectively—**2 minutes 39 seconds combined** in their recorded run timers.
Retry backup setup is outside that timer. The time ceilings were safeguards,
not expected runtimes or targets to fill.

These are numerical findings about the sampled matrices and the search method.
They demonstrate many concrete eigenpairs beyond the octonions; they neither
prove that every matrix has such an eigenpair nor show that any unresolved
matrix lacks one.

## What was calculated

A Cayley–Dickson number is represented by a vector of real coordinates. Each
algebra level doubles the number of coordinates: level 3 is the eight-dimensional
octonions, level 6 has 64 coordinates, and level 8 has 256. The matrix in this
study remains **2 by 2**; each of its four entries is an algebra element.

For a matrix with entries `a, b, c, d`, the search looks for a right eigenpair
whose eigenvector is `(1, x)`. Its two equations are

```text
a + b*x = lambda
c + d*x = x*lambda
```

Substitution gives the nonlinear residual

```text
F(x) = x*(b*x) + x*a - d*x - c.
```

The brackets matter. In these algebras multiplication need not be associative:
`x*(b*x)` cannot simply be rewritten as `(x*b)*x`. CayleyKernel evaluates the
specified products; Newton's method proposes corrections, and GMRES solves the
associated real linear correction problems approximately. A line search chooses
how much of each correction to accept.

Every reported successful candidate has a saved `x` and `lambda` and finite
recorded residuals for **both original matrix rows**, each at most **1e-8**.
The Newton residual target is **1e-9**; acceptance allows ten times that target.
The computations use real float64 coordinates. “Verified” here means numerical
substitution within that absolute tolerance, not a certified exact root or a
complete list of eigenvalues. The search also restricts the eigenvector to the
form `(1,x)`.

## How CayleyKernel accelerated the work

The existing individual solver remained available as a reference. The batched
implementation supplies actual two-dimensional arrays to `CayleyKernel.multiply`:

```text
(batch size, algebra dimension), for example (64, 256).
```

Each row is an independent algebra product; the batch axis is not another
algebra coordinate. Products are batched throughout residual evaluation,
Jacobian applications, Krylov corrections and line searches. Arrays stay on the
GPU during a chunk, and the intermediate `b*x` is reused within a Newton step.
Each search retains its own convergence and step-acceptance decisions; finished
rows leave active computation.

The comparison used identical matrices and starting vectors with equal iteration
budgets. Timings synchronize the GPU and include solving, transfers, diagnostics
and final verification, excluding input generation and disk output.

| Screening workload, batch cap 64 | Individual | Batched | Measured speedup | Steady repeats |
|---|---:|---:|---:|---:|
| Level 6, 64 coordinates | 19.711 s | 1.839 s | 10.72× | 4 |
| Level 8, 256 coordinates | 35.913 s | 2.007 s | 17.89× | 1 |

At level 6, Newton corrections were 765 versus 764 and Krylov iterations were
44,095 versus 44,061. At level 8, both implementations used exactly 851 Newton
corrections and 50,061 Krylov iterations. Thus the speed advantage accompanies
nearly identical numerical work. Multiplication calls fell from 288,130 to
5,169 at level 6 and from 325,680 to 5,029 at level 8.

An archive audit found no per-start success disagreements across these solvers
or repeated measurements. Among jointly verified starts, maximum relative root
differences were approximately 1.18e-10 at level 6 and 7.93e-16 at level 8.
The kernel was treated as established infrastructure; it was not modified or
repeatedly revalidated during the later surveys.

Batching is not automatically faster for every workload. Earlier batch-size-1
runs were substantially slower than the individual solver because small GPU
operations carried substantial overhead. The demonstrated gains apply to the
recorded batched workloads, not all hardware, dimensions or batch sizes.

## Survey results

The large sparse survey used 4,096 matrices at each level. Each matrix entry
was independently generated with **three nonzero coordinates**, then normalized.
Every matrix received two starts: zero and a random normal vector scaled by the
square root of dimension. The screening allowance was 20 Newton corrections,
GMRES restart length 20, and three restart cycles per correction.

| Level | Matrices | Verified from zero | Added by random start | Verified with either start |
|---|---:|---:|---:|---:|
| 6 | 4,096 | 3,125 (76.29%) | 686 | 3,811 (93.04%) |
| 8 | 4,096 | 2,908 (70.996%) | 901 | 3,809 (92.99%) |
| **Total** | **8,192** | **6,033** | **1,587** | **7,620 (93.02%)** |

The second start made a substantial contribution, especially at level 8. The
similar final coverage at the two levels does not establish equal mathematical
difficulty; it describes this sampling family and these solver settings.
The complete sparse survey took **830.8 seconds**, about 13 minutes 51 seconds,
including its checkpoint work.

### Continuing the unresolved matrices

Only the 572 unresolved matrices were selected. The saved best candidates
received up to 30 additional Newton corrections with restart length 20 and six
GMRES cycles. **No new starts were generated.** Both outer and inner budgets
increased, so this experiment measures their combined effect.

| Level | Retried | Newly verified | Total verified in original survey | Still unresolved |
|---|---:|---:|---:|---:|
| 6 | 285 | 123 | 3,934 / 4,096 (96.04%) | 162 |
| 8 | 287 | 147 | 3,956 / 4,096 (96.58%) | 140 |
| **Total** | **572** | **270** | **7,890 / 8,192 (96.31%)** | **302** |

This recovered **47.20% of the unresolved subset** in 76.7 seconds. Among the
302 remaining continuation attempts, 214 exhausted their iteration budgets and
88 ended with stalled line searches. These are different reasons to consider
a changed search strategy; neither is a diagnosis of nonexistent eigenvalues.

### A denser coefficient family

The second experiment sampled 512 new matrices at each level, using **16 nonzero
coordinates per entry**, rather than three. Normalization, two-start policy and
screening budgets matched the large sparse survey. It used a separate seed.

| Level | Matrices | Verified from zero | Added by random start | Verified with either start |
|---|---:|---:|---:|---:|
| 6 | 512 | 468 (91.41%) | 34 | 502 (98.05%) |
| 8 | 512 | 481 (93.95%) | 24 | 505 (98.63%) |
| **Total** | **1,024** | **949** | **58** | **1,007 (98.34%)** |

This survey finished in **82.5 seconds**. Its remaining 17 matrices account for
34 unsuccessful starts, all ending at the iteration budget. There were no
recorded line-search stalls anywhere in this denser survey. Its largest accepted
row residual was about 9.40e-9, within the stated 1e-8 acceptance threshold.

The higher coverage is noteworthy: **denser input did not make this search
harder in the observed sample**. It also reduced reliance on the random start.
Better conditioning or fewer special sparse configurations are possible
explanations, but neither was measured. The surveys use different samples and
sample sizes, so the results are not a controlled pairing of the same matrices.
Moreover, 16 coordinates are denser, not fully dense: they occupy 25% of an
entry at level 6 and 6.25% at level 8.

### Continuing the unresolved denser matrices

All 17 unresolved matrix IDs were matched to the denser survey and each received
one continuation from its saved best candidate, with **no fresh starts**. The
allowance was 30 additional Newton corrections, GMRES restart length 20 and six
cycles, using the batched GPU solver at tolerance 1e-9. Both correction budgets
increased, so their separate contributions are not isolated.

| Level | Retried | Newly verified | Total verified in original survey | Still unresolved |
|---|---:|---:|---:|---:|
| 6 | 10 | 5 | 507 / 512 (99.02%) | 5 |
| 8 | 7 | 3 | 508 / 512 (99.22%) | 4 |
| **Total** | **17** | **8** | **1,015 / 1,024 (99.12%)** | **9** |

The run completed all cases in **15.129 seconds**, recovering **47.06%** of the
unresolved subset. Recovered candidates needed 1–15 additional Newton
corrections. Their first-row residuals were zero in the recorded arithmetic;
their second-row residuals ranged from 7.21e-11 to 2.56e-10. Both rows are finite
and below 1e-8 for all eight recoveries. Denser screening plus continuation took
97.6 seconds in the recorded timers, excluding retry backup setup.

Five remaining attempts exhausted 30 corrections; four ended with stalled line
searches. All nine used the full 120 Krylov iterations per attempted correction.
The histories show slowly decreasing or plateauing residuals, rather than
candidates just above the acceptance threshold:

| Level | Sample ID suffix | Final maximum row residual | Termination |
|---|---:|---:|---|
| 6 | 208 | 0.0780114483 | iteration budget |
| 6 | 259 | 0.2123177902 | line search stalled |
| 6 | 439 | 0.1076640818 | line search stalled |
| 6 | 445 | 0.0185449710 | iteration budget |
| 6 | 459 | 0.0979223809 | line search stalled |
| 8 | 9 | 0.0117311527 | iteration budget |
| 8 | 79 | 0.0003389893552 | line search stalled |
| 8 | 87 | 0.0804409778 | iteration budget |
| 8 | 134 | 0.0199031310 | iteration budget |

IDs have prefix `survey-20261101-level-L-sample-`. For example, level-6 sample
259 decreased from 0.2123179948 to 0.2123177902 before stalling, while level-8
sample 79 decreased from 0.0004542206 to 0.0003389894 before stalling. Level-6
sample 445 made more sustained progress, from 0.0201816696 to 0.0185449710,
but remained far above tolerance. These histories do not establish a cause:
full Krylov budgets do not by themselves diagnose conditioning, and stalled
local searches do not show that eigenpairs are absent.

The output vector hashes, exact selected matrix-ID set, coefficients, previous
residuals and byte-identical backed-up original manifest were checked without
recomputing algebra products. The eight recoveries are additional coverage of
the original 1,024 matrices, not eight new independent matrices.

### Alternative starts resolve all nine remaining denser matrices

The fixed trial allowed up to four alternative starts per unresolved matrix,
using the same 30-correction, restart-20, six-cycle budget as the continuation.
All **nine matrices verified on the first alternative start**, `-saved_x`.
Only nine searches were executed out of a maximum of 36: zero, perturbed and
random-direction starts were never scheduled. The successful start was
deterministic; the configured random seed did not contribute to these results.

| Level | Retried | Newly verified | Final verified in original survey | Still unresolved |
|---|---:|---:|---:|---:|
| 6 | 5 | 5 | 512 / 512 (100%) | 0 |
| 8 | 4 | 4 | 512 / 512 (100%) | 0 |
| **Total** | **9** | **9** | **1,024 / 1,024 (100%)** | **0** |

The run completed in **8.569 seconds**. All nine searches terminated at the
1e-9 Newton residual tolerance, requiring 10–21 Newton corrections. Recorded
first-row residuals were zero; second-row residuals ranged from 8.76e-11 to
1.91e-10. Both recorded rows are finite and below 1e-8 for every recovery.
The denser screening and its two targeted follow-ups together took **106.191
seconds (about 1 minute 46 seconds)** in the recorded run timers, excluding
retry backup setup.

The exact nine matrix IDs, coefficients, previous residuals, output vector
hashes and byte-identical backed-up input manifest were checked. Cumulative
coverage counts each original matrix once. The earlier continuation table
records the intermediate state; all nine cases listed there are now resolved.

Changing the initial candidate succeeded where further local correction had
plateaued. This supports sensitivity to the starting point as an explanation
for these nine search failures, without diagnosing the precise mechanism.
Negation is an initial guess, not an assumed algebraic symmetry or a formula
for a root: its initial residuals were approximately 2.11–3.65, substantially
larger than the saved candidates' residuals, yet the ensuing searches converged.
No arithmetic or precision changes were required. Complete coverage of this
finite sample remains numerical evidence, not a universal existence theorem.

## Relationship to the paper and Cayley–Dickson concepts

The local companion analysis concerns Chapman, Levin and Vishkautsan,
*General polynomials and eigenvalues over Cayley–Dickson algebras*, supplied
version `2607.23613v3.pdf`. Its Section 5 develops an octonion eigenvalue
transformation. Earlier companion experiments reproduced the printed octonion
example and checked the original right-eigenvector equations.

The present level-6 and level-8 surveys operate **beyond octonions**. Earlier
local experiments showed that identities used in the octonion transformation
can fail at higher levels, consistent with the loss of alternativity and the
appearance of zero divisors. The direct solver therefore works with the original
bracketed equation rather than assuming the octonion transformation remains
valid. The many successful direct searches are evidence of concrete higher-level
eigenpairs, not an extension of the paper's proof.

This distinction is important: losing an algebraic identity can invalidate a
particular derivation without preventing a particular matrix from having an
eigenpair. Likewise, a local numerical search can stall even when another start
or additional correction work finds a solution. The recovered 270 matrices
illustrate the latter point directly.

Nothing in these surveys contradicts the paper's stated hypotheses. No new
universal existence theorem, complete spectrum, or certified rootless matrix
has been established. See the existing [paper analysis](ANALYSIS.md) and
[earlier GPU exploration](GPU_EXPLORATION.md) for the mathematical context;
those earlier records are not new tests performed for this report.

## Checkpoint engineering and reproducibility

During the large sparse run, the aggregate JSON grew to approximately 33 MB.
Rewriting it twice per chunk added growing host-side overhead even though GPU
solve times stayed near 2.4–2.7 seconds per chunk. The survey now saves each
chunk independently, updates a small live status file, accumulates summaries
incrementally, and writes the full aggregate only at startup and exit.

The newer denser survey completed with approximately 2.6 seconds per chunk on
average, including its overall recorded overhead. This is consistent with
removing the growing serialization cost, but the two surveys are not a matched
before/after benchmark of checkpointing, so no separate checkpoint speedup is
claimed. Deadline and graceful-interrupt paths preserve current candidates;
completed chunk files provide recovery after a forced exit.

Seeds, coefficient supports, starting vectors, selected eigenpairs, both-row
residuals, termination histories, configuration and file hashes are retained.
The continuation archive was checked to contain exactly the original unresolved
matrix IDs, with matching previous residuals. No multiplication was recomputed
for this report.

## Interpretation and stopping point

The targeted denser investigation is complete: every sampled matrix now has a
saved numerically verified eigenpair. No further denser retries are needed.
If start-policy efficiency is studied later, a concrete question is whether
negating a stalled candidate should precede additional continuation. This trial
shows success on nine selected failures, but does not compare policy ordering
or establish an optimal strategy. No broader survey or benchmark repeat has
been launched.

## Evidence

The compact [machine-readable summary](results/batched-survey-summary-20261029.json)
contains source paths, SHA-256 hashes, per-level outcomes, budgets, residual
extrema, termination counts and benchmark measurements. The full archives below
are not distributed in GitHub. These historical paths identify the recorded
runs; they are not available downloads or inputs supplied by a clone:

- Sparse survey: `/tmp/cayley-large-survey.mIv6eg/results/results.json`
- Continuation: `/tmp/cayley-followup.86dicp/continued/results.json`
- Denser survey: `/tmp/cayley-dense-survey.04kBPb/results/results.json`
- Denser continuation: `/tmp/cayley-dense-continuation.VM7tKo/continued/results.json`
- Denser alternative starts: `/tmp/cayley-dense-fresh-starts.ihiOi1/retried/results.json`

The original archives remain unchanged. Their `/tmp` locations are local working
storage and should be retained elsewhere for long-term preservation of all
vectors; the compact summary alone is not a substitute for those vectors.

## Why this matters, and where it might lead

An encouraging feature of these results is how much remains possible after
some familiar algebraic rules stop working. The paper's octonion-specific
construction did not extend unchanged to higher Cayley–Dickson algebras. Yet
established numerical methods, applied to the original eigenpair equations,
found solutions beyond octonions—including every matrix in our denser sample
at dimensions 64 and 256.

We did not invent Newton's method or GMRES. Our contribution here is their
application to these bracketed equations, the GPU implementation that makes
the searches practical, and the recorded evidence of what succeeds and fails.
The especially encouraging observation was that the last nine denser cases
all converged after simply negating the saved starting candidate. A search
that appeared stuck had not exhausted what the algebra allowed.

### Potential value for calculations

An eigenpair identifies a vector on which a matrix's action has a simpler
description:

```text
M v = v lambda.
```

Finding such pairs gives us concrete tools for investigating operators,
equations and dynamics in higher Cayley–Dickson algebras. Instead of asking
only which familiar identities have been lost, we can ask what useful
structure remains and how to calculate with it.

The familiar benefits of eigenvalues—diagonalization, fast matrix powers and
stability analysis—do not automatically carry over. Nonassociativity means
that each intended use needs its own justification. For example, knowing
`M v = v lambda` does not by itself justify replacing `M(v lambda)` with
`(M v) lambda`, or concluding that repeated applications are described by
ordinary powers of `lambda`. The brackets and the actual operator matter.
Finding one eigenpair is a useful first step toward studying these questions.

### What this means for storage

A 256-dimensional algebra element still stores 256 real coordinates.
Eigenvalues alone provide no extra storage capacity, compression or error
protection. Those benefits would require a specific encoding and decoding
scheme, together with measurements showing an advantage over an ordinary
representation at comparable precision and reconstruction quality.

There may be useful ways to organize information through the algebra's
structure. That is an application to design and test, rather than a benefit
already demonstrated by the eigenpair experiments.

### What we can claim today

We have demonstrated computational access: we can find and numerically verify
meaningful operator structure in these higher algebras. The results are
encouraging because the failure of an octonion construction did not prevent
direct searches from succeeding broadly.

Our present limits are equally concrete. The surveys concern 2-by-2 matrices
from specified coefficient families, and search for eigenvectors of the form
`(1,x)`. Acceptance uses floating-point residuals, not exact root certificates.
Complete coverage of one finite sample proves neither universal existence nor
completeness of the computed spectrum. We have not demonstrated a storage
advantage, a general diagonalization method, or an application speedup over
conventional real or complex methods. The measured solver speedups compare
batched and individual implementations of this search.

### Focused questions worth testing next

These are possible next investigations, not additional runs already performed:

1. **Can a better starting-point policy reduce wasted work?** On a fixed set
   of saved cases, compare negating a stalled candidate with continuing it,
   using an equal total compute allowance. The nine recoveries make this a
   concrete question; they do not establish the best policy.
2. **What does an eigenpair predict about repeated computation?** For a few
   saved pairs, compare repeated applications of the explicitly defined
   operator with candidate eigenvalue-based formulas. Determine which
   identities hold, under what conditions, before building an acceleration
   method around them.
3. **How robust are the recovered pairs?** Perturb a small selection of
   matrices and track residuals and candidate changes. This would investigate
   sensitivity relevant to applications; a solver switching between roots
   must be distinguished from sensitivity of a particular root.
4. **Can an application benefit from this structure?** Choose one small
   calculation or encoding task, define its accuracy and resource requirements,
   and compare it with an ordinary real or complex implementation. A practical
   advantage must be measured in the task itself.

The opportunity is to turn accessible mathematical structure into useful
computation. We now have saved examples, working numerical tools and specific
questions to pursue. Whether that structure delivers a practical advantage
is the next application-specific question.

A [paired-sign experiment](PAIRED_SIGNS.md) is now available to test starting
point sensitivity on an unselected sample, running both `x0` and `-x0` for
every matrix with identical budgets. It records which individual matrices are
rescued by changing the sign, rather than comparing only overall failure rates.
The [paired-sign results](PAIRED_SIGN_RESULTS_20261029.md) now include completed
experiments at dimensions 64, 256, 1,024 and 4,096. Across 256 distinct new
matrices, 245 verified from at least one sign; 50 verified from exactly one.
Coverage was 63/64 at dimension 1,024 and 57/64 at dimension 4,096. Recorded
solver-chunk time rose from 13.739 to 130.249 seconds between those two
dimensions, despite only about 14% more Newton and Krylov work. These paired
experiments have their own samples, budgets and evidence summary; they do not
alter the earlier denser survey's completed 1,024/1,024 coverage.
