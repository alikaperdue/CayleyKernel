# Opposite starts and higher dimensions: results and interpretation

**29 October 2026 · Saved GPU experiments · Seed 20261102**

The new runs answer two useful questions. Opposite starting points often
rescue different matrices, and direct eigenpair searches remain productive
at dimensions **1,024 and 4,096**. The largest dimension takes substantially
longer and has lower coverage under the same screening budget. These are
observations about the sampled problems and solver, not an existence boundary.

## What was run

The full archives are not distributed in GitHub; the committed summary records
their provenance and aggregate findings, rather than all input and output vectors.
The historical paths below are not downloads supplied by a clone.

Four archives were supplied under `/tmp/cayley-paired-signs.vHkxDP/`:

| Directory | Levels | Status | Recorded elapsed time |
|---|---|---|---:|
| `results` | 6, 8 | complete | 10.547 s |
| `results2` | 6, 8, 10 | interrupted | 7.772 s |
| `results3` | 10 | complete | 13.856 s |
| `results4` | 12 | complete | 130.655 s |

The complete runs contain **256 distinct matrices and 512 searches**: 64
matrices at each of four levels. The interrupted archive repeats the first
32 matrices at levels 6, 8 and 10. Its input chunks are identical to the
corresponding complete-run inputs. Its completed level-6 and level-8 search
records are also identical; its 32 level-10 pairs were stopped. It contributes
no additional independent matrices and is excluded from the combined outcomes.
In particular, its apparent 0/32 at level 10 is unfinished work, not a failure
rate. The subsequent completed level-10 run verified 63/64.
2
## What the sign comparison tells us

Here positive means the sampled vector `x0`, and negative means `-x0`.
The sampled vector has mixed-sign coordinates; neither label identifies a
preferred algebraic direction.

| Level | Dimension | Both verify | Positive only | Negative only | Neither | Either verifies |
|---|---:|---:|---:|---:|---:|---:|
| 6 | 64 | 53 | 4 | 6 | 1 | 63/64 (98.44%) |
| 8 | 256 | 50 | 9 | 3 | 2 | 62/64 (96.88%) |
| 10 | 1,024 | 48 | 10 | 5 | 1 | 63/64 (98.44%) |
| 12 | 4,096 | 44 | 9 | 4 | 7 | 57/64 (89.06%) |
| **Total** | | **195** | **32** | **18** | **11** | **245/256 (95.70%)** |

From `x0` alone, 227/256 matrices verified (88.67%). From `-x0` alone,
213/256 verified (83.20%). Taking either sign raised coverage to 245/256.
Thus the opposite start rescued **18 of the 29 positive-start failures**;
in the other direction, the positive start rescued **32 of the 43
negative-start failures**. These are paired outcomes, not independent trials
to be added to the matrix count.

This answers the central practical question: changing the sign can change
which matrices converge. Fifty of 256 matrices (19.53%) verified from exactly
one member of the pair. Aggregate failure rates alone would hide that fact.
Negation is useful as an alternative start, but it is not a universally better
start: rescue works in both directions, and the original sign happened to
perform better overall in this sample. With a symmetric random starting
distribution, the labels have no built-in preferred orientation. These runs
do not establish a general advantage for either label.

The paired starts have exactly equal norms, so the differences cannot be
attributed to their initial distance from the origin. They still do not
separate position from direction, because negation changes both. Radius
experiments would require additional starts along the same ray.

### Opposite starts can also find different eigenpairs

Among the 195 matrices for which both signs verified, **115 produced clearly
different saved root candidates and different eigenvalue candidates** at a
relative-distance threshold of 1e-6. The counts were 27/53, 32/50, 28/48 and
28/44 at levels 6, 8, 10 and 12, respectively. The remaining 80 pairs were
within that threshold for both root and eigenvalue.

Distance here is `norm(u-v)/max(1,norm(u),norm(v))`, applied separately to
`x` and `lambda`. Both candidates in each comparison passed the original row
checks. This is numerical evidence that changing starts can explore different
solutions as well as rescue failures; it is not a certified multiplicity count
or a complete spectrum.

## How the cost grew

Each row below covers the same 64 matrices / 128 searches. Times are sums of
the two recorded solver-chunk timers, including diagnostics and final
verification, but excluding input generation and the subsequent archive writes.
Whole-run timers above include additional overhead. These are observed timings,
not repeated controlled performance benchmarks.

| Level | Dimension | Solver-chunk time | Ratio to previous sampled level | Newton corrections | Krylov iterations |
|---|---:|---:|---:|---:|---:|
| 6 | 64 | 5.057 s | — | 1,746 | 101,706 |
| 8 | 256 | 5.391 s | 1.07x | 1,893 | 112,087 |
| 10 | 1,024 | 13.739 s | 2.55x | 2,027 | 120,438 |
| 12 | 4,096 | 130.249 s | 9.48x | 2,302 | 137,784 |

Each jump here is **two doubling levels**, so the algebra dimension increases
fourfold. Runtime does not follow a uniform fourfold rule: it barely changes
from level 6 to 8, increases 2.55-fold from 8 to 10, and increases 9.48-fold
from 10 to 12. The level-12 chunks were individually 65.214 and 65.035
seconds, so the large increase is present in both chunks.

From level 10 to 12, Newton work rose about 14% and Krylov work about 14%,
while time rose nearly 9.5-fold. Multiplication calls actually fell slightly,
from 13,266 to 13,162, while the number of algebra-product rows rose from
643,468 to 734,460. Average product batch occupancy increased from 48.5 to
55.8 rows. More solver iterations or reduced batch occupancy therefore do
not explain the timing increase by themselves. The wider vectors make each
product and Krylov operation more expensive; these archives do not isolate
the relative costs or establish a general scaling law. A cause beyond that
would require focused timing or profiling, not a kernel correctness replay.

The three complete runs took **155.058 seconds combined** in their recorded
whole-run timers. The interrupted repeat is additional work, reported
separately above.

## What happened to the unresolved searches

All 11 matrices unresolved after both signs had attempts ending at the
iteration budget. **No line-search stalls or nonfinite terminations were
recorded in the complete runs.** All 512 searches reached a numerical
termination; none was stopped by the deadline or interruption.

| Level | Sample suffix | Best recorded maximum row residual |
|---|---:|---:|
| 6 | 52 | 2.929e-6 |
| 8 | 20 | 1.024e-5 |
| 8 | 25 | 4.362e-2 |
| 10 | 19 | 1.458e-7 |
| 12 | 3 | 1.362e-4 |
| 12 | 6 | 1.521e-6 |
| 12 | 17 | 2.628e-3 |
| 12 | 28 | 5.040e-7 |
| 12 | 38 | 1.584e-8 |
| 12 | 48 | 1.533e-8 |
| 12 | 53 | 1.111e-7 |

IDs have prefix `survey-20261102-level-L-sample-`. Level-12 samples 38 and
48 narrowly miss the 1e-8 acceptance threshold and were still improving.
Other histories show progress well above tolerance or local plateaus. For
example, level-12 sample 3's better branch moved from approximately 1.427e-4
to 1.362e-4 over the last four corrections, whereas sample 28's better branch
approximately halved its residual each correction over that interval.
There is no evidence here that the unresolved matrices lack eigenpairs.

The GMRES allowance was nearly fully used: mean Krylov iterations per Newton
correction rose from 58.25 at level 6 to 59.85 at level 12, against a cap of 60.
This signals heavy use of the inner budget, but does not diagnose conditioning
or establish whether more Newton steps or more GMRES work would help most.

Termination reason and acceptance are distinct. Ten individual searches
verified at the looser row acceptance threshold while ending at the iteration
budget: three at level 8, one at level 10 and six at level 12. They are correctly
counted as verified. Across all accepted searches in these complete runs, the
largest recorded row residual was **3.684e-9**, within the 1e-8 threshold.

## What is new, and what remains open

The completed level-10 and level-12 runs extend this paired GPU evidence to
matrix entries with 1,024 and 4,096 coordinates. The matrices remain 2 by 2;
they have not become 1,024-by-1,024 or 4,096-by-4,096 matrices. Strong coverage
persists at dimension 1,024, and 57 concrete verified eigenpairs were found
at dimension 4,096 under the short screening allowance.

The lower level-12 coverage is a reason to inspect solver budgets, not evidence
of a mathematical cutoff. Each level uses only 64 matrices from one seed
family. Coefficient support stays at 16, so its fraction falls from 25% at
dimension 64 to 0.391% at dimension 4,096. Dimension, relative sparsity and the
particular sampled matrices all change together. A dimension-only difficulty
claim would go beyond the experiment.

A focused next test, if desired, is to continue the 11 saved best candidates
for a small fixed number of additional Newton steps **without changing GMRES
settings**. That would measure the benefit of extra outer corrections at a
fixed inner budget, particularly for the near-threshold cases. No continuation,
broader survey, benchmark repeat or kernel validation was launched for this
analysis. The original denser survey's 1,024/1,024 result remains separate;
these are new matrices and screening outcomes.

## Evidence and checks

The [machine-readable summary](results/paired-sign-summary-20261029.json)
contains source paths and hashes, per-level outcomes, work and timing totals,
remaining-case histories, and the overlap audit. Source hashes match the
current runners. All archived input and output vector hashes were checked,
as were exact opposite nonzero starts, identical coefficients within each
pair, encoded coefficients against input arrays, chunk manifests against
aggregate records, and selected vectors against full search archives.
Both recorded row residuals were checked against acceptance for every attempt.
Candidate-distance calculations use saved vectors; no algebra products were
recomputed. Original archives remain unchanged.
