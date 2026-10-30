# Eigen search audit — October 29, 2026

The audit uses the user's three saved GPU screening archives under
`/tmp/cayley-eigen-screen.xkI9Ai`. It checks archive hashes, vector shapes and
finiteness, per-start recorded residuals, matrix identifiers, iteration budgets,
and agreement across solvers and repetitions. It does not execute multiplication
or recompute eigen equations. The kernel is unchanged. The user separately
reported the GPU workflow/product test passing in 0.51 seconds.

The full audit and retry input was saved locally at
`scripts/results/eigen-benchmark-audit-20261029/results.json`. That ignored
archive is **not included in GitHub**. It includes copies of the source archives
under `original/` and selects one best saved candidate for each unique matrix,
with source measurement/start provenance. The committed
[compact audit snapshot](results/gpu-eigen-audit.json) records the audit findings;
it does not include all vectors required to repeat the audit or retries.

| Level | Unique matrices | Verified matrices | Unresolved matrices | Verified starts per measurement |
|---|---:|---:|---:|---:|
| 6 | 32 | 26 | 6 | 45 / 64 |
| 8 | 32 | 29 | 3 | 47 / 64 |

There are **no per-start success disagreements**, including across the separate
level-6 archives. Among jointly verified starts, maximum relative differences
between individual and batched saved roots are approximately 1.18e-10 at level 6
and 7.93e-16 at level 8. Different valid roots would also have been acceptable.
The maximum recorded verified row residual is 3.27e-9, below the 1e-8 acceptance
threshold. None of these findings calls for another kernel validation run.

Steady-state medians are 19.710762 vs 1.839331 seconds at level 6 (four repeats,
10.72x speedup), and 35.913041 vs 2.007397 seconds at level 8 (one steady sample,
17.89x speedup). These apply to this screening workload and its budgets.

## Unresolved candidates

| Level | Matrix seed | Best recorded eigenpair residual |
|---|---:|---:|
| 6 | 29619 | 0.419892 |
| 6 | 29620 | 0.200493 |
| 6 | 29634 | 0.239559 |
| 6 | 29642 | 0.448534 |
| 6 | 29643 | 0.157637 |
| 6 | 29644 | 0.254505 |
| 8 | 31614 | 0.0000537734 |
| 8 | 31618 | 0.000923293 |
| 8 | 31624 | 0.00891111 |

Every selected candidate consumed 15 Newton corrections and 900 Krylov
iterations: the full 15 × 20 × 3 screening allowance. This supports trying more
correction work; it does not establish that more iterations will converge.
At the start level, 16 of 19 unresolved level-6 starts reached the Newton limit,
while 3 terminated earlier. All 17 unresolved level-8 starts reached the limit.
Those counts include failed siblings of matrices that another start solved;
only the nine matrices above warrant retries. Old archives lack solver reasons
and full histories, so an exact cause of those three earlier terminations cannot
be recovered from the saved iteration totals alone. Future benchmarks record
these diagnostics.

## Next GPU run: continue only these nine candidates

From the repository root, set `AUDIT_INPUT` to a complete local audit archive
(the directory containing `results.json` and its referenced vectors). Obtain
that archive separately or generate one with the audit command below. The
compact committed snapshot cannot serve as retry input.

```bash
: "${AUDIT_INPUT:?Set AUDIT_INPUT to a complete local audit archive directory}"
RETRY_ROOT=$(mktemp -d /tmp/cayley-unresolved.XXXXXX)
python -m scripts.retry_eigen_frontier \
  --input "$AUDIT_INPUT" \
  --output "$RETRY_ROOT/continued" \
  --backend gpu --resume-saved --starts 0 \
  --batch-size 64 --maxiter 30 --gmres-restart 20 --gmres-cycles 6 \
  --progress-seconds 30
```

This adds up to 30 Newton corrections per remaining candidate, with up to 120
Krylov iterations per correction, starting from the best archived vector. It
runs once, without timing repetitions or new random starts. Search histories,
termination reasons, previous and final residuals, and verified vectors are
saved. Only six rows at level 6 and three at level 8 are available, so these
retries do not have the occupancy of the 64-search throughput benchmark.
A batch-size cap of 64 does not imply 64 actual rows here.

Inspect that result before further work. If candidates remain unresolved, fresh
starts can target only those matrices with a separate run:

```bash
python -m scripts.retry_eigen_frontier \
  --input "$RETRY_ROOT/continued" --output "$RETRY_ROOT/fresh-starts" \
  --backend gpu --starts 4 --seed 20261029 \
  --batch-size 64 --maxiter 30 --gmres-restart 20 --gmres-cycles 6 \
  --progress-seconds 30
```

The retry runner backs up its input and refuses existing output directories.
Failures remain unresolved, not counterexamples to eigenvalue existence.
The nine-candidate GPU continuation has not been run by the agent.

## Reproduce the audit without solver computation

This requires the three complete original screening archives, which are not
included in GitHub. Set `SCREEN_ARCHIVE_ROOT` to their parent directory. The
historical `/tmp` path above identifies the recorded run, not a downloadable
location. To create new screening archives, follow [the benchmark workflow](BATCHED_EIGEN.md).

```bash
: "${SCREEN_ARCHIVE_ROOT:?Set SCREEN_ARCHIVE_ROOT to your screening archive directory}"
AUDIT_ROOT=$(mktemp -d /tmp/cayley-audit.XXXXXX)
python -m scripts.audit_eigen_benchmarks \
  --input \
    "$SCREEN_ARCHIVE_ROOT/level-6-batch-64-repeat-4" \
    "$SCREEN_ARCHIVE_ROOT/level-6-batch-64-level-8" \
    "$SCREEN_ARCHIVE_ROOT/level-6-batch-64" \
  --output "$AUDIT_ROOT/report"
```
