# Fresh-matrix survey with incremental checkpoints

The targeted retries left three level-6 matrices unresolved after larger Krylov
spaces and twelve fresh starts. Those cases are retained in their existing
archives. This survey asks a different question: how often does the current
search procedure verify eigenpairs on a fresh reproducible collection, and how
much does a second starting point help?

It performs no kernel validation and no individual-solver benchmark repeats.
It uses the established batched solver and its final eigenpair verification.

## Run

From the repository root in a Python environment with NumPy and SciPy:

```bash
SURVEY_ROOT=$(mktemp -d /tmp/cayley-fresh-survey.XXXXXX)
python -m scripts.survey_eigen_frontier \
  --backend gpu --levels 6,8 --matrices-per-level 256 \
  --starts 2 --batch-size 64 --seed 20261030 \
  --maxiter 20 --gmres-restart 20 --gmres-cycles 3 \
  --minutes 30 --progress-seconds 30 \
  --output "$SURVEY_ROOT/results"
```

The target is **512 new matrices / 1,024 starts**. Each chunk normally contains
32 matrices and 64 starts. Levels alternate between chunks, so an early stop
still samples both levels once the first two chunks finish. The run stops when
the finite target is reached or 30 minutes expires; it does not keep computing
to fill the time allowance. Given the previous screening timings it may finish
much sooner, but this new seed collection can have different convergence costs.

Each matrix uses the same sparse coefficient family as the benchmark: four
independent normalized real vectors, each with three randomly selected nonzero
coordinates (or all coordinates when dimension is less than three). `--support`
can change that in a separate run. Conclusions apply to this sampling family,
not all possible matrices. The first start is zero; the second is an independent
normal vector scaled by the square root of dimension. All starts are evaluated
so their individual success rates and the additional coverage from the second
start can be compared. This differs intentionally from production retries,
which stop scheduling starts after success.

The RNG key is `[seed, level, sample_index]`; changing chunk size does not change
matrices or starting points. Matrix IDs and generation keys are saved. This
fresh seed stream is distinct from the earlier benchmark seed scheme. Each level
uses different matrices, so level-to-level differences are descriptive rather
than paired comparisons of the same matrices.

## Progress, stopping and saved evidence

Every batch prints START, periodic elapsed-time notices, and SAVED with verified
and unresolved matrix counts, stalled and budget-exhausted start counts,
multiplication calls and cumulative elapsed time. Counts refer to matrices or
starts as explicitly labelled.

**Press Ctrl-C once to stop gracefully.** The solver checks a stop flag between
Newton iterations, Krylov iterations and line-search trials. It discards an
unfinished correction, retains the last accepted candidate, performs the usual
final verification, and saves the current chunk before exiting. Time expiry
uses the same path. It cannot interrupt an individual GPU operation; final
verification and disk writes may take the run slightly past the time ceiling.
These are cooperative bounds, not a hard process kill.

A second Ctrl-C forces an exit. A forced kill or process error may lose the
current unfinished chunk, but cannot erase completed chunk checkpoints. Input
vectors for the active chunk are saved before solving and identified in the
live status. `run stopped` is recorded separately from numerical stalls
or exhausted iteration budgets; stopped unresolved searches have not completed
their allotted search and must not be interpreted as equivalent failures.

Output files:

- `results.json`: full aggregate written at startup and exit (including graceful
  interruption). It includes run status, per-level coverage,
  per-start coverage, additional matrices solved by later starts, termination
  reasons, residual histories, selected candidates and vector hashes.
- `status.json`: small atomic live status with completed-chunk count, saved-matrix
  count, per-level summaries and the active chunk. Read this during a run; the
  full aggregate is intentionally not rewritten per chunk.
- `chunk-N.json`: independent checkpoint containing that chunk's completed
  matrix records; written before updating live status.
- `chunk-N-inputs.npz`: all coefficients and initial candidates for the chunk.
- `chunk-N-searches.npz`: all final root/lambda candidates in start order.
- `level-L-sample-S.npz`: best candidate per matrix, selected by verification
  status first, then residual. These records are compatible with
  `scripts.retry_eigen_frontier.py` for subsequent targeted retries.

The full manifest now has exactly two writes in a normal run, regardless of
chunk count. Per-level summaries are accumulated from each new chunk rather
than rescanning all prior matrix records. This removes the expanding aggregate
serialization cost without changing solver arithmetic or checkpoint frequency.

After a forced kill, a separate aggregate can be rebuilt from committed chunk
files without recomputing products. For example (replace the path):

```bash
python - <<'PYRECOVER'
from pathlib import Path
from scripts.survey_eigen_frontier import recover
source = Path('/tmp/your-survey/results')
recover(source, source/'recovered-results.json')
PYRECOVER
```

This refuses an existing recovery output, verifies selected vector hashes, and
leaves the original manifest and checkpoints unchanged. Recovery contains only
committed chunks, not unfinished work. The recovered JSON resides alongside the
vectors it references.

No existing output directory is overwritten. Checkpoints are written after each
chunk; JSON replacement is atomic and flushed before replacement. This protects
completed results during normal interrupts; it is not a guarantee against disk
failure. Interrupted runs are not automatically resumed. Their saved matrix
candidates can be supplied to the retry runner, and their per-chunk inputs retain
what is needed to reproduce searches.

## Read the results before expanding

Send the SAVED/summary output and output directory back for analysis. Useful
questions are how many unique matrices verify, how many are rescued by the
second start, and whether remaining candidates stall or exhaust their budgets.
Keep the finite run and the stopped-search counts when comparing coverage.
Do not increase the workload simply because time remains in the allowance.

Workflow tests (no kernel validation):

```bash
python -m pytest -q \
  tests/papers/arxiv_2607_23613/test_survey_frontier.py \
  tests/papers/arxiv_2607_23613/test_eigen_workflow.py \
  tests/papers/arxiv_2607_23613/test_retry.py
```
