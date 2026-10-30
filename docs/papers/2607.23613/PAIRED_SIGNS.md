# Paired starting-point sign experiment

This experiment asks whether opposite initial guesses rescue different matrices
under identical correction budgets. It uses the established batched solver;
solver arithmetic and the original bracketed eigenpair equation are unchanged.

For every newly generated matrix, draw one nonzero random normal starting
vector `x0`, scaled by the square root of dimension, and search from **both
`x0` and `-x0`**. Both searches run even if the first succeeds. Neither starts
from a previously discovered root, and matrices are not selected according to
whether an earlier search succeeded. The labels positive and negative identify
the two members of the pair; they do not mean all coordinates have that sign.
Each matrix and pair has a reproducible RNG key independent of chunk size.

## Run

From the repository root, in a Python environment with NumPy and SciPy:

```bash
PAIR_ROOT=$(mktemp -d /tmp/cayley-paired-signs.XXXXXX)
python -m scripts.survey_eigen_frontier \
  --paired-signs --starts 2 \
  --backend gpu --levels 6,8 --matrices-per-level 64 \
  --support 16 --seed 20261102 --batch-size 64 \
  --maxiter 20 --gmres-restart 20 --gmres-cycles 3 \
  --tol 1e-9 --minutes 5 --progress-seconds 30 \
  --output "$PAIR_ROOT/results"
```

The finite target is **128 matrices and 256 searches**, with 64 matrices per
level. The two signs receive the same screening budgets as the original denser
survey. This is a new paired experiment, not a continuation or an addition to
the earlier survey's coverage. No benchmark or kernel validation is performed.
Five minutes is a cooperative ceiling, not a target runtime. Ctrl-C once saves
the current candidates and exits gracefully.

## Read the result

Each level reports five mutually exclusive counts in `paired_sign_outcomes`:

| Outcome | Meaning |
|---|---|
| `both_verified` | Both signs yielded accepted eigenpairs |
| `positive_only` | Only the search starting at `x0` verified |
| `negative_only` | Only the search starting at `-x0` verified |
| `neither_verified` | Neither search verified after its numerical termination |
| `incomplete` | At least one search was stopped by interruption or deadline |

For completed pairs, positive-start failures are `negative_only + neither_verified`;
negative-start failures are `positive_only + neither_verified`. Divide by the
number of completed pairs to obtain failure percentages. The discordant counts
(`positive_only` and `negative_only`) reveal rescues hidden by similar overall
failure percentages. Equal totals can conceal very different failing matrices.

Because the distribution of `x0` is symmetric about zero, neither sign has a
preferred orientation in the sampling design. Similar success rates with many
discordant pairs would support trying the opposite sign as a useful second
start. The sign test does not separate position from direction: both change
under negation. Testing radius would require additional paired starts such as
`x0/2` and `2*x0`. Finite counts do not prove symmetry for every matrix or an
optimal start policy, and the two searches can converge to different roots.

Acceptance still requires both original row residuals finite and at most 1e-8
at solver tolerance 1e-9. Unresolved searches do not establish nonexistence.

The existing survey archive format retains matrix IDs and coefficients,
`chunk-N-inputs.npz` with both starting vectors, `chunk-N-searches.npz` with
both final candidates, and per-sign residual histories, reasons and work counts
in the JSON records. `status.json` carries live summaries; `results.json` is
written at startup and exit. Independent chunk checkpoints preserve completed
work. The recovery procedure in [FRESH_SURVEY.md](FRESH_SURVEY.md) also rebuilds
paired outcome counts. Existing output directories are refused.

Return the final console summaries and results path for analysis of both
coverage and per-sign work. No additional runs should be scheduled merely
because time remains.

## Recorded results

The [October 29 results and interpretation](PAIRED_SIGN_RESULTS_20261029.md)
cover the completed level-6/8 trial, additional completed trials at levels 10
and 12, and an interrupted overlapping run. Across 256 distinct completed
matrices, 245 verified; 50 verified from only one of the opposite starts.
