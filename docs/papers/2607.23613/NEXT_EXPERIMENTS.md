# Next experiments after the 8,192-matrix survey

This public research roadmap records proposals made after the initial survey.
Later outcomes are recorded in [the results report](BATCHED_RESULTS_REPORT.md);
these commands are optional experiments, not pending release requirements.

The completed sparse survey verified 7,620 of 8,192 matrices, leaving 572
unresolved: 285 at level 6 and 287 at level 8. Its results are preserved at
`/tmp/cayley-large-survey.mIv6eg/results`. No kernels or multiplication checks
are changed or rerun for these experiments.

The survey checkpoint improvement writes each chunk once and a small live
`status.json`, instead of repeatedly serializing the growing `results.json`.
The full aggregate is finalized on exit, including graceful interruption.
Summaries are updated incrementally. This addresses host serialization overhead;
it is not a change to GPU arithmetic and has not yet been timed on the large
GPU survey. Tests check the number of aggregate writes and checkpoint recovery.

## 1. Continue only unresolved saved candidates

This continuation needs the full original survey archive, including saved
vectors, which is not distributed in GitHub. Set `SURVEY_INPUT` to your complete
local survey directory. The historical `/tmp` path above is provenance only.
To generate a new survey archive, use [the fresh-survey workflow](FRESH_SURVEY.md).
The committed summary alone cannot reproduce this continuation.

```bash
: "${SURVEY_INPUT:?Set SURVEY_INPUT to a complete local survey archive directory}"
FOLLOWUP_ROOT=$(mktemp -d /tmp/cayley-followup.XXXXXX)
python -m scripts.retry_eigen_frontier \
  --input "$SURVEY_INPUT" \
  --output "$FOLLOWUP_ROOT/continued" \
  --backend gpu --resume-saved --starts 0 \
  --batch-size 64 --maxiter 30 --gmres-restart 20 --gmres-cycles 6 \
  --minutes 15 --progress-seconds 30
```

Only the 572 unresolved matrices are selected. This adds up to 30 Newton
corrections from their best saved candidates, with a larger inner linear-solve
budget. No new starting points are generated. The experiment asks how much
coverage can be recovered with additional correction work, rather than by
changing initial guesses. Because both Newton and inner budgets increase, it
does not isolate their separate effects.

The input is copied and hash-checked before retries. Results are saved per
chunk of at most 64 matrices, with per-matrix residuals and cumulative progress.
Ctrl-C once requests a graceful save; a second interrupt can abandon the
current chunk. A deadline stops further chunks; the final report lists
`unprocessed_matrix_ids` separately. A time-stopped candidate is not evidence
that its numerical iteration budget failed. The 15-minute search clock begins
after source backup; backup and final writes add to wall time.

The retry aggregate contains the selected unresolved subset, not all original
8,192 matrices. For overall coverage, add newly verified matrices to the original
7,620; never count the retry subset as a separate random survey.

## 2. A smaller survey with denser coefficients

```bash
DENSE_ROOT=$(mktemp -d /tmp/cayley-dense-survey.XXXXXX)
python -m scripts.survey_eigen_frontier \
  --backend gpu --levels 6,8 --matrices-per-level 512 \
  --support 16 --starts 2 --batch-size 64 --seed 20261101 \
  --maxiter 20 --gmres-restart 20 --gmres-cycles 3 \
  --minutes 15 --progress-seconds 30 \
  --output "$DENSE_ROOT/results"
```

This targets 1,024 new matrices and 2,048 starts. Each coefficient has 16 nonzero
coordinates instead of 3, and is normalized as before. This is denser but not
fully dense: 16/64 coordinates at level 6 and 16/256 at level 8. Screening
iteration budgets match the large sparse survey. A new seed avoids identifier
collisions with earlier samples. It is a comparison of independently sampled
families, not a paired experiment on the same matrices.

Both commands stop early upon completing their finite workloads. Their search
allowances total at most 30 minutes, with additional time possible for source
backup, an in-flight operation, verification and writes. Neither pads execution
to fill its allowance. The survey retains independent chunk checkpoints and
live status; the final aggregate contains per-start coverage and termination
histories. No timing repetitions or basic kernel validation are included.

## Consolidated report after these runs

Wait for the resulting directories and terminal summaries before finalizing the
report. It should explain:

- How CayleyKernel evaluates correctly bracketed Cayley–Dickson products in the
  nonlinear eigenpair search, and how true batched products accelerated the
  measured workloads.
- The distinction between kernel arithmetic, numerical search convergence and
  the mathematical existence claims discussed in the paper.
- Verified matrices versus successful starts, the benefit of random starts,
  finite-budget failures, and the recoveries from continuing saved candidates.
- The scope of the sparse and denser sampling families and any changes in
  coverage; comparisons must not silently mix solver budgets or populations.
- Measured timing evidence: roughly 10.72x at level 6 (four steady repeats) and
  17.89x at level 8 (one steady measurement), with nearly identical numerical
  work. These are workload measurements, not universal acceleration factors.
- Noteworthy observations without calling them discoveries prematurely: the
  nearly equal 93% two-start coverage at levels 6 and 8, the stronger benefit
  of the second start at level 8, and persistent local solver plateaus.
- Follow-up questions such as coefficient density, additional starts, better
  correction strategies, and broader algebra levels, guided by the new outcomes.

Paper-specific comparisons should be checked against the local paper analysis
when writing the report. Finite verified examples do not establish a universal
eigenvalue theorem, and unresolved searches do not refute one.
