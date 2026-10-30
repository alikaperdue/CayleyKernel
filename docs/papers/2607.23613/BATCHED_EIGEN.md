# Batched eigenpair searches

`paper_quadratic_frontier.py --experiment eigen --solver batched` and
`retry_eigen_frontier.py --solver batched` use the coordinated solver in
`scripts/batched_eigen_frontier.py`. Eigen experiments and retries now default to `--solver batched` with batch size
64. Root experiments retain their individual solver. Explicit `--solver individual`
retains SciPy Newton/GMRES and the paper's continuation/transformation diagnostic path.
Its default numerical parameters are unchanged; the Newton helper additionally
reports Krylov iterations and accepts explicit correction budgets.
The batched experiment directly solves the original quadratic equation; it does
not reproduce the transformed-equation diagnostics. Compare performance with the
benchmark below, not by timing these two different experimental pipelines.

## Computation and convergence

Matrices are generated with the same `seed + 1000*level + sample` coefficients
as before, grouped by dimension, then processed in chunks of `--batch-size`.
Each GPU product receives two **CuPy arrays of shape `(rows, 2**level)`**.
The final axis alone denotes algebra coordinates. No GPU product loops over
individual candidates. The CPU implementation is a deliberately slow recursive
reference, limited to dimension 256; it is not a performance fallback.

Each Newton iteration computes and caches `BX = multiply(B, X)` and evaluates

```python
F = multiply(X, BX) + multiply(X, A) - multiply(D, X) - C
JH = (multiply(H, BX) + multiply(X, multiply(B, H))
      + multiply(H, A) - multiply(D, H))
```

Restarted GMRES maintains a separate Arnoldi basis, Hessenberg system, Givens
rotations, and residual target for each row. The inner Jacobian applications
batch all still-active Krylov rows. Finished rows are excluded from subsequent
products; basis storage is retained until the restart boundary. Two passes of
modified Gram–Schmidt improve numerical stability. Triangular corrections are
computed with vectorized row operations, without per-row host linear solves.
True linear residuals are checked at restart boundaries. This implementation
can take a different numerical trajectory from SciPy GMRES; identical roots
or identical iteration counts are not assumed.

Line search batches pending trials; each row accepts its own first decreasing
step. Accepted rows stop participating in that line search. Newton-active row
indices shrink after convergence, nonfinite residuals, or stalls. Partial final
chunks work identically. Coefficients and solution storage remain allocated
within the current chunk; active gathers compact the working views. There is
no refill from later chunks while a chunk is finishing, so very uneven search
lengths can still underutilize the GPU.

Vectors, Krylov bases, and correction algebra stay on the device within a
chunk. Only row IDs and scalar residual diagnostics cross to the host per
Newton iteration, and vectors cross at completion. Dynamic compaction and
inner-loop termination checks still require synchronization; this is not a
fully asynchronous solver. Basis memory is roughly
`8 * batch_size * (min(gmres_restart, dimension)+1) * dimension` bytes, plus
coefficient arrays, Hessenberg systems, and temporaries. Reduce the batch size
or restart length if device memory is tight.

Fresh runs use a deterministic random candidate after coefficient generation.
Retry rounds use the historical negative-saved, zero, perturbed-saved, and
random-direction start sequence. Each matrix has its own generator seeded by
`restart_seed = CLI_seed + matrix_seed`; compaction does not change that
sequence. One start per unresolved matrix is scheduled per round, stopping
further starts as soon as an eigenpair verifies. This keeps starts from solved
matrices out of later rounds without speculative work on all their siblings.

Final verification explicitly evaluates both original rows for `(1, x)` in
the paper's convention:

```text
a*1 + b*x - lambda*1
c*1 + d*x - x*lambda
```

Both norms must be finite and at most `10*tol`. Results retain matrix IDs,
seeds, coefficients, solver histories, both row residuals, root/lambda vectors,
and SHA-256 hashes. Fresh batched runs also save the initial candidate. Nonfinite residual diagnostics are JSON `null`; all failed
or exhausted searches are **unresolved**, never evidence of nonexistence.
`iterations` is the last evaluated Newton iteration index; `newton_steps` counts
attempted corrections (including a final failed line search).

Outputs require a new directory. Retry runs copy and hash-check the entire
source tree under `original/` before searching. Source and historical results
are never changed. Retry vector filenames include a case index to avoid
collisions. Batched experiment times are reported per dimension group;
per-result `save_seconds` measures serialization only.

## Correctness checks

From the repository root, activate the environment containing NumPy, SciPy,
pytest, and (for GPU checks) CuPy and the built kernel:

```bash
# Activate your Python environment before running this command.
python -m pytest -q \
  tests/papers/arxiv_2607_23613/test_frontier.py \
  tests/papers/arxiv_2607_23613/test_batched_frontier.py
```

Tests compare batched residuals/Jacobians with individual signed-permutation
calculations, recursive doubling, finite differences, and dense independent
Jacobian solves on small cases. They cover restarted GMRES, mixed convergence,
compaction, partial chunks, nonfinite inputs, iteration budgets, retry seeds,
archive preservation, benchmark output, and both original rows. The GPU test
asserts real 2D CuPy inputs and device outputs, and compares nonassociative
sedenion products with the independent CPU reference:

```bash
python -m pytest -q -rs \
  tests/papers/arxiv_2607_23613/test_batched_frontier.py \
  -k gpu_uses_real_2d_inputs
```

Confirm that this test **passes rather than skips** on the CUDA machine.

## Run experiments and retries

```bash
RUN_ROOT=$(mktemp -d /tmp/cayley-batched.XXXXXX)
python -m scripts.paper_quadratic_frontier \
  --experiment eigen --solver batched --backend gpu \
  --levels 3,4,6,8 --samples 64 --starts 4 --seed 23613 \
  --batch-size 32 --maxiter 40 --gmres-restart 40 --gmres-cycles 15 \
  --line-search 12 --output "$RUN_ROOT/eigen"

python -m scripts.retry_eigen_frontier \
  --input "$RUN_ROOT/eigen" --output "$RUN_ROOT/retry" \
  --solver batched --backend gpu --batch-size 32 --starts 12 \
  --seed 20261028 --maxiter 60 --gmres-restart 40 --gmres-cycles 15
```

All correction controls also apply to the individual direct Newton solve.
The original continuation stage retains its historical parameters. `--starts`
on the paper runner applies to batched direct searches only; individual mode
retains its single continuation-derived start.

## Fair benchmark

The benchmark runs the existing individual Newton/GMRES implementation and
the coordinated implementation on **identical coefficients and initial vectors**,
with equal tolerance, Newton, Krylov, and line-search budgets. Every starting
point is evaluated in both paths, even if a sibling start succeeds, so neither
path receives easier work due to scheduling. It reports both search counts and
unique matrix counts. This is a solver-throughput benchmark, not the timing of
an early-stop multistart policy.

```bash
BENCH_ROOT=$(mktemp -d /tmp/cayley-eigen-bench.XXXXXX)
python -m scripts.benchmark_eigen_frontier \
  --backend cpu --level 3 --matrices 4 --starts 2 \
  --batch-size 3 --repeats 3 --output "$BENCH_ROOT/cpu"

# Start here on the GPU; inspect this result before expanding the workload.
python -m scripts.benchmark_eigen_frontier \
  --backend gpu --screen --progress-seconds 30 \
  --output "$BENCH_ROOT/gpu-level-6-screen"
```

The `--screen` preset uses level 6, 32 matrices × 2 starts (64 searches),
batch size 64, 15 Newton steps, GMRES restart 20 with 3 cycles, and one
steady-state repeat. It runs four measurements: first-use and steady-state for
each solver. These reduced budgets apply equally to both solvers; success counts
are not directly comparable with earlier full-budget runs. Explicit CLI options
override preset values. This is a screening configuration, not a runtime guarantee.
Ordinary runs without `--screen` retain their previous defaults.

Every console line identifies backend, level, dimension, batch size, solver,
and measurement phase/repeat where applicable. START is printed before each
measurement, RUNNING gives elapsed wall time every 30 seconds, and DONE includes
counts and iteration totals. The heartbeat is an elapsed-time notice, not an
estimate of convergence or completion, and does not inspect or synchronize CUDA.
Use `--progress-seconds 0` to disable it. Printing can add a small timing overhead.
Ctrl-C during a solve records `status: interrupted` and the active measurement;
completed measurements remain saved and the unfinished measurement is excluded.

After reviewing the screen, a larger run can be requested explicitly with, for
example, `--screen --matrices 64 --maxiter 40 --gmres-restart 40
--gmres-cycles 15 --repeats 3`, always using a new output directory. Do not start
the previous level/batch sweep automatically: level 6 at batch size 1 took about
68 minutes for one batched measurement on the user's GPU. The supplied level-4
measurements showed about 3× speedup at batch 64, but do not establish level-6
performance. Review screen timings and verified/unresolved counts first.

`results.json` reports device setup and input-generation times separately,
first-use time for each solver, alternating-order steady-state measurements,
medians, verified/unresolved counts, multiplication-call and row totals, batch
size histograms, per-search Newton corrections and Krylov iterations, and both
row residuals. Inputs and each run's output vectors are archived with hashes.
Timing includes coefficient plan construction, transfers, solving, diagnostics,
and final verification; it excludes input generation and disk writes. The
current CUDA stream is synchronized immediately before and after each timed
solve. First-use runs share a CUDA context: they are not independent process-cold
measurements. Module import cost is outside the reported setup timer.

GPU multiplication counts include every actual `CayleyKernel.multiply` call.
For CPU comparisons, counts also include historical fixed signed-permutation
plan applications, separately identified. The batched CPU oracle is much slower
than those plans, so CPU timings must not be extrapolated to GPU throughput.
Compare the measured times together with success rates and iteration counts;
batching alone is not evidence of a speedup.

Local CPU evidence (4 octonion matrices, 2 starts each, batch size 3, default
budgets, 2 steady-state repeats): both implementations verified 7/8 starts;
individual median was about 0.113 s and batched median about 0.316 s, with
2,840 versus 1,833 multiplication applications. CUDA execution was unavailable:
`cudaErrorNoDevice`. Subsequent GPU measurements and the archived-outcome audit are documented in
[BATCHED_AUDIT_20261029.md](BATCHED_AUDIT_20261029.md).


## Routine workflow regression tests

After validating the unchanged kernel, use these targeted tests for changes to
search scheduling, defaults, audit selection, continuation and archival behavior:

```bash
python -m pytest -q \
  tests/papers/arxiv_2607_23613/test_eigen_workflow.py \
  tests/papers/arxiv_2607_23613/test_retry.py
```

These are workflow tests, not a repeat of the library multiplication validation.
`--resume-saved` on the retry runner continues saved candidates before the
historical fresh-start sequence. With `--starts 0`, it performs continuation
only. Initial saved candidates are checked first, so already verified matrices
are not solved again. The original fresh-start RNG sequence is unchanged.
Future benchmarks save solver termination reasons and residual histories in
`solver_diagnostics` for investigating stalls versus exhausted budgets.

For a bounded new-seed survey, see [FRESH_SURVEY.md](FRESH_SURVEY.md). It saves
independent chunk checkpoints, supports graceful Ctrl-C and a time ceiling,
and measures matrix coverage and the benefit of a second start.
