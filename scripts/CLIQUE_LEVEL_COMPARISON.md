# Controlled comparison of clique growth across levels

Run from the repository root:

```bash
python scripts/clique_level_comparison.py
```

This answers a different question from the accumulating search in
`clique_growth.py`: **does searching higher levels help more than spending
the same candidate budget at level 32?** The earlier program remains
available for accumulating large witnesses across levels.

Defaults are levels **32--35**, **512 candidates per extension step**,
**five steps**, and **three independent seeded replicates**. Each level
and replicate starts from the **exact same original 69-clique**. A result
is carried forward only inside its own level/control chain. There is no
transfer between levels, controls, or replicates.

## Experimental arms

| Arm | Ambient level | Candidate support indices | Purpose |
| --- | ---: | --- | --- |
| `native-32` | 32 | Below 2^32 | Fixed-level control, extended repeatedly |
| `native-33` | 33 | Below 2^33 | Higher-level treatment |
| `native-34` | 34 | Below 2^34 | Higher-level treatment |
| `native-35` | 35 | Below 2^35 | Higher-level treatment |
| `embedded-33` | 33 | Below 2^32 | Check the unchanged level-32 subalgebra |
| `embedded-34` | 34 | Below 2^32 | Same embedding check at level 34 |
| `embedded-35` | 35 | Below 2^32 | Same embedding check at level 35 |

There are **105 phases**: seven arms times five steps times three replicates.
Each chain gets 2,560 candidate evaluations when all five steps complete.
Candidates in a pool are distinct and exclude that chain's current clique;
an unsuccessful candidate can be sampled again at a later step.

Each pool consists of uniformly sampled canonical `e_a +/- e_b` dyads with
the original witness's fixed support XOR. The translated-support heuristic
in the earlier experiment is omitted, so changing the highest index bit
cannot change the amount of heuristic sampling at different levels.
Separate pseudorandom streams couple the low support bits, high support
bits, and signs. An embedded control receives exactly the level-32 pool
whenever its clique agrees with the fixed-level control.

The higher-level pools do not require every candidate to use a new high
coordinate: they sample uniformly throughout their larger support space.
Thus this compares these explicitly defined sampling procedures, rather
than exhaustively comparing the whole algebras.

Every arm has the same candidate count, extension algorithm, and five-second
clique-search cap. This is equal **candidate/search-budget effort**, not
equal CPU time: higher-level integer processing and certificate verification
can take longer. The independent doubling oracle verifies all reported
witnesses in their actual ambient levels, in addition to the full
`basis` checks.

## Reading the comparison

For example:

```text
rep=0 step=2 native-33: anchor=110 added=8 clique=118 candidates=512 verified
  versus level 32: 33:+4, 34:-2, 35:+7
```

The differences compare the accumulated clique sizes **after the same number
of steps in the same replicate**. A positive value means that higher-level
chain found more elements than the level-32 control at that point. A
negative value means the fixed-level control did better. Each chain remains
monotone internally; differences between chains can increase or decrease.

The summary reports the final matched-step differences separately for each
replicate, their mean, and their range. Steps in the same chain are
dependent observations; they are not counted as independent replicates.
Three replicates are exploratory and do not establish a general advantage,
a growth law, or an ambient clique maximum.

The embedded controls should have **identical witnesses** to `native-32`
when both entire histories completed exact searches. The program checks
this and stops on a mismatch. Timed-out searches can return different
lower bounds, so their embedding comparisons are marked inconclusive.
`history_exact` means every candidate-pool extension in that chain was
solved exactly; it never means an ambient maximum was proved.

## Budgets and saved files

The default soft budget is ten minutes. The program starts a complete
matched block of seven arms only while budget remains, then lets that
block finish. It can therefore exceed the budget by one block. This keeps
candidate counts balanced for every completed comparison. Arm order rotates
between steps and replicates. If interrupted during a block, its completed
phases are retained, but that partial block is excluded from comparisons.

Every invocation creates a fresh directory below
`clique_level_comparison_results/`, containing:

- `configuration.json`: sampling scheme, arms, and effort budgets.
- `baseline_clique.json`: the original starting witness.
- `trials.jsonl`: per-phase measurements and complete verified witnesses.
- `best_clique.json`: the largest verified witness observed.
- `summary.json`: progress, matched differences, embedding checks, and status.

For a longer comparison within the same soft budget:

```bash
python scripts/clique_level_comparison.py --replicates 10 --minutes 10
```

Use `--vertices`, `--steps`, `--levels`, `--seed`, and `--output-dir` to
change the design. The baseline's own level is always included as the
fixed-level control even if omitted from `--levels`. If the budget expires
midway through the design, endpoint comparisons use the last complete
matched step in each replicate and record which step that was; do not
compare replicate endpoints at unequal steps as if their budgets matched.
Prefer a run that completes the specified design when drawing comparisons.

Tests (pytest required):

```bash
python tests/test_clique_level_comparison.py
python -m pytest -q tests/test_clique_level_comparison.py tests/test_clique_growth.py tests/test_relation_graphs.py
```

## Completed validation experiment

All 105 phases of the default design completed. Final sizes were:

| Replicate | Level 32 control | Level 33 | Level 34 | Level 35 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 198 | 198 | 198 | 198 |
| 1 | 219 | 219 | 219 | 219 |
| 2 | 193 | 193 | 193 | 193 |

All 45 exact embedded-control checks produced identical witnesses to their
level-32 controls. The higher-level native candidates did use higher support
indices; their final *sizes* matched despite having different supports.
The [summary](results/clique_level_comparison_summary.json) and
[largest witness](results/clique_level_comparison_best.json) are saved.

Thus this experiment found no higher-level benefit under this baseline,
fixed-XOR dyad sampler, and matched budget. It does not establish equality
of ambient clique numbers. These are five-step chains from independent
restarts; their sizes should not be compared directly with the earlier
323-clique's longer accumulating history as an estimate of algorithm quality.
