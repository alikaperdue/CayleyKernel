# Searching for clique growth

From the repository root, run:

```bash
python scripts/clique_growth.py
```

The experiment implements the levels **32, 33, 34, 35** plan with **512
candidates and five trials per level**. Each trial has a free search and
an anchored extension search, for 40 phases in the default plan. It stops
when the plan finishes or its soft **ten-minute total budget** is reached,
whichever comes first. Results appear after every completed phase.
Python 3.10 or newer is required. No GPU or extra graph package is needed.

For a longer exploration within approximately ten minutes, repeat the plan
with fresh trial IDs:

```bash
python scripts/clique_growth.py --rounds 100 --minutes 10
```

The budget includes initial verification, graph construction, clique search,
certificate verification, and output. It is checked before each phase;
the current phase finishes and saves its witness even if it crosses the
deadline. Each exact clique search has its own five-second budget by
default. These limits do not provide a strict wall-clock deadline on a
slow machine or with very large user-supplied samples.

## What each phase does

**Free search:** sample 512 canonical dyads `e_a +/- e_b` with a shared
support XOR chosen separately for each trial. This uses the same seeded
sampler as `relation_graphs.py`, so level 32, seed 2026, trial 2 reproduces
the earlier 69-clique. A completed exact search proves the maximum only
inside this finite sample. Isolated candidates without a witnessed
annihilator are excluded.

**Anchored extension:** select the largest clique already known at this
level or a lower level, starting from the saved 69-clique. Embed its
elements unchanged at the current level. Generate up to 512 additional
dyads with the same support XOR. At most half the pool comes from
translating anchor supports by XOR with the highest basis-index bit; the
rest uses random full-width supports. Translation is only a candidate
generator, not an assertion that products vanish.

Retain candidates that annihilate **every** anchor element in both orders,
then search their induced graph for a clique. Adding that clique to the
anchor produces a verified extension that preserves every anchor member.
Every improvement becomes available as an anchor for subsequent searches
at the same or higher levels. Higher-level witnesses are never used at a
lower level. An exact extension maximum applies only to extensions of
that particular anchor using that particular candidate pool.

`added=25 eligible=27`, for example, means 27 candidates annihilated the
entire anchor, and 25 of them could also annihilate one another and were
added to it. `added=0` means this pool supplied no extension; it does not
prove that the anchor cannot be extended elsewhere.

Every reported clique is validated with full `basis` products and
again using an independent iterative evaluation of the canonical doubling
definition. All ordered products between distinct members must vanish.
The independent oracle has a bounded cache and does not call `twist` or
`basis`. No dimensional growth formula is assumed.

## Saved files

Each invocation creates a new `clique_growth_results/run-.../` directory,
leaving earlier runs intact. The absolute path is printed at startup.

| File | Contents |
| --- | --- |
| `configuration.json` | Levels, seed, sampling settings, budgets, and baseline size |
| `baseline_clique.json` | The exact starting witness, preserved when the best improves |
| `trials.jsonl` | One record per completed phase, including its full clique certificate |
| `best_clique.json` | The largest verified witness found, initially the baseline |
| `summary.json` | Current progress, per-level results, best size, and final stopping reason |

Support indices are zero-based hexadecimal strings. Each element is an
array of `[index_hex, coefficient]` pairs. The best witness can be used as
a baseline in a future run:

```bash
python scripts/clique_growth.py --baseline clique_growth_results/run-EXAMPLE/best_clique.json --levels 35,36,37,38
```

Requested levels must be at least the baseline's level. Full reproducibility
of the candidate sequences also requires the same baseline, seed, levels,
candidate count, trial count, and rounds. Timed-out searches can find
different lower bounds on machines of different speeds; subsequent
extensions can then differ because they use the discovered witnesses.

Use `--output-dir PATH` to place run directories elsewhere. Pressing Ctrl+C
preserves completed phases and the best saved witness. Repeated invocations
start new runs rather than automatically resuming previous trial IDs.

## Interpretation

A verified clique of size `k` at level `n` proves that the ambient
orthogonality graph for the standard real algebra of dimension `2^n`
has clique number at least `k`. The standard embedding gives the same lower
bound at every higher level. Finding bigger cliques through sampling does
not by itself prove a rule for increasing that bound with the level, nor
establish a largest published example.

The default 40-phase plan was run in about **40 seconds** in the development
environment. Its verified extension sizes after each level were:

| Level | Largest extension found |
| --- | ---: |
| 32 | 157 |
| 33 | 216 |
| 34 | 285 |
| 35 | 323 |

The [323-element witness](results/clique_growth_32_35_best.json) and
[run summary](results/clique_growth_32_35_summary.json) are saved alongside
the earlier certificates. All **104,006 ordered products** between distinct
members of the final clique were checked by both verification methods.
The extension sizes also reflect accumulated search effort, not just the
changing level; they are not evidence by themselves of a dimensional growth
formula. Runtime varies by machine and by search settings.

The focused experiment omits graph diameters: the immediate question is
whether the cliques can be enlarged. It also avoids repeating the dot and
commutativity controls that are redundant for this dyad sampler.

Regression checks (pytest required):

```bash
python tests/test_clique_growth.py
python -m pytest -q tests/test_clique_growth.py tests/test_relation_graphs.py
```
