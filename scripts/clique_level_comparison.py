#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare levels with equal candidate budgets and independent clique histories.

Run: python scripts/clique_level_comparison.py
Default: levels 32--35, 512 candidates/step, 5 steps, 3 replicates, soft 10 min.
"""

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import clique_growth as growth
from scripts.relation_graphs import budget, levels_arg, positive


def uniform_pool(anchor, base_level, support_level, count, seed, replicate, step):
    """Uniform canonical same-XOR dyads, without the old translation heuristic.

    Low support bits and relative signs use streams shared across levels.
    Additional high bits use a separate stream. Embedded controls therefore
    receive exactly the same pool as the baseline control when anchors agree.
    """
    mask = anchor[0][0][0] ^ anchor[0][1][0]
    excluded = set(anchor)
    dim = 1 << support_level
    available = dim - 2 - sum(x[1][0] < dim for x in excluded)
    if count > available:
        raise ValueError("not enough distinct dyads for the requested equal candidate budget")
    prefix = f"compare:{seed}:{replicate}:{step}"
    low_rng = random.Random(prefix + ":low")
    high_rng = random.Random(prefix + ":high")
    sign_rng = random.Random(prefix + ":sign")
    if dim <= 4 * (count + len(anchor) + 1):
        all_vertices, _ = growth.candidates(support_level, dim - 2, low_rng, xor_mask=mask)
        choices = sorted(set(all_vertices) - excluded)
        return sorted(low_rng.sample(choices, count))
    pool = set()
    while len(pool) < count:
        a = (low_rng.getrandbits(base_level)
             | (high_rng.getrandbits(support_level - base_level) << base_level))
        b = a ^ mask
        sign = sign_rng.choice((-1, 1))
        if not a or not b:
            continue
        a, b = sorted((a, b))
        x = ((a, 1), (b, sign))
        if x not in excluded:
            pool.add(x)
    return sorted(pool)


def make_arms(levels, base_level):
    arms = []
    for level in sorted(set(levels) | {base_level}):
        arms.append({"id": f"native-{level}", "level": level, "support_level": level,
                     "role": "fixed-level-control" if level == base_level else "higher-level"})
        if level != base_level:
            arms.append({"id": f"embedded-{level}", "level": level, "support_level": base_level,
                         "role": "embedding-control"})
    return arms


def comparison_summary(records, arms, base_level):
    """Compare complete matched blocks only; steps within a chain are dependent."""
    grouped = {}
    for row in records:
        grouped.setdefault((row["replicate"], row["step"]), {})[row["arm"]] = row
    control = f"native-{base_level}"
    complete = {key: rows for key, rows in grouped.items() if len(rows) == len(arms)}
    pairs = []
    embedding_checks = []
    for (replicate, step), rows in sorted(complete.items()):
        for arm in arms:
            if arm["role"] == "higher-level":
                target = rows[arm["id"]]
                pairs.append({"replicate": replicate, "step": step, "level": arm["level"],
                              "control_size": rows[control]["size"], "higher_level_size": target["size"],
                              "difference_from_control": target["size"] - rows[control]["size"],
                              "both_histories_exact": target["history_exact"] and rows[control]["history_exact"]})
            elif arm["role"] == "embedding-control":
                target, reference = rows[arm["id"]], rows[control]
                exact = target["history_exact"] and reference["history_exact"]
                identical = (target["certificate"]["vertices"] == reference["certificate"]["vertices"]
                             if "certificate" in target and "certificate" in reference else None)
                embedding_checks.append({"replicate": replicate, "step": step, "level": arm["level"],
                                         "both_histories_exact": exact, "identical_witness": identical,
                                         "passed": identical if exact else None})
    last_step = {}
    for replicate, step in complete:
        last_step[replicate] = max(step, last_step.get(replicate, -1))
    endpoints = [p for p in pairs if p["step"] == last_step[p["replicate"]]]
    stats = {}
    for level in sorted({p["level"] for p in endpoints}):
        selected = [p for p in endpoints if p["level"] == level]
        deltas = [p["difference_from_control"] for p in selected]
        stats[str(level)] = {"replicates": len(deltas), "differences": deltas,
                             "mean_difference": statistics.mean(deltas),
                             "min_difference": min(deltas), "max_difference": max(deltas),
                             "all_histories_exact": all(p["both_histories_exact"] for p in selected)}
    return {"complete_matched_blocks": len(complete),
            "partial_blocks_excluded": len(grouped) - len(complete),
            "paired_step_results": pairs, "last_complete_step_by_replicate": last_step,
            "endpoint_differences": stats, "embedding_control_checks": embedding_checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=levels_arg, default=[32, 33, 34, 35])
    parser.add_argument("--vertices", type=positive, default=512)
    parser.add_argument("--steps", type=positive, default=5)
    parser.add_argument("--replicates", type=positive, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--clique-seconds", type=budget, default=5.0)
    parser.add_argument("--minutes", type=budget, default=10.0)
    parser.add_argument("--baseline", type=Path, default=growth.BASELINE)
    parser.add_argument("--output-dir", type=Path, default=Path("clique_level_comparison_results"))
    args = parser.parse_args(argv)
    started = perf_counter()
    baseline = growth.load_baseline(args.baseline)
    base_level = baseline["level"]
    if min(args.levels) < base_level:
        parser.error("all levels must be at least the baseline level")
    if args.vertices > (1 << base_level) - 2 - baseline["size"]:
        parser.error("candidate budget exceeds the baseline control's available dyads")
    arms = make_arms(args.levels, base_level)
    directory = args.output_dir.resolve() / ("run-" + uuid4().hex[:12])
    directory.mkdir(parents=True)
    config = {"arms": arms, "candidates_per_step": args.vertices, "steps": args.steps,
              "replicates": args.replicates, "seed": args.seed, "minutes": args.minutes,
              "clique_seconds": args.clique_seconds, "baseline_level": base_level,
              "baseline_size": baseline["size"], "sampler": "uniform fixed-XOR dyads; no translated supports",
              "budget_type": "equal candidate counts and clique time caps, not equal CPU time",
              "carry_between_arms": False, "carry_between_replicates": False}
    growth.save_json(directory / "configuration.json", config)
    growth.save_json(directory / "baseline_clique.json", baseline)
    summary = {"status": "running", "output_directory": str(directory), "configuration": config,
               "completed_phases": 0, "best_size": baseline["size"], "best_level": base_level}
    growth.save_json(directory / "best_clique.json", baseline)
    growth.save_json(directory / "summary.json", summary)
    records = []
    deadline = started + args.minutes * 60
    best_size = baseline["size"]
    stopped = False
    print(f"Every arm/replicate starts from the SAME {baseline['size']}-clique.", flush=True)
    print(f"Fixed-level control={base_level}; {len(arms)} arms; {args.replicates} replicates; "
          f"{args.steps} extension steps; {args.vertices} candidates/step", flush=True)
    print(f"Results: {directory}", flush=True)

    def checkpoint():
        summary.update(comparison_summary(records, arms, base_level))
        summary["elapsed_seconds"] = perf_counter() - started
        growth.save_json(directory / "summary.json", summary)

    try:
        for replicate in range(args.replicates):
            # These are independent chains. A result in one must never seed
            # another, even when its ambient level is smaller.
            anchors = {arm["id"]: growth.decode(baseline["vertices"]) for arm in arms}
            exact_history = {arm["id"]: True for arm in arms}
            for step in range(args.steps):
                if perf_counter() >= deadline:
                    stopped = True
                    break
                # Finish a whole matched block before checking the deadline.
                # Rotate execution order so the control is not always first.
                offset = (replicate + step) % len(arms)
                order = arms[offset:] + arms[:offset]
                block = {}
                for arm in order:
                    phase_start = perf_counter()
                    anchor = anchors[arm["id"]]
                    pool = uniform_pool(anchor, base_level, arm["support_level"], args.vertices,
                                        args.seed, replicate, step)
                    if len(pool) != args.vertices:
                        raise AssertionError("candidate budget mismatch")
                    found, search = growth.extend_with_pool(anchor, pool, args.clique_seconds)
                    source = {"experiment": "controlled level comparison", "arm": arm["id"],
                              "replicate": replicate, "step": step, "seed": args.seed}
                    certificate = growth.witness(found, arm["level"], source)
                    if not set(anchor) <= set(found):
                        raise AssertionError("extension lost anchor members")
                    anchors[arm["id"]] = found
                    exact_history[arm["id"]] &= search["maximum_extension_proven_in_pool"]
                    record = {**source, **search, "level": arm["level"],
                              "support_level": arm["support_level"], "role": arm["role"],
                              "size": len(found), "history_exact": exact_history[arm["id"]],
                              "gain_from_baseline": len(found) - baseline["size"],
                              "candidate_budget_total": (step + 1) * args.vertices,
                              "certificate": certificate, "phase_seconds": perf_counter() - phase_start}
                    with (directory / "trials.jsonl").open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
                    records.append(record)
                    block[arm["id"]] = record
                    summary["completed_phases"] += 1
                    if len(found) > best_size:
                        best_size = len(found)
                        summary.update(best_size=best_size, best_level=arm["level"])
                        growth.save_json(directory / "best_clique.json", certificate)
                    checkpoint()
                    print(f"rep={replicate} step={step} {arm['id']}: "
                          f"anchor={len(anchor)} added={len(found)-len(anchor)} clique={len(found)} "
                          f"candidates={len(pool)} verified; {record['phase_seconds']:.2f}s", flush=True)
                control = block[f"native-{base_level}"]
                for arm in arms:
                    if arm["role"] != "embedding-control":
                        continue
                    embedded = block[arm["id"]]
                    if embedded["history_exact"] and control["history_exact"]:
                        if embedded["certificate"]["vertices"] != control["certificate"]["vertices"]:
                            raise AssertionError("embedded control differs from exact baseline control")
                differences = [f"{arm['level']}:{block[arm['id']]['size']-control['size']:+d}"
                               for arm in arms if arm["role"] == "higher-level"]
                print(f"  versus level {base_level}: " + ", ".join(differences), flush=True)
            if stopped:
                break
        summary["status"] = "budget_reached" if stopped else "plan_completed"
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        print("Interrupted; completed phases saved; partial blocks excluded from comparisons.", flush=True)
    except Exception:
        summary["status"] = "failed"
        raise
    finally:
        checkpoint()
    print(f"{summary['status']}: {summary['complete_matched_blocks']} complete matched blocks; "
          f"best verified clique={summary['best_size']}", flush=True)
    for level, stats in summary["endpoint_differences"].items():
        print(f"level {level} endpoint differences versus control: {stats['differences']} "
              f"(mean {stats['mean_difference']:+.2f})", flush=True)
    print("Differences describe these matched sampling histories, not an ambient growth theorem.")
    return summary


if __name__ == "__main__":
    main()
