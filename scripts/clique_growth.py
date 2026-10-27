#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Search levels 32--35 for larger cliques and extensions of a saved witness.

Run: python scripts/clique_growth.py
Default: 512 candidates, 5 trials/level, one round, soft 10-minute budget.
Python >= 3.10; standard library and the local CayleyKernel package only.
"""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import random
import sys
from time import perf_counter
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.relation_graphs import (build_graphs, candidates, maximum_clique,
                                    positive, levels_arg, budget, verify_clique)

BASELINE = Path(__file__).resolve().parent / "results/relation_graphs_clique69.json"


def encode(vertices):
    return [[[hex(k), c] for k, c in x] for x in vertices]


def decode(rows):
    return [tuple((int(k, 16), c) for k, c in row) for row in rows]


@lru_cache(maxsize=131072)
def doubling_sign(a, b, level):
    """Independent iterative doubling oracle; no twist or basis.

    Evaluate (ac-conj(d)b, da+b conj(c)) from the highest doubling bit down.
    Cache size is bounded; only small levels are intended in this experiment.
    """
    sign = 1
    for bit in range(level - 1, -1, -1):
        half = 1 << bit
        upper_a, upper_b = bool(a & half), bool(b & half)
        a, b = a & (half - 1), b & (half - 1)
        if upper_a:
            if b:
                sign = -sign
            if upper_b:
                sign = -sign
        if upper_b:
            a, b = b, a
    return sign


def validate(vertices, level):
    """Require distinct canonical nonzero dyads within this algebra."""
    if len(vertices) != len(set(vertices)):
        raise ValueError("duplicate projective representatives")
    for x in vertices:
        if (len(x) != 2 or not 0 < x[0][0] < x[1][0] < 1 << level
                or x[0][1] != 1 or x[1][1] not in (-1, 1)):
            raise ValueError("invalid canonical dyad or support outside the algebra")


def certify(vertices, level):
    """Check all edges via basis and independently via doubling."""
    validate(vertices, level)
    verify_clique(vertices, list(range(len(vertices))), "orthogonality")
    products = 0
    for i, x in enumerate(vertices):
        for j, y in enumerate(vertices):
            if i == j:
                continue
            out = {}
            for a, ca in x:
                for b, cb in y:
                    k = a ^ b
                    out[k] = out.get(k, 0) + ca * cb * doubling_sign(a, b, level)
            if any(out.values()):
                raise AssertionError(f"independent doubling check failed for pair {i},{j}")
            products += 1
    return {"basis_verified": True, "independent_doubling_verified": True,
            "ordered_products_checked": products}


def witness(vertices, level, source):
    return {"level": level, "size": len(vertices), "vertices": encode(vertices),
            "source": source, "verification": certify(vertices, level)}


def load_baseline(path):
    record = json.loads(path.read_text())
    if "certificates" in record:
        vertices = decode(record["certificates"]["orthogonality"])
    else:
        vertices = decode(record["vertices"])
    if len(vertices) < 2:
        raise ValueError("baseline must contain at least two vertices")
    masks = {x[0][0] ^ x[1][0] for x in vertices if len(x) == 2}
    if len(masks) != 1:
        raise ValueError("baseline must have a common support XOR")
    return witness(vertices, record["level"], {"baseline_file": str(path.resolve())})


def extension_pool(anchor, level, count, rng):
    """Same-XOR neighbors: translated anchor supports plus full-width draws.

    Translating supports by XOR preserves their shared support mask, but
    does NOT assert annihilation. All relations are tested afterwards.
    At most half the pool is structured; the rest is sampled randomly.
    """
    mask = anchor[0][0][0] ^ anchor[0][1][0]
    excluded = set(anchor)
    capacity = (1 << level) - 2 - len(excluded)
    count = min(count, capacity)
    shifted = set()
    top = 1 << (level - 1)
    for x in anchor:
        a, b = sorted((x[0][0] ^ top, x[1][0] ^ top))
        if a:
            for sign in (-1, 1):
                shifted.add(((a, 1), (b, sign)))
    choices = sorted(shifted - excluded)
    rng.shuffle(choices)
    pool = set(choices[:count // 2])
    if (1 << level) <= 4 * (count + len(anchor) + 1):
        all_vertices, _ = candidates(level, (1 << level) - 2, rng, xor_mask=mask)
        choices = sorted(set(all_vertices) - excluded - pool)
        rng.shuffle(choices)
        pool.update(choices[:count - len(pool)])
    else:
        while len(pool) < count:
            a = rng.randrange(1, 1 << level)
            b = a ^ mask
            if not b:
                continue
            a, b = sorted((a, b))
            x = ((a, 1), (b, rng.choice((-1, 1))))
            if x not in excluded:
                pool.add(x)
    return sorted(pool)


def search_free(level, count, seed, trial, seconds):
    vertices, mask = candidates(level, count, random.Random(f"{seed}:{level}:{trial}"))
    graph = build_graphs(vertices, ["orthogonality"])["orthogonality"]
    ids = [i for i, neighbors in enumerate(graph) if neighbors]
    inverse = {v: i for i, v in enumerate(ids)}
    restricted = [{inverse[w] for w in graph[v]} for v in ids]
    result = maximum_clique(restricted, seconds)
    found = [vertices[ids[i]] for i in result["vertices"]]
    return found, {"phase": "free", "sampled_vertices": len(vertices),
                   "active_vertices": len(ids), "edges": sum(map(len, graph)) // 2,
                   "xor_mask_hex": hex(mask), "sample_maximum_proven": result["maximum_proven"],
                   "search_seconds": result["seconds"], "search_nodes": result["search_nodes"]}


def search_extension(anchor, level, count, seed, trial, seconds):
    pool = extension_pool(anchor, level, count,
                          random.Random(f"growth:{seed}:{level}:{trial}"))
    return extend_with_pool(anchor, pool, seconds)


def extend_with_pool(anchor, pool, seconds):
    """Find an extension preserving every anchor member in a supplied pool."""
    vertices = anchor + pool
    graph = build_graphs(vertices, ["orthogonality"])["orthogonality"]
    eligible = set(range(len(anchor), len(vertices)))
    for i in range(len(anchor)):
        eligible &= graph[i]
    ids = sorted(eligible)
    inverse = {v: i for i, v in enumerate(ids)}
    restricted = [{inverse[w] for w in graph[v] & eligible} for v in ids]
    result = maximum_clique(restricted, seconds)
    found = anchor + [vertices[ids[i]] for i in result["vertices"]]
    return found, {"phase": "extension", "anchor_size": len(anchor),
                   "candidate_pool": len(pool), "compatible_with_entire_anchor": len(ids),
                   "added_vertices": len(found) - len(anchor),
                   "xor_mask_hex": hex(anchor[0][0][0] ^ anchor[0][1][0]),
                   "maximum_extension_proven_in_pool": result["maximum_proven"],
                   "search_seconds": result["seconds"], "search_nodes": result["search_nodes"]}


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=levels_arg, default=[32, 33, 34, 35])
    parser.add_argument("--vertices", type=positive, default=512)
    parser.add_argument("--trials", type=positive, default=5)
    parser.add_argument("--rounds", type=positive, default=1,
                        help="repeat the plan with new trial IDs (default: 1)")
    parser.add_argument("--minutes", type=budget, default=10.0,
                        help="soft total budget including verification (default: 10)")
    parser.add_argument("--clique-seconds", type=budget, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output-dir", type=Path, default=Path("clique_growth_results"))
    args = parser.parse_args(argv)
    args.levels = sorted(set(args.levels))
    started = perf_counter()
    baseline = load_baseline(args.baseline)
    if min(args.levels) < baseline["level"]:
        parser.error("levels must be at least the baseline's level")
    directory = args.output_dir.resolve() / ("run-" + uuid4().hex[:12])
    directory.mkdir(parents=True)
    config = {"levels": args.levels, "vertices": args.vertices, "trials": args.trials,
              "rounds": args.rounds, "minutes": args.minutes, "clique_seconds": args.clique_seconds,
              "seed": args.seed, "baseline_level": baseline["level"],
              "baseline_file": str(args.baseline.resolve()),
              "baseline_size": baseline["size"], "algebra": "standard real Cayley-Dickson",
              "relation": "xy = yx = 0 on distinct projective representatives"}
    save_json(directory / "configuration.json", config)
    save_json(directory / "baseline_clique.json", baseline)
    best = baseline
    incumbents = {baseline["level"]: baseline}
    summary = {"configuration": config, "output_directory": str(directory),
               "status": "running", "completed_phases": 0, "best_size": best["size"],
               "best_level": best["level"], "per_level": {}}
    save_json(directory / "best_clique.json", best)
    save_json(directory / "summary.json", summary)
    print(f"Baseline: {best['size']}-clique at level {best['level']}, independently verified", flush=True)
    print(f"Levels={args.levels}; {args.vertices} candidates; {args.trials} trials; "
          f"{args.rounds} round(s); soft budget={args.minutes:g} min", flush=True)
    print(f"Results: {directory}", flush=True)
    deadline = started + 60 * args.minutes
    stopped = False
    try:
        for round_id in range(args.rounds):
            for level in args.levels:
                for local_trial in range(args.trials):
                    trial = round_id * args.trials + local_trial
                    for phase in ("free", "extension"):
                        if perf_counter() >= deadline:
                            stopped = True
                            break
                        phase_start = perf_counter()
                        seconds = min(args.clique_seconds, max(0.0, deadline - phase_start))
                        if phase == "free":
                            found, record = search_free(level, args.vertices, args.seed, trial, seconds)
                        else:
                            incumbent = max((v for n, v in incumbents.items() if n <= level),
                                            key=lambda v: v["size"])
                            found, record = search_extension(decode(incumbent["vertices"]), level,
                                                            args.vertices, args.seed, trial, seconds)
                        source = {"phase": phase, "level": level, "trial": trial, "seed": args.seed}
                        certificate = witness(found, level, source)
                        incumbent = max((v for n, v in incumbents.items() if n <= level),
                                        key=lambda v: v["size"])
                        if certificate["size"] > incumbent["size"]:
                            incumbents[level] = certificate
                        improved = certificate["size"] > best["size"]
                        if improved:
                            best = certificate
                            save_json(directory / "best_clique.json", best)
                        record.update(source)
                        record.update({"size": len(found), "certificate": certificate,
                                       "phase_seconds": perf_counter() - phase_start})
                        with (directory / "trials.jsonl").open("a") as stream:
                            stream.write(json.dumps(record) + "\n")
                        counts = summary["per_level"].setdefault(str(level),
                                  {"free_best": 0, "extension_best": 0, "completed_phases": 0})
                        counts[phase + "_best"] = max(counts[phase + "_best"], len(found))
                        counts["completed_phases"] += 1
                        summary["completed_phases"] += 1
                        summary.update(best_size=best["size"], best_level=best["level"],
                                       elapsed_seconds=perf_counter() - started)
                        save_json(directory / "summary.json", summary)
                        detail = (f" added={record['added_vertices']} eligible={record['compatible_with_entire_anchor']}"
                                  if phase == "extension" else f" edges={record['edges']}")
                        proven = record.get("sample_maximum_proven",
                                            record.get("maximum_extension_proven_in_pool"))
                        scope = "sample" if phase == "free" else "extension in pool"
                        print(f"level={level} trial={trial} {phase}: clique={len(found)}{detail} "
                              f"({scope} {'maximum' if proven else 'lower bound'}); "
                              f"verified by both methods; {record['phase_seconds']:.2f}s"
                              + (" NEW BEST" if improved else ""), flush=True)
                    if stopped:
                        break
                if stopped:
                    break
            if stopped:
                break
        summary["status"] = "budget_reached" if stopped else "plan_completed"
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
        print("Interrupted; completed phases and best certificate are saved.", flush=True)
    except Exception:
        summary["status"] = "failed"
        raise
    finally:
        summary["elapsed_seconds"] = perf_counter() - started
        summary.update(best_size=best["size"], best_level=best["level"])
        save_json(directory / "summary.json", summary)
    print(f"{summary['status']}: {summary['completed_phases']} phases; "
          f"best verified clique={best['size']} at level {best['level']}", flush=True)
    print("Finite samples and extensions only; no universal growth rule or ambient maximum proved.")
    return summary


if __name__ == "__main__":
    main()
