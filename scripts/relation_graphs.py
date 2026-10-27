#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exact sparse relation-graph experiments in the standard real CD tower.

No dense vectors, GPU, recursion, or third-party graph packages are needed.
Level n denotes dimension 2**n, not dimension n. Python >= 3.10 required.
See scripts/RELATION_GRAPHS.md for definitions and interpretation.
"""

import argparse
from collections import deque
import json
from pathlib import Path
import random
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from CayleyKernel import basis, twist


RELATIONS = ("commute", "orthogonality", "zero-either", "dot")
DEFAULT_LEVELS = [4, 5] + [1 << k for k in range(3, 22)]
DEFAULT_VERTICES = 192
DEFAULT_TRIALS = 3


def sparse_product(x, y):
    """Canonical (zero-based index, integer coefficient) tuples -> exact xy."""
    out = {}
    for a, ca in x:
        for b, cb in y:
            encoded = basis(a + 1, b + 1)
            k = abs(encoded) - 1
            out[k] = out.get(k, 0) + ca * cb * (1 if encoded > 0 else -1)
    return {k: v for k, v in out.items() if v}


def related(x, y, relation, xy=None, yx=None):
    if relation == "dot":
        ys = dict(y)
        return sum(c * ys.get(k, 0) for k, c in x) == 0
    if xy is None:
        xy = sparse_product(x, y)
    if yx is None:
        yx = sparse_product(y, x)
    if relation == "commute":
        return xy == yx
    if relation == "orthogonality":
        return not xy and not yx
    if relation == "zero-either":
        return not xy or not yx
    raise ValueError("unknown relation: " + relation)


def candidates(level, count, rng, family="dyads", xor_mask=None):
    """Distinct projective representatives; exclude scalar support index 0.

    Dyads share a random XOR mask so that four-term cancellation is possible.
    Both relative signs are included; a global sign is canonicalized away.
    """
    dim = 1 << level
    if family == "basis":
        count = min(count, dim - 1)
        indices = set()
        if dim <= 2 * count + 1:
            indices = set(rng.sample(range(1, dim), count))
        else:
            while len(indices) < count:
                indices.add(rng.randrange(1, dim))
        return [((a, 1),) for a in sorted(indices)], None
    mask = xor_mask if xor_mask is not None else rng.randrange(1, dim)
    if not 0 < mask < dim:
        raise ValueError("XOR mask must satisfy 0 < mask < 2**level")
    count = min(count, dim - 2)
    vertices = set()
    # Small levels can be enumerated without rejection near saturation.
    if dim <= 2 * count + 2:
        pool = [((a, 1), (a ^ mask, s)) for a in range(1, dim)
                if a < (a ^ mask) for s in (-1, 1)]
        rng.shuffle(pool)
        vertices.update(pool[:count])
    else:
        while len(vertices) < count:
            a = rng.randrange(1, dim)
            b = a ^ mask
            if not b:
                continue
            a, b = sorted((a, b))
            vertices.add(((a, 1), (b, rng.choice((-1, 1)))))
    return sorted(vertices), mask


def build_graphs(vertices, relations):
    graphs = {r: [set() for _ in vertices] for r in relations}
    # A fixed-XOR dyad sample has exactly two output indices per product.
    # Classify cancellation with four signs, avoiding large output indices
    # and dictionary hashing on every edge. Witness checks below still use
    # full sparse_product through basis, in both multiplication orders.
    fixed_dyads = bool(vertices) and all(
        len(x) == 2 and 0 < x[0][0] < x[1][0] and x[0][1] == 1
        and x[1][1] in (-1, 1) for x in vertices)
    if fixed_dyads:
        mask = vertices[0][0][0] ^ vertices[0][1][0]
        fixed_dyads = (all(x[0][0] ^ x[1][0] == mask for x in vertices)
                       and len(set(vertices)) == len(vertices))
    algebraic = any(r != "dot" for r in relations)
    for i, x in enumerate(vertices):
        for j in range(i):
            y = vertices[j]
            if fixed_dyads:
                zero = False
                if algebraic:
                    (a, ca), (b, cb) = x
                    (c, cc), (d, cd) = y
                    zero = (ca * cc * twist(a, c) + cb * cd * twist(b, d) == 0
                            and ca * cd * twist(a, d) + cb * cc * twist(b, c) == 0)
                # Pure, mutually Euclidean-orthogonal samples have yx=-xy.
                for relation, graph in graphs.items():
                    if relation == "dot" or zero:
                        graph[i].add(j)
                        graph[j].add(i)
                continue
            xy = yx = None
            if algebraic:
                xy, yx = sparse_product(x, y), sparse_product(y, x)
            for r, graph in graphs.items():
                if related(x, y, r, xy, yx):
                    graph[i].add(j)
                    graph[j].add(i)
    return graphs


def maximum_clique(graph, seconds=2.0):
    """Iterative bitset search with greedy-color upper bounds.

    Any proper coloring bounds clique size by its number of colors. Search
    explores reverse coloring order so each prefix has a valid upper bound.
    Timeout returns a verified lower bound, never a claimed maximum.
    """
    start = perf_counter()
    adjacency = [sum(1 << v for v in neighbors) for neighbors in graph]
    best = []
    # Always find a greedy witness even when the exact-search budget is zero.
    remaining = set(range(len(graph)))
    while remaining:
        v = max(remaining, key=lambda v: (len(graph[v] & remaining), -v))
        best.append(v)
        remaining &= graph[v]
    def color_sort(mask):
        order, bounds = [], []
        color = 0
        while mask:
            color += 1
            available = mask
            while available:
                bit = available & -available
                v = bit.bit_length() - 1
                order.append(v)
                bounds.append(color)
                mask &= ~bit
                available &= ~bit & ~adjacency[v]
        return order, bounds

    def frame(chosen, mask):
        order, bounds = color_sort(mask)
        return [chosen, mask, order, bounds, len(order) - 1]

    stack = [frame([], (1 << len(graph)) - 1)]
    nodes = 0
    complete = True
    while stack:
        if perf_counter() - start >= seconds:
            complete = False
            break
        chosen, mask, order, bounds, position = stack[-1]
        nodes += 1
        if position < 0 or len(chosen) + bounds[position] <= len(best):
            stack.pop()
            continue
        v = order[position]
        stack[-1][4] -= 1
        stack[-1][1] &= ~(1 << v)
        extended = chosen + [v]
        if len(extended) > len(best):
            best = extended
        child = mask & adjacency[v]
        if child:
            stack.append(frame(extended, child))
    return {"vertices": sorted(best), "size": len(best), "maximum_proven": complete,
            "search_nodes": nodes, "seconds": perf_counter() - start}


def graph_distances(graph):
    """All-source BFS, exact for the finite induced graph only."""
    unseen = set(range(len(graph)))
    components = []
    while unseen:
        root = min(unseen)
        component = set()
        queue = [root]
        while queue:
            v = queue.pop()
            if v in component:
                continue
            component.add(v)
            queue.extend(graph[v] - component)
        unseen -= component
        diameter = 0
        endpoints = [root, root]
        for source in sorted(component):
            distance = {source: 0}
            pending = deque([source])
            while pending:
                v = pending.popleft()
                for w in graph[v]:
                    if w not in distance:
                        distance[w] = distance[v] + 1
                        pending.append(w)
                        if distance[w] > diameter:
                            diameter, endpoints = distance[w], [source, w]
        components.append({"size": len(component), "diameter": diameter,
                           "diameter_endpoints": endpoints})
    return {"connected": len(components) <= 1,
            "diameter": components[0]["diameter"] if len(components) == 1 else None,
            "components": components}


def verify_clique(vertices, witness, relation):
    """Recompute every witness edge, without trusting the adjacency cache."""
    if len(witness) != len(set(witness)):
        raise AssertionError("duplicate clique vertex")
    for offset, i in enumerate(witness):
        for j in witness[:offset]:
            if not related(vertices[i], vertices[j], relation):
                raise AssertionError("clique verification failed")


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def levels_arg(value):
    try:
        result = [int(x) for x in value.split(",")]
        if not result or min(result) < 2:
            raise ValueError
        return result
    except ValueError:
        raise argparse.ArgumentTypeError("use comma-separated doubling levels >= 2")


def budget(value):
    import math
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=levels_arg, default=DEFAULT_LEVELS,
                        help="doubling levels (default: 4,5,8,16,...,2097152)")
    parser.add_argument("--vertices", type=positive, default=DEFAULT_VERTICES,
                        help="samples per graph (default: 192)")
    parser.add_argument("--trials", type=positive, default=DEFAULT_TRIALS,
                        help="independent masks/samples per level (default: 3)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--family", choices=("basis", "dyads"), default="dyads")
    parser.add_argument("--xor-mask", type=lambda x: int(x, 0),
                        help="fixed nonzero support XOR, decimal or 0x hexadecimal")
    parser.add_argument("--relations", nargs="+", choices=RELATIONS, default=["orthogonality"],
                        help="default: orthogonality; other dyad relations are redundant controls")
    parser.add_argument("--clique-seconds", type=budget, default=5.0,
                        help="exact search budget per graph (default: 5); 0 uses greedy only")
    parser.add_argument("--clique-bound", type=positive,
                        help="user-supplied bound on clique SIZE for selected relations")
    parser.add_argument("--output", type=Path, help="JSON including full hex supports and clique witnesses")
    parser.add_argument("--checkpoint", type=Path, default=Path("relation_graphs_witnesses.jsonl"),
                        help="append certificates after every run (default: relation_graphs_witnesses.jsonl)")
    parser.add_argument("--no-checkpoint", dest="checkpoint", action="store_const", const=None,
                        help="disable incremental certificate file")
    args = parser.parse_args(argv)
    if args.xor_mask is not None and (args.family != "dyads" or
                                      not 0 < args.xor_mask < (1 << min(args.levels))):
        parser.error("--xor-mask requires dyads and 0 < mask < 2**min(levels)")
    report = {"algebra": "standard real Cayley-Dickson; NOT split",
              "index_base": 0, "seed": args.seed,
              "family": args.family, "requested_vertices": args.vertices,
              "trials": args.trials, "clique_seconds": args.clique_seconds,
              "scope": "finite induced graphs on sampled projective representatives",
              "clique_bound": args.clique_bound, "runs": []}
    print(report["algebra"], flush=True)
    print("Cliques bound ambient clique size from below; sampled diameters do not bound ambient diameter.")
    print(f"{args.trials} trials/level | up to {args.vertices} vertices | "
          f"clique budget={args.clique_seconds:g}s/graph", flush=True)
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        print(f"Appending per-run certificates to {args.checkpoint}", flush=True)
    for level in args.levels:
        for trial in range(args.trials):
            rng = random.Random(f"{args.seed}:{level}:{trial}")
            vertices, mask = candidates(level, args.vertices, rng, args.family, args.xor_mask)
            print(f"level={level} dimension=2^{level} trial={trial} vertices={len(vertices)}", flush=True)
            start = perf_counter()
            graphs = build_graphs(vertices, args.relations)
            run = {"level": level, "trial": trial,
                   "xor_mask_hex": hex(mask) if mask is not None else None,
                   "vertices": ([[[hex(k), c] for k, c in x] for x in vertices]
                                if args.output else None),
                   "graph_build_seconds": perf_counter() - start, "graphs": {}}
            for relation, graph in graphs.items():
                # An annihilation edge certifies that both endpoints are zero
                # divisors. Do not call unclassified isolated samples vertices
                # of a zero-divisor graph.
                ids = ([i for i, neighbors in enumerate(graph) if neighbors]
                       if relation in ("orthogonality", "zero-either")
                       else list(range(len(graph))))
                inverse = {v: i for i, v in enumerate(ids)}
                restricted = [{inverse[w] for w in graph[v]} for v in ids]
                clique = maximum_clique(restricted, args.clique_seconds)
                clique["vertices"] = [ids[i] for i in clique["vertices"]]
                verify_clique(vertices, clique["vertices"], relation)
                clique["verified"] = True
                exceeds = args.clique_bound is not None and clique["size"] > args.clique_bound
                distances = graph_distances(restricted)
                for component in distances["components"]:
                    component["diameter_endpoints"] = [ids[i] for i in component["diameter_endpoints"]]
                result = {"vertices": ids, "edges": sum(map(len, graph)) // 2,
                          "vertices_with_neighbors": sum(bool(g) for g in graph),
                          "clique": clique, "distances": distances,
                          "exceeds_user_clique_bound": exceeds}
                run["graphs"][relation] = result
                qualifier = "maximum" if clique["maximum_proven"] else "lower bound"
                print(f"  {relation:14} edges={result['edges']:5} clique={clique['size']} "
                      f"({qualifier}, verified) components={len(result['distances']['components'])}"
                      + (" EXCEEDS USER CLIQUE BOUND" if exceeds else ""), flush=True)
            # Full samples are retained only when explicitly requested. The
            # default checkpoint needs only the much smaller clique supports.
            run["certificates"] = {
                relation: [[[hex(k), c] for k, c in vertices[i]]
                           for i in result["clique"]["vertices"]]
                for relation, result in run["graphs"].items()}
            print(f"  graph build={run['graph_build_seconds']:.3f}s", flush=True)
            if args.checkpoint:
                certificate = {k: v for k, v in run.items() if k != "vertices"}
                certificate["configuration"] = {k: v for k, v in report.items() if k != "runs"}
                with args.checkpoint.open("a") as stream:
                    stream.write(json.dumps(certificate) + "\n")
            report["runs"].append(run)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Full hexadecimal supports and witnesses: {args.output}")
    best = {r: max(run["graphs"][r]["clique"]["size"] for run in report["runs"])
            for r in graphs}
    print("Largest verified cliques: " + ", ".join(f"{r}={size}" for r, size in best.items()))
    return report


if __name__ == "__main__":
    main()
