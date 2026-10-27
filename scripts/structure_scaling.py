#!/usr/bin/env python3
"""Exact, sparse structural comparisons; Python 3.10+, no CUDA required.

Run: python3 scripts/structure_scaling.py
Indices in reports are zero based, hexadecimal; level n means dimension 2**n.
This tests CD identities and a specified dimagma specialization, not a general
Kingdon implementation or claims about physical applicability. Python big-int
operations take time/memory proportional to operand size, despite a fixed
number of bitwise operations in the basis formula. Full tables cost D**2.
"""

import argparse
import csv
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from CayleyKernel import basis

# EDIT THESE to change the default experiment size.
# Depth is the basis-index bit width: depth n represents dimension 2**n.
# Thus 2**18 below tests 262,144-bit indices, in dimension 2**262144.
# For a maximum algebra dimension of 2**18 instead, set MAX_DEPTH = 18.
MAX_DEPTH = 2**18
SAMPLES_PER_DEPTH = 100


def depth_schedule(max_depth):
    """Small controls, then geometrically spaced depths including the endpoint."""
    if max_depth < 1:
        raise ValueError("maximum depth must be positive")
    depths = set(range(1, min(4, max_depth) + 1))
    depth = 8
    while depth < max_depth:
        depths.add(depth)
        depth *= 2
    depths.add(max_depth)
    return sorted(depths)


SOURCES = {
    "kingdon": "https://arxiv.org/abs/2310.09972",
    "kingdon_definition": "https://jdhsmith.math.iastate.edu/math/ACLA.pdf",
    "dimagma_2024": "https://doi.org/10.3390/math12070996",
}


def e(i):
    return {i: 1}


def add(*xs):
    out = {}
    for x in xs:
        for i, c in x.items():
            out[i] = out.get(i, 0) + c
    return {i: c for i, c in out.items() if c}


def neg(x):
    return {i: -c for i, c in x.items()}


def conj(x):
    return {i: c if i == 0 else -c for i, c in x.items()}


def mul(x, y):
    out = {}
    for i, a in x.items():
        for j, b in y.items():
            label = basis(i + 1, j + 1)
            k = abs(label) - 1
            out[k] = out.get(k, 0) + a * b * (1 if label > 0 else -1)
    return {i: c for i, c in out.items() if c}


def assoc(x, y, z):
    return add(mul(mul(x, y), z), neg(mul(x, mul(y, z))))


def norm(x):
    return sum(c * c for c in x.values())


def parts(x, half):
    return ({i: c for i, c in x.items() if i < half},
            {i - half: c for i, c in x.items() if i >= half})


def doubled_product(x, y, half):
    """Classical CD doubling, evaluated through smaller basis indices."""
    a, b = parts(x, half)
    c, d = parts(y, half)
    first = add(mul(a, c), neg(mul(conj(d), b)))
    second = add(mul(d, a), mul(b, conj(c)))
    return add(first, {i + half: v for i, v in second.items()})


def dimagma_product(x, y, half):
    """Eqs. (13)-(16): X=B=A, h=t=0, xx'=xx', other actions zero.

    Formula (13) reduces to (ac,bd): the direct product dimagma A x A.
    This is a specified admissible choice, not Sec. 4.1's h=id,t=-id.
    Formula (12) gives ordinary componentwise addition for this choice.
    """
    a, b = parts(x, half)
    c, d = parts(y, half)
    return add(mul(a, c), {i + half: v for i, v in mul(b, d).items()})


def dimagma_conj(x, half):
    # Product involution, distinct from the usual CD conjugation on A x A.
    a, b = parts(x, half)
    return add(conj(a), {i + half: v for i, v in conj(b).items()})


@lru_cache(None)
def reference_sign(i, j, n):
    """Independent recursive oracle used ONLY for small calibration tables."""
    if n == 0:
        return 1
    h = 1 << (n - 1)
    a, b = i % h, j % h
    if i < h and j < h:
        return reference_sign(a, b, n - 1)
    if i < h:
        return reference_sign(b, a, n - 1)
    if j < h:
        return (-1 if b else 1) * reference_sign(a, b, n - 1)
    return -(-1 if b else 1) * reference_sign(b, a, n - 1)


def calibrate(n=5):
    for i, j in itertools.product(range(1 << n), repeat=2):
        expected = reference_sign(i, j, n) * ((i ^ j) + 1)
        for a, b in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            if basis(a * (i + 1), b * (j + 1)) != a * b * expected:
                raise RuntimeError("Basis convention disagrees with recursive CD oracle")
    return {"level": n, "signed_products": 4 * (1 << (2 * n)), "passed": True}


def encoded(x):
    # Hex avoids Python's decimal digit limit and preserves every input bit.
    return [[hex(i), c] for i, c in sorted(x.items())]


def residuals(x, y, z, n):
    xy = mul(x, y)
    h = 1 << (n - 1)
    dp = lambda a, b: dimagma_product(a, b, h)
    dc = lambda a: dimagma_conj(a, h)
    return {
        "anti_involution": add(conj(xy), neg(mul(conj(y), conj(x)))),
        "involution_squared": add(conj(conj(x)), neg(x)),
        "scalar_norm": add(mul(x, conj(x)), {0: -norm(x)}),
        "associativity": assoc(x, y, z),
        "left_alternativity": assoc(x, x, y),
        "right_alternativity": assoc(y, x, x),
        "flexibility": assoc(x, y, x),
        "moufang_middle": add(mul(xy, mul(z, x)),
                              neg(mul(mul(x, mul(y, z)), x))),
        "norm_composition": {0: norm(xy) - norm(x) * norm(y)}
                            if norm(xy) != norm(x) * norm(y) else {},
        "cd_doubling_consistency": add(xy, neg(doubled_product(x, y, h))),
        "dimagma_direct_product_difference": add(xy, neg(dp(x, y))),
        "dimagma_direct_product_anti_involution": add(dc(dp(x, y)), neg(dp(dc(y), dc(x)))),
    }


def measure(n, triples, design):
    start = perf_counter()
    counts, first = {}, {}
    total = 0
    for x, y, z in triples:
        total += 1
        for name, value in residuals(x, y, z, n).items():
            counts[name] = counts.get(name, 0) + bool(value)
            if value and name not in first:
                first[name] = {"x": encoded(x), "y": encoded(y), "z": encoded(z),
                               "residual": encoded(value)}
    return {"level": n, "dimension": "2**%d" % n, "design": design,
            "trials": total, "failures": counts, "first_witness": first,
            "seconds": perf_counter() - start}


def sparse(rng, n, terms):
    # Independent uniform basis draws with +/-1 coefficients; collisions merge.
    return add(*({rng.getrandbits(n): rng.choice((-1, 1))} for _ in range(terms)))


def low_witness():
    # A repeated basis unit alone can miss the failure of alternativity.
    for i, j, k in itertools.product(range(1, 16), repeat=3):
        x, y = add(e(i), e(j)), e(k)
        if assoc(x, x, y):
            return x, y, e(1)
    raise RuntimeError("No expected sedenion alternativity witness found")


def gpu_check(level, samples, rng):
    """Optional comparison; failures to load CUDA remain visible as errors."""
    import numpy as np
    from CayleyKernel import multiply
    d = 1 << level
    start = perf_counter()
    for _ in range(samples):
        x, y = sparse(rng, level, 3), sparse(rng, level, 3)
        a, b = np.zeros(d, dtype=np.int64), np.zeros(d, dtype=np.int64)
        for i, c in x.items():
            a[i] = c
        for i, c in y.items():
            b[i] = c
        expected = np.zeros(d, dtype=np.int64)
        for i, c in mul(x, y).items():
            expected[i] = c
        if not np.array_equal(multiply(a, b), expected):
            raise RuntimeError("GPU multiply disagrees with exact sparse products")
    return {"level": level, "samples": samples, "passed": True,
            "seconds": perf_counter() - start}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", nargs="+", type=int,
                   help="explicit depths; overrides the automatic depth schedule")
    p.add_argument("--max-depth", type=int, default=MAX_DEPTH,
                   help="maximum scheduled basis-index bit width (default: %(default)s)")
    p.add_argument("--samples", type=int, default=SAMPLES_PER_DEPTH)
    p.add_argument("--terms", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--exhaustive-level", type=int, default=3,
                   help="all basis triples at each level through this (max 5)")
    p.add_argument("--output", type=Path, default=Path("structure_scaling_results"))
    p.add_argument("--table-level", type=int,
                   help="stream full basis sign table at this level (0..10)")
    p.add_argument("--gpu-level", type=int, help="optional dense GPU validation (1..20)")
    p.add_argument("--gpu-samples", type=int, default=2)
    args = p.parse_args()
    if args.max_depth < 1:
        p.error("--max-depth must be positive")
    if args.levels is None:
        args.levels = depth_schedule(args.max_depth)
    if (min(args.levels) < 1 or args.samples < 1 or args.terms < 1
            or not 0 <= args.exhaustive_level <= 5 or args.gpu_samples < 1
            or args.table_level is not None and not 0 <= args.table_level <= 10
            or args.gpu_level is not None and not 1 <= args.gpu_level <= 20):
        p.error("positive levels/samples/terms required; check bounded table/exhaustive/GPU levels")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"sources": SOURCES, "python": platform.python_version(),
                "arguments": {k: str(v) if isinstance(v, Path) else v
                              for k, v in vars(args).items()},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "convention": "zero-based basis indices, exact hex witnesses; dimension=2**level",
                "interpretation": "Necessary Kingdon identity comparisons only. Dimagma: specified h=t=0 direct product, xx'=xx' and other actions zero. Differences are expected and not violations. No physical-model conclusions.",
                "calibration": calibrate()}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    witness = low_witness()
    # Flush each row so expensive runs retain completed levels.
    with (args.output / "results.jsonl").open("w") as f:
        for n in sorted(set(args.levels)):
            rng = random.Random(args.seed + n)
            designs = [("uniform_basis_sample", ((e(rng.getrandbits(n)), e(rng.getrandbits(n)),
                         e(rng.getrandbits(n))) for _ in range(args.samples)))]
            # Consume each generator before creating the next: one deterministic RNG stream.
            for design, triples in designs:
                row = measure(n, triples, design)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print("n=%d %s: %s" % (n, design, row["failures"]), flush=True)
            experiments = ["uniform_sparse_sample"]
            if n >= 4:
                experiments.append("embedded_sedenion_witness")
            if n <= args.exhaustive_level:
                experiments.append("exhaustive_basis_triples")
            for design in experiments:
                if design == "uniform_sparse_sample":
                    triples = ((sparse(rng, n, args.terms), sparse(rng, n, args.terms),
                                sparse(rng, n, args.terms)) for _ in range(args.samples))
                elif design == "embedded_sedenion_witness":
                    triples = [witness]
                else:
                    triples = ((e(i), e(j), e(k)) for i, j, k in
                               itertools.product(range(1 << n), repeat=3))
                row = measure(n, triples, design)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print("n=%d %s: %s" % (n, design, row["failures"]), flush=True)
    if args.table_level is not None:
        with (args.output / "basis_table.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["i_hex", "j_hex", "output_index_hex", "sign"])
            for i, j in itertools.product(range(1 << args.table_level), repeat=2):
                label = basis(i + 1, j + 1)
                w.writerow([hex(i), hex(j), hex(abs(label) - 1), 1 if label > 0 else -1])
    if args.gpu_level is not None:
        result = gpu_check(args.gpu_level, args.gpu_samples, random.Random(args.seed))
        (args.output / "gpu.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Saved", args.output.resolve())


if __name__ == "__main__":
    main()
