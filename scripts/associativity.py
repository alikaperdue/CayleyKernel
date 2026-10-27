#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exact basis associativity experiments, without dense algebra vectors.

Level n means dimension 2**n; all indices are zero based. Run from any cwd:
  python3 scripts/associativity.py --plot --output /tmp/associativity

Output is deterministic. Use 'seed' option for comparison tests
  python3 scripts/associativity.py --plot --output /tmp/associativity --seed 2027

Only plotting requires matplotlib. CSV/JSON generation uses the standard
library. Python >= 3.10 is needed by CayleyKernel.twist's int.bit_count().
The measurements are evidence, not a proof of a limiting law.
"""

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from CayleyKernel import twist


REGIMES = ("uniform", "dense", "sparse", "clustered")
SOURCE = "https://arxiv.org/html/2505.11747v3#S3"
DEFAULT_LEVELS = [3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 127, 128, 129,
                  256, 512, 1023, 1024, 1025, 2048, 4096, 8192, 16384]


def uniform_rate(n, independent_only=False):
    """Analytical reference for this basis convention, not a fitted curve."""
    if n < 3:
        return None if independent_only else 0.
    inv = math.ldexp(1., -n)
    return (.5 + 3.5 * inv / (1 - inv) if independent_only
            else .5 - 14 * inv**2 + 24 * inv**3)


def prefix_independence_probability(j):
    return math.prod(1 - math.ldexp(1., r - j) for r in range(3)) if j >= 3 else 0.


def draw_parity(rng, n, pattern):
    """Uniform on each requested weight-parity class; a/b/c are bits 0/1/2."""
    values = []
    for i in range(3):
        x = rng.getrandbits(n)
        values.append(x ^ ((x.bit_count() & 1) ^ ((pattern >> i) & 1)))
    return tuple(values)


def bernoulli_index(rng, n, threshold, precision=16):
    """Independent bits with P(1)=threshold/2**precision, via bit planes.

    Each bit position receives its own uniform precision-bit integer.
    Comparing all these integers to threshold in parallel preserves bit
    independence. Actual quantized probabilities are recorded in output.
    """
    mask = (1 << n) - 1
    if threshold <= 0:
        return 0
    if threshold >= 1 << precision:
        return mask
    less, equal = 0, mask
    for bit in range(precision - 1, -1, -1):
        plane = rng.getrandbits(n)
        if (threshold >> bit) & 1:
            less |= equal & ~plane
            equal &= plane
        else:
            equal &= ~plane
    return less & mask


def fixed_weight(rng, n, weight):
    """Uniform weight-k subset; sample its complement when that is cheaper."""
    complement = weight > n // 2
    k = n - weight if complement else weight
    value = sum(1 << j for j in rng.sample(range(n), k))
    return value ^ ((1 << n) - 1) if complement else value


def probe_positions(n):
    """Resolve the first eight bits individually, then logarithmic positions."""
    positions = set(range(min(8, n)))
    bit = 8
    while bit < n:
        positions.add(bit)
        if bit + bit // 2 < n:
            positions.add(bit + bit // 2)
        bit *= 2
    positions.add(n - 1)
    return sorted(positions)


def associativity_ratio(a, b, c):
    """Return +1 for agreement of bracketings, -1 for disagreement."""
    return twist(a, b) * twist(a ^ b, c) * twist(b, c) * twist(a, b ^ c)


def independent(a, b, c):
    """Linear independence over F_2 for three binary index vectors."""
    return bool(a and b and c and a != b and a != c and b != c and a ^ b ^ c)


def type_signature(a, b, c):
    """Nonzero associators in the paper's Type 1, 2, 3 orderings.

    The labels A/B/C/X apply to the paper's pure independent triads.
    Other triples are retained separately rather than assigned these labels.
    """
    return tuple(int(associativity_ratio(*t) < 0)
                 for t in ((a, c, b), (a, b, c), (b, a, c)))


def draw_triple(rng, n, regime, weight=3, width=8):
    if n < 1 or weight < 0 or (regime in ("sparse", "dense") and weight > n) or width < 1:
        raise ValueError("require n >= 1, width >= 1, and a valid sparse/dense weight")
    if regime == "uniform":
        return tuple(rng.getrandbits(n) for _ in range(3))
    if regime in ("dense", "sparse"):
        mask = (1 << n) - 1
        values = tuple(fixed_weight(rng, n, weight) for _ in range(3))
        return tuple(mask ^ x for x in values) if regime == "dense" else values
    if regime == "clustered":
        width = min(width, n)
        start = rng.randrange(n - width + 1)
        return tuple(rng.getrandbits(width) << start for _ in range(3))
    raise ValueError("unknown regime: " + regime)


def wilson(failures, total):
    """Pointwise 95% Wilson interval for independent Bernoulli draws."""
    if total == 0:
        return None, None
    z = 1.959963984540054
    p = failures / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denom
    return (0. if failures == 0 else max(0., center - radius),
            1. if failures == total else min(1., center + radius))


def rate_row(failures, total, exact=False):
    rate = failures / total if total else None
    lo, hi = (rate, rate) if exact else wilson(failures, total)
    return dict(failures=failures, total=total, rate=rate, low95=lo, high95=hi)


def feature_values(a, b, c, n):
    """Marginal buckets: overlaps in tenths of level, positions exact.

    A bucket k means floor(10 * popcount / n) == k, with 10 reserved
    for full overlap. These are marginal summaries, not joint criteria.
    """
    result = {}
    for name, value in (("ab_overlap", a & b), ("ac_overlap", a & c),
                        ("bc_overlap", b & c), ("abc_overlap", a & b & c),
                        ("a_weight", a), ("b_weight", b), ("c_weight", c),
                        ("xor_weight", a ^ b ^ c)):
        result[name] = 10 * value.bit_count() // max(1, n)
    for name, value in (("tz_a", a), ("tz_b", b), ("tz_c", c)):
        result[name] = (value & -value).bit_length() - 1  # -1 for zero
    result["ab_overlap_count"] = (a & b).bit_count()
    result["pair_overlap_parity"] = overlap_parity(a, b, c)
    return result


def overlap_parity(a, b, c):
    """Three-bit category: ab, ac, bc overlap parities in bits 0, 1, 2."""
    return ((a & b).bit_count() & 1) | (((a & c).bit_count() & 1) << 1) | (((b & c).bit_count() & 1) << 2)


def lowest_pattern(a, b, c):
    """Occupancy (a,b,c in bits 0,1,2) at the first occupied bit; 0 if empty."""
    union = a | b | c
    bit = union & -union
    return int(bool(a & bit)) | (int(bool(b & bit)) << 1) | (int(bool(c & bit)) << 2)


def exhaustive_triples(n):
    return itertools.product(range(1 << n), repeat=3)


def measure(n, triples, exact=False, neighbors_rng=None, positions=None):
    """Stream triples; store aggregates only, with bounded memory per level.

    Each neighbor experiment uses one independent base per row; no walks.
    Correlations are centered Pearson correlations of failure indicators.
    """
    start = perf_counter()
    strata = defaultdict(Counter)
    features = defaultdict(Counter)
    signatures = Counter()
    neighbors = defaultdict(Counter)
    structure = defaultdict(Counter)
    sparse_order = defaultdict(Counter)
    rules = defaultdict(Counter)
    for a, b, c in triples:
        if min(a, b, c) < 0 or max(a, b, c).bit_length() > n:
            raise ValueError("index outside level")
        sig = type_signature(a, b, c)
        failure = sig[1]
        is_independent = independent(a, b, c)
        groups = ["all"]
        if a and b and c and len({a, b, c}) == 3:
            groups.append("distinct_nonzero")
        if is_independent:
            groups.append("independent")
        disjoint = bool(a and b and c and not ((a & b) | (a & c) | (b & c)))
        if disjoint:
            groups.append("disjoint_nonzero")
        for group in groups:
            strata[group]["total"] += 1
            strata[group]["failures"] += failure
        signatures[(is_independent, "".join(map(str, sig)))] += 1
        for name, bucket in feature_values(a, b, c, n).items():
            features[(name, bucket)]["total"] += 1
            features[(name, bucket)]["failures"] += failure
        key = (lowest_pattern(a, b, c), overlap_parity(a, b, c))
        structure[key]["total"] += 1
        structure[key]["failures"] += failure
        sparse_order[(disjoint, key[0])]["total"] += 1
        sparse_order[(disjoint, key[0])]["failures"] += failure
        if neighbors_rng is None:
            continue
        j = neighbors_rng.choice(positions) if positions is not None else neighbors_rng.randrange(n)
        bit = 1 << j
        mask = (1 << n) - 1
        prefix_mask = bit - 1
        prefix_independent = independent(a & prefix_mask, b & prefix_mask, c & prefix_mask)
        variants = [("flip_" + axis, tuple(x ^ bit if i == k else x
                     for i, x in enumerate((a, b, c))))
                    for k, axis in enumerate("abc")]
        variants += [("flip_shared", (a ^ bit, b ^ bit, c ^ bit)),
                     ("increment_a", ((a + 1) & mask, b, c))]
        for kind, triple in variants:
            # Increment has no selected bit position.
            bucket = -1 if kind == "increment_a" else j
            other = int(associativity_ratio(*triple) < 0)
            neighbors[(kind, bucket)][(failure, other)] += 1
            if kind != "increment_a":
                predicted_change = kind in ("flip_a", "flip_c")
                count = rules[(kind, j, prefix_independent)]
                count["total"] += 1
                count["violations"] += (failure != other) != predicted_change

    summary = [dict(stratum=group, **rate_row(count["failures"], count["total"], exact))
               for group, count in sorted(strata.items())]
    overlap = [dict(feature=name, bucket=bucket,
                    **rate_row(count["failures"], count["total"], exact))
               for (name, bucket), count in sorted(features.items())]
    types = []
    for is_independent, signature in itertools.product(
            sorted({key[0] for key in signatures}),
            ("".join(bits) for bits in itertools.product("01", repeat=3))):
        count = signatures[is_independent, signature]
        denom = sum(v for (ind, _), v in signatures.items() if ind == is_independent)
        label = {"000": "associative", "100": "A", "010": "B",
                 "001": "C", "111": "X"}.get(signature, "unclassified")
        types.append(dict(independent=is_independent, signature=signature,
                          label=label if is_independent else "degenerate",
                          count=count, total=denom, fraction=count / denom))
    nearby = []
    for (kind, bucket), counts in sorted(neighbors.items()):
        total = sum(counts.values())
        p = (counts[1, 0] + counts[1, 1]) / total
        q = (counts[0, 1] + counts[1, 1]) / total
        variance = p * (1 - p) * q * (1 - q)
        corr = ((counts[1, 1] / total - p * q) / math.sqrt(variance)
                if variance > 0 else None)
        nearby.append(dict(perturbation=kind, bit_position=bucket,
                           correlation=corr, n00=counts[0, 0], n01=counts[0, 1],
                           n10=counts[1, 0], n11=counts[1, 1],
                           **rate_row(counts[0, 1] + counts[1, 0], total)))
    return dict(summary=summary, overlap=overlap, signatures=types, neighbors=nearby,
                sparse_order=[dict(disjoint=disjoint, lowest_pattern=owner,
                                   **rate_row(count["failures"], count["total"], exact))
                              for (disjoint, owner), count in sorted(sparse_order.items())],
                rule_checks=[dict(perturbation=kind, bit_position=j, prefix_independent=condition,
                                  **rate_row(count["violations"], count["total"]))
                             for (kind, j, condition), count in sorted(rules.items())],
                structure=[dict(lowest_pattern=low, overlap_parity=parity,
                                **rate_row(count["failures"], count["total"], exact))
                           for (low, parity), count in sorted(structure.items())],
                seconds=perf_counter() - start)


def measure_rate(triples):
    """Rate-only sweeps spend their budget on independent base triples."""
    failures = total = 0
    for triple in triples:
        failures += associativity_ratio(*triple) < 0
        total += 1
    return rate_row(failures, total)


def sweep_configurations(levels, weights, densities, widths):
    """Return distinct, explicit effective parameters; no clipped duplicates."""
    for n in sorted(set(levels)):
        for regime in ("sparse", "dense"):
            for weight in sorted(set(weights)):
                if weight <= n:
                    yield dict(level=n, sweep="weight", regime=regime,
                               parameter=weight, effective_weight=weight, effective_width=0)
        density_weights = set()
        for density in sorted(set(densities)):
            weight = round(density * n)
            if weight in density_weights:
                continue
            density_weights.add(weight)
            yield dict(level=n, sweep="density", regime="fixed_density",
                       parameter=weight / n, effective_weight=weight, effective_width=0)
        for width in sorted(set(widths)):
            if width <= n:
                yield dict(level=n, sweep="width", regime="clustered",
                           parameter=width, effective_weight=0, effective_width=width)


def run_sweep(config, samples, seed):
    n = config["level"]
    regime = config["regime"]
    rng = seeded_rng(seed, "sweep", *config.values())
    if regime == "fixed_density":
        triples = (tuple(fixed_weight(rng, n, config["effective_weight"]) for _ in range(3))
                   for _ in range(samples))
    else:
        triples = (draw_triple(rng, n, regime, config["effective_weight"],
                               max(1, config["effective_width"])) for _ in range(samples))
    return dict(**config, **measure_rate(triples))


def study_configurations(levels, densities):
    """Controlled parity, adjacent-weight, and independent-bit comparisons."""
    for n in sorted(set(levels)):
        base = dict(level=n, study="", parameter=0., effective_weight=-1,
                    parity_pattern=-1, effective_density=-1.)
        for pattern in range(8):
            yield dict(base, study="parity", parameter=pattern, parity_pattern=pattern)
        for k in range(max(0, n // 2 - 2), min(n, n // 2 + 2) + 1):
            yield dict(base, study="half_weight", parameter=k - n // 2,
                       effective_weight=k, effective_density=k / n)
        for p in sorted(set(densities)):
            threshold = round(p * 65536)
            yield dict(base, study="bernoulli", parameter=p, effective_density=threshold / 65536)
            k = round(p * n)
            yield dict(base, study="fixed_density", parameter=p,
                       effective_weight=k, effective_density=k / n)


def run_study(config, samples, seed):
    n = config["level"]
    rng = seeded_rng(seed, "controlled-study", *sorted(config.items()))
    kind = config["study"]
    if kind == "parity":
        triples = (draw_parity(rng, n, config["parity_pattern"]) for _ in range(samples))
    elif kind == "bernoulli":
        threshold = round(config["effective_density"] * 65536)
        triples = (tuple(bernoulli_index(rng, n, threshold) for _ in range(3)) for _ in range(samples))
    else:
        triples = (tuple(fixed_weight(rng, n, config["effective_weight"]) for _ in range(3))
                   for _ in range(samples))
    return dict(**config, **measure_rate(triples))


def exact_disjoint_orders(weight):
    """Enumerate all (3k)!/(k!)**3 disjoint support orderings, without sampling.

    Compressing unused bit positions preserves twist, so this is the exact
    conditional distribution given disjoint weight-k supports, and its
    unconditioned fixed-k limit as ambient level tends to infinity.
    """
    bits = 3 * weight
    mask = (1 << bits) - 1
    counts = defaultdict(Counter)
    for aa in itertools.combinations(range(bits), weight):
        a = sum(1 << j for j in aa)
        remaining = [j for j in range(bits) if not (a >> j) & 1]
        for bb in itertools.combinations(remaining, weight):
            b = sum(1 << j for j in bb)
            c = mask ^ a ^ b
            failure = int(associativity_ratio(a, b, c) < 0)
            owner = lowest_pattern(a, b, c)
            for key in (0, owner):  # 0 denotes all owners combined.
                counts[key]["total"] += 1
                counts[key]["failures"] += failure
    rows = []
    assert counts[0]["total"] == math.factorial(bits) // math.factorial(weight)**3
    for owner, count in sorted(counts.items()):
        fraction = Fraction(count["failures"], count["total"])
        rows.append(dict(weight=weight, lowest_pattern=owner,
                         numerator=fraction.numerator, denominator=fraction.denominator,
                         **rate_row(count["failures"], count["total"], exact=True)))
    return rows


def seeded_rng(seed, *configuration):
    """Stable streams independent of run order and Python hash randomization."""
    digest = hashlib.sha256(repr((seed,) + configuration).encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(tables, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, grid = plt.subplots(3, 3, figsize=(19, 15), layout="constrained")
    axes = list(grid.flat)
    colors = dict(zip(REGIMES, ("#315f9b", "#d27a16", "#33824a", "#925ba7")))

    def curve(ax, subset, x, label, color=None, offset=0):
        subset = sorted(subset, key=lambda r: r[x])
        if not subset:
            return
        ax.errorbar([r[x] for r in subset], [r["rate"] - offset for r in subset],
                    yerr=[[max(0., r["rate"] - r["low95"]) for r in subset],
                          [max(0., r["high95"] - r["rate"]) for r in subset]],
                    marker="o", markersize=3, linewidth=1.3, capsize=2, label=label, color=color)

    ax = axes[0]
    rows = [r for r in tables["summary"] if r["stratum"] == "all" and r["level"] > 0]
    for regime in REGIMES:
        curve(ax, [r for r in rows if r["regime"] == regime], "level", regime, colors[regime])
    exact = [r for r in rows if r["mode"] == "exhaustive"]
    ax.scatter([r["level"] for r in exact], [r["rate"] for r in exact], marker="x", color="black", label="exact uniform", zorder=5)
    ax.set(xscale="log", xlabel="Level n (dimension 2**n)", ylabel="Failure probability", title="Regimes over doubling levels")
    reference_levels = sorted({r["level"] for r in rows if r["level"] >= 3})
    ax.plot(reference_levels, [uniform_rate(n) for n in reference_levels],
            color="black", linestyle=":", linewidth=1, label="uniform formula")
    sweeps = tables.get("sweeps", [])
    sweep_levels = sorted({r["level"] for r in sweeps})
    for n in sweep_levels:
        for regime, style in (("sparse", "-"), ("dense", "--")):
            subset = [r for r in sweeps if r["sweep"] == "weight" and r["level"] == n and r["regime"] == regime]
            before = len(axes[1].lines)
            curve(axes[1], subset, "parameter", "%s, n=%d" % (regime, n))
            if len(axes[1].lines) > before:
                axes[1].lines[before].set_linestyle(style)
        curve(axes[2], [r for r in sweeps if r["sweep"] == "density" and r["level"] == n],
              "parameter", "n=%d" % n)
        curve(axes[3], [r for r in sweeps if r["sweep"] == "width" and r["level"] == n],
              "parameter", "n=%d" % n)
    axes[1].set(xscale="log", xlabel="Number of ones (sparse) / zeros (dense)",
                ylabel="Failure probability", title="Fixed-weight sweep: solid sparse, dashed dense")
    axes[2].set(xlabel="Set-bit density k/n (exactly k ones per index)",
                ylabel="Failure probability", title="Fixed-density sweep")
    axes[3].set(xscale="log", xlabel="Shared cluster width (bits)",
                ylabel="Failure probability", title="Cluster width versus ambient level")
    # Small-width clusters have an exhaustive reference at their own level.
    axes[3].scatter([r["level"] for r in exact], [r["rate"] for r in exact],
                    marker="x", color="black", label="exact at width n")
    sampled = [r for r in tables["summary"] if r["mode"] == "sample"]
    level = max(r["level"] for r in sampled) if sampled else max(r["level"] for r in rows)
    subset = [r for r in tables["neighbors"] if r["level"] == level and r["regime"] == "uniform"]
    for kind in ("flip_a", "flip_b", "flip_c", "flip_shared"):
        part = sorted((r for r in subset if r["perturbation"] == kind), key=lambda r: r["bit_position"])
        curve(axes[4], part, "bit_position", kind)
        part = [r for r in part if r["correlation"] is not None]
        if part:
            axes[5].plot([r["bit_position"] for r in part], [r["correlation"] for r in part], "o-", markersize=3, label=kind)
    axes[4].set(xscale="symlog", xlabel="Bit position j (linear through 8, then log)",
                ylabel="P(R changes)", title="Uniform perturbations, n=%d" % level)
    axes[5].set(xscale="symlog", xlabel="Bit position j (linear through 8, then log)",
                ylabel="Pearson correlation", title="Centered correlation with perturbed triple", ylim=(-1.05, 1.05))
    axes[4].set_xscale("symlog", linthresh=8)
    axes[5].set_xscale("symlog", linthresh=8)
    for regime in REGIMES:
        part = [r for r in tables["overlap"] if r["level"] == level
                and r["regime"] == regime and r["feature"] == "pair_overlap_parity"]
        curve(axes[6], part, "bucket", regime, colors[regime])
    axes[6].set(xlabel="Parity category (ab, ac, bc in bits 0, 1, 2)",
                ylabel="Conditional failure probability", title="Overlap parity at n=%d (95%% intervals)" % level,
                xticks=range(8), xticklabels=[format(i, "03b") for i in range(8)])
    ax = axes[7]
    for label in ("associative", "A", "B", "C", "X"):
        part = sorted((r for r in tables["signatures"] if r["independent"]
                       and r["regime"] == "uniform" and r["label"] == label),
                      key=lambda r: r["level"])
        if part:
            ax.plot([r["level"] for r in part], [r["fraction"] for r in part], "o-", label=label)
    ax.set(xscale="log", xlabel="Level n", ylabel="Fraction of independent triples",
           title="Uniform three-ordering classification")
    for stratum in ("all", "independent"):
        part = [r for r in tables["summary"] if r["regime"] == "uniform" and r["stratum"] == stratum]
        curve(axes[8], part, "level", stratum, offset=.5)
    axes[8].set(xscale="log", xlabel="Level n", ylabel="Failure probability minus 1/2",
                title="Uniform convergence: all vs independent triples")
    axes[8].axhline(0, color="gray", linestyle="--")
    for i, ax in enumerate(axes):
        if i in (0, 1, 2, 3, 6):
            ax.axhline(.5, color="gray", linestyle="--", linewidth=.7)
        ax.grid(alpha=.18)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7, ncol=2 if i == 1 else 1)
    fig.suptitle("Basis associativity: exact counts, independent samples, and controlled sweeps\n"
                 "Error bars are pointwise 95% Wilson intervals; lines connect measured configurations", fontsize=15)
    fig.savefig(output / "associativity.png", dpi=160)
    fig.savefig(output / "associativity.pdf")
    plt.close(fig)
    plot_structure(tables.get("structure", []), level, output, plt)
    plot_studies(tables, output, plt)


def plot_studies(tables, output, plt):
    """Keep the overview readable; put mechanistic controls on a second page."""
    fig, grid = plt.subplots(2, 3, figsize=(19, 11), layout="constrained")
    axes = list(grid.flat)
    rows = tables.get("studies", [])
    levels = sorted({r["level"] for r in rows})

    def curve(ax, part, x, label, **kwargs):
        part = sorted(part, key=lambda r: r[x])
        if not part:
            return
        ax.errorbar([r[x] for r in part], [r["rate"] for r in part],
                    yerr=[[max(0., r["rate"] - r["low95"]) for r in part],
                          [max(0., r["high95"] - r["rate"]) for r in part]],
                    marker="o", markersize=4, capsize=2, label=label, **kwargs)

    for n in levels:
        curve(axes[0], [r for r in rows if r["level"] == n and r["study"] == "parity"],
              "parameter", "n=%d" % n)
        curve(axes[1], [r for r in rows if r["level"] == n and r["study"] == "half_weight"],
              "parameter", "n=%d" % n)
    axes[0].set(title="Control total-weight parity of each argument",
                xlabel="Weight-parity pattern cba (1=odd)", ylabel="Failure probability",
                xticks=range(8), xticklabels=[format(i, "03b") for i in range(8)])
    axes[1].set(title="Adjacent weights near half density", xlabel="k - floor(n/2) (exactly k ones per index)",
                ylabel="Failure probability", xticks=range(-2, 3))
    if levels:
        n = max(levels)
        for kind in ("bernoulli", "fixed_density"):
            curve(axes[2], [r for r in rows if r["level"] == n and r["study"] == kind],
                  "parameter", kind)
        axes[2].set(title="Same requested density, different constraints (n=%d)" % n)
    axes[2].set(xlabel="Requested density (effective densities recorded in CSV)", ylabel="Failure probability")
    orders = tables.get("exact_orders", [])
    exact = [r for r in orders if r["lowest_pattern"] == 0]
    if exact:
        axes[3].plot([r["weight"] for r in exact], [r["rate"] for r in exact],
                     "kD--", label="exact disjoint-support limit", markersize=5)
    sweeps = tables.get("sweeps", [])
    for n in sorted({r["level"] for r in sweeps}):
        curve(axes[3], [r for r in sweeps if r["level"] == n and r["sweep"] == "weight"
                        and r["regime"] == "sparse"], "parameter", "sampled n=%d" % n)
    axes[3].set(xscale="log", title="Fixed sparse weight versus exact order counts",
                xlabel="Set bits per index k", ylabel="Failure probability")
    part = [r for r in tables.get("sparse_order", []) if r["regime"] == "sparse" and r["disjoint"]]
    weight = part[0].get("weight", 3) if part else 3
    for owner, label, color in ((1, "a first", "#315f9b"), (2, "b first", "#d27a16"), (4, "c first", "#33824a")):
        curve(axes[4], [r for r in part if r["lowest_pattern"] == owner], "level", label, color=color)
        ref = next((r for r in orders if r["weight"] == weight and r["lowest_pattern"] == owner), None)
        if ref:
            axes[4].axhline(ref["rate"], color=color, linestyle="--", linewidth=1)
    axes[4].set(xscale="log", title="Lowest-bit owner, conditioned on disjoint supports (k=%d)" % weight,
                xlabel="Level n; dashed lines are exact order fractions", ylabel="Conditional failure probability")
    checks = [r for r in tables.get("rule_checks", []) if r["regime"] == "uniform"]
    if checks:
        n = max(r["level"] for r in checks)
        checks = [r for r in checks if r["level"] == n]
        for kind in ("flip_a", "flip_b"):
            grouped = defaultdict(Counter)
            for row in checks:
                if row["perturbation"] == kind:
                    grouped[row["bit_position"]]["total"] += row["total"]
                    grouped[row["bit_position"]]["failures"] += row["failures"]
            combined = [dict(bit_position=j, **rate_row(c["failures"], c["total"])) for j, c in grouped.items()]
            curve(axes[5], combined, "bit_position", kind + ": all prefixes")
            curve(axes[5], [r for r in checks if r["perturbation"] == kind and r["prefix_independent"]],
                  "bit_position", kind + ": independent prefixes", linestyle="--")
        positions = sorted({r["bit_position"] for r in checks})
        axes[5].plot(positions, [1 - prefix_independence_probability(j) for j in positions],
                     "k:", label="P(prefix dependent): bound")
        axes[5].set(title="High-bit rule exceptions (uniform n=%d)" % n)
    axes[5].set_xscale("symlog", linthresh=8)
    axes[5].set(xlabel="Bit position j", ylabel="Rule violation probability")
    for i, ax in enumerate(axes):
        if i < 3:
            ax.axhline(.5, color="gray", linestyle=":", linewidth=1)
        ax.grid(alpha=.18)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Mechanistic controls: parity, density, support order, and high-bit rules\n"
                 "Conditioning is labeled; error bars are pointwise 95% Wilson intervals", fontsize=15)
    fig.savefig(output / "controls.png", dpi=160)
    fig.savefig(output / "controls.pdf")
    plt.close(fig)


def plot_structure(rows, level, output, plt):
    """Joint conditioning reveals structure that marginal averages can hide."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for ax, regime in zip(axes.flat, REGIMES):
        part = {(r["lowest_pattern"], r["overlap_parity"]): r for r in rows
                if r["level"] == level and r["regime"] == regime}
        values = [[part.get((low, parity), {}).get("rate", float("nan"))
                   for parity in range(8)] for low in range(1, 8)]
        im = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for (low, parity), row in part.items():
            if low:
                ax.text(parity, low - 1, "%.2f\nn=%d" % (row["rate"], row["total"]),
                        ha="center", va="center", fontsize=7,
                        color="white" if row["rate"] < .55 else "black")
        ax.set(title="%s, level %d" % (regime, level),
               xlabel="Overlap parity (ab, ac, bc in bits 0, 1, 2)",
               ylabel="Occupancy at lowest occupied bit (a, b, c in bits 0, 1, 2)",
               xticks=range(8), xticklabels=[format(i, "03b") for i in range(8)],
               yticks=range(7), yticklabels=[format(i, "03b") for i in range(1, 8)])
    fig.colorbar(im, ax=list(axes.flat), label="Failure probability (blank cells have no observations)", shrink=.8)
    fig.suptitle("Joint binary structure: conditional rates and sample counts", fontsize=15)
    fig.savefig(output / "structure.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exhaustive-max", type=int, default=6, help="enumerate levels 0..N (default 6)")
    parser.add_argument("--allow-large-exhaustive", action="store_true", help="allow exponential enumeration above level 6")
    parser.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS,
                        help="sampled levels (default: 23 levels from 3 through 16384, including odd/even neighbors)")
    parser.add_argument("--samples", type=int, default=50000, help="independent base triples per regime and level (default 50000)")
    parser.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES))
    parser.add_argument("--weight", type=int, default=3, help="fixed ones for sparse / zeros for dense")
    parser.add_argument("--width", type=int, default=8, help="shared window width for clustered")
    parser.add_argument("--no-sweeps", action="store_true", help="skip weight/density/cluster-width sweeps")
    parser.add_argument("--sweep-samples", type=int, default=10000, help="draws per sweep configuration (default 10000)")
    parser.add_argument("--sweep-levels", type=int, nargs="+", default=[32, 128, 1024])
    parser.add_argument("--weights", type=int, nargs="+", default=list(range(1, 13)) + [16, 24, 32])
    parser.add_argument("--densities", type=float, nargs="+", default=[0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1])
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--no-studies", action="store_true", help="skip additional mechanistic controls and exact support orders")
    parser.add_argument("--study-levels", type=int, nargs="+", default=[32, 128, 1024])
    parser.add_argument("--study-samples", type=int, default=25000, help="independent triples per control configuration")
    parser.add_argument("--study-densities", type=float, nargs="+", default=[.01, .05, .1, .25, .5, .75, .9, .95, .99])
    parser.add_argument("--order-max-weight", type=int, default=5, help="exact disjoint support order counts at weights 1..K")
    parser.add_argument("--allow-large-orderings", action="store_true", help="allow factorial enumeration above weight 5")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("/tmp/cayley-associativity"))
    parser.add_argument("--plot", action="store_true", help="create overview, mechanistic controls, and structure charts (requires matplotlib)")
    parser.add_argument("--replot", action="store_true", help="plot saved results.json in --output without resampling")
    args = parser.parse_args(argv)
    if args.replot:
        try:
            plot_results(json.loads((args.output / "results.json").read_text()), args.output)
        except (OSError, ImportError) as exc:
            parser.exit(1, "%s\n" % exc)
        return
    if args.exhaustive_max < 0 or args.samples < 1 or min(args.levels) < 1 or args.width < 1:
        parser.error("levels/width/samples must be positive; exhaustive-max must be nonnegative")
    if args.weight < 0 or (set(args.regimes) & {"sparse", "dense"} and args.weight > min(args.levels)):
        parser.error("weight must be nonnegative and at most every sampled level for sparse/dense")
    if args.exhaustive_max > 6 and not args.allow_large_exhaustive:
        parser.error("enumeration costs 2**(3*n); use --allow-large-exhaustive above level 6")
    if (args.sweep_samples < 1 or min(args.sweep_levels) < 1 or min(args.weights) < 1
            or min(args.widths) < 1 or any(not 0 <= d <= 1 for d in args.densities)):
        parser.error("sweep samples/levels/weights/widths must be positive; densities must be in [0, 1]")
    if (args.study_samples < 1 or min(args.study_levels) < 1 or args.order_max_weight < 1
            or any(not 0 <= d <= 1 for d in args.study_densities)):
        parser.error("study samples/levels/order weight must be positive; study densities must be in [0, 1]")
    if args.order_max_weight > 5 and not args.allow_large_orderings:
        parser.error("support-order enumeration costs (3k)!/(k!)**3; use --allow-large-orderings above weight 5")
    tables = {name: [] for name in ("summary", "overlap", "signatures", "neighbors", "structure", "sparse_order", "rule_checks")}
    tables["sweeps"] = []
    tables["studies"] = []
    tables["exact_orders"] = []
    runs = []
    configurations = [(n, "exhaustive", "exhaustive") for n in range(args.exhaustive_max + 1)]
    configurations += [(n, "sample", regime) for n in sorted(set(args.levels))
                       for regime in dict.fromkeys(args.regimes)]
    sweeps = [] if args.no_sweeps else list(sweep_configurations(
        args.sweep_levels, args.weights, args.densities, args.widths))
    studies = [] if args.no_studies else list(study_configurations(args.study_levels, args.study_densities))
    sample_total = (len([c for c in configurations if c[1] == "sample"]) * args.samples
                    + len(sweeps) * args.sweep_samples + len(studies) * args.study_samples)
    exact_total = sum(1 << (3 * n) for n in range(args.exhaustive_max + 1))
    order_total = 0 if args.no_studies else sum(math.factorial(3*k) // math.factorial(k)**3
                                               for k in range(1, args.order_max_weight + 1))
    print("Planned: %s exhaustive triples + %s sampled triples + %s exact support orders; %d main runs + %d sweeps + %d controls" %
          (format(exact_total, ","), format(sample_total, ","), format(order_total, ","),
           len(configurations), len(sweeps), len(studies)), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    def checkpoint():
        for name, rows in tables.items():
            write_csv(args.output / (name + ".csv"), rows)

    for n, mode, regime in configurations:
        exact = mode == "exhaustive"
        rng = seeded_rng(args.seed, n, regime, args.weight, args.width, "base")
        triples = exhaustive_triples(n) if exact else (
            draw_triple(rng, n, regime, args.weight, args.width) for _ in range(args.samples))
        result = measure(n, triples, exact=exact, neighbors_rng=None if exact else
                         seeded_rng(args.seed, n, regime, args.weight, args.width, "neighbors"),
                         positions=None if exact else probe_positions(n))
        context = dict(level=n, mode=mode, regime=regime, weight=args.weight, width=args.width)
        for name in result.keys() & tables.keys():
            tables[name].extend(dict(**context, **row) for row in result[name])
        runs.append(dict(**context, seconds=result["seconds"]))
        all_row = next(r for r in result["summary"] if r["stratum"] == "all")
        print("n=%d %-10s %d/%d = %.8f (%.2fs)" %
              (n, regime, all_row["failures"], all_row["total"], all_row["rate"], result["seconds"]), flush=True)
        checkpoint()
    for config in sweeps:
        start = perf_counter()
        row = run_sweep(config, args.sweep_samples, args.seed)
        tables["sweeps"].append(row)
        runs.append(dict(**config, seconds=perf_counter() - start))
        print("sweep n=%d %-7s %-13s parameter=%g %d/%d = %.6f (%.2fs)" %
              (row["level"], row["sweep"], row["regime"], row["parameter"],
               row["failures"], row["total"], row["rate"], runs[-1]["seconds"]), flush=True)
        write_csv(args.output / "sweeps.csv", tables["sweeps"])
    for config in studies:
        start = perf_counter()
        row = run_study(config, args.study_samples, args.seed)
        tables["studies"].append(row)
        runs.append(dict(**config, seconds=perf_counter() - start))
        print("control n=%d %-13s parameter=%g %d/%d = %.6f (%.2fs)" %
              (row["level"], row["study"], row["parameter"], row["failures"], row["total"],
               row["rate"], runs[-1]["seconds"]), flush=True)
        write_csv(args.output / "studies.csv", tables["studies"])
    if not args.no_studies:
        for weight in range(1, args.order_max_weight + 1):
            start = perf_counter()
            rows = exact_disjoint_orders(weight)
            tables["exact_orders"].extend(rows)
            row = next(r for r in rows if r["lowest_pattern"] == 0)
            runs.append(dict(study="exact_orders", weight=weight, seconds=perf_counter() - start))
            print("exact disjoint weight=%d %d/%d = %d/%d (%.2fs)" %
                  (weight, row["failures"], row["total"], row["numerator"], row["denominator"], runs[-1]["seconds"]), flush=True)
            write_csv(args.output / "exact_orders.csv", tables["exact_orders"])
    (args.output / "results.json").write_text(json.dumps(tables, allow_nan=False) + "\n")
    metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    python=platform.python_version(), source=SOURCE, runs=runs,
                    schema_version=3, sampled_triples=sample_total, exhaustive_triples=exact_total,
                    exact_support_orders=order_total,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    convention="zero-based indices; n is bit width; dimension 2**n",
                    intervals="Pointwise 95% Wilson for independent draws; exhaustive intervals collapse to exact rate",
                    buckets="overlap: floor(10*popcount/n); tz and ab_overlap_count exact; pair_overlap_parity in bits ab/ac/bc",
                    neighbor_positions={n: probe_positions(n) for n in sorted(set(args.levels))},
                    neighbor_design="one uniformly selected probe position per independent base; CSV bit_position is exact, -1 for increment",
                    density_design="uniform fixed-weight indices, k=round(requested_density*n); parameter records actual k/n; duplicate k removed",
                    controls="parity: uniform within each parity class; half_weight: five neighboring k; density: fixed weight versus independent Bernoulli bits with recorded 16-bit probability",
                    high_bit_rule="If low-j-bit prefixes are independent: a/c flips reverse R; b/shared flips preserve R; rule_checks failures count violations, not associativity failures",
                    exact_orders_design="all disjoint support orders, k ones each: (3k)!/(k!)**3; owner=0 is pooled; other owners condition on lowest occupied bit",
                    neighbor_intervals="Intervals concern disagreement, not Pearson correlation; no multiple-comparison correction")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.plot:
        try:
            plot_results(tables, args.output)
        except ImportError as exc:
            parser.exit(1, "Data saved; plotting requires matplotlib: %s\n" % exc)
    print("Saved results to", args.output.resolve())


if __name__ == "__main__":
    main()
