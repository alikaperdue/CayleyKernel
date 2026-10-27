# SPDX-License-Identifier: Apache-2.0
"""CPU-only associativity checks, including an independent doubling oracle."""

import csv
from functools import lru_cache
import itertools
import json
import random

import pytest

from CayleyKernel import twist
from scripts.associativity import (
    associativity_ratio, draw_triple, exhaustive_triples, independent,
    main, measure, seeded_rng, type_signature, wilson, fixed_weight,
    probe_positions, sweep_configurations, run_sweep,
    draw_parity, bernoulli_index, uniform_rate, prefix_independence_probability,
    exact_disjoint_orders, study_configurations, run_study,
)

pytestmark = pytest.mark.cpu


@lru_cache(None)
def doubling_sign(a, b, n):
    if n == 0:
        return 1
    half = 1 << (n - 1)
    upper_a, upper_b = a >= half, b >= half
    a, b = a % half, b % half
    sign = -1 if upper_a and b else 1
    if upper_a and upper_b:
        sign = -sign
    if upper_b:
        a, b = b, a
    return sign * doubling_sign(a, b, n - 1)


def iterative_doubling_sign(a, b, n):
    """Same doubling definition without Python's recursion-depth limit."""
    sign = 1
    for bit in range(n - 1, -1, -1):
        half = 1 << bit
        upper_a, upper_b = a >= half, b >= half
        a, b = a % half, b % half
        if upper_a and b:
            sign = -sign
        if upper_a and upper_b:
            sign = -sign
        if upper_b:
            a, b = b, a
    return sign


def test_twist_and_bracketings_against_doubling():
    n = 4
    for a, b in itertools.product(range(1 << n), repeat=2):
        assert twist(a, b) == doubling_sign(a, b, n)
    for a, b, c in exhaustive_triples(n):
        left = doubling_sign(a, b, n) * doubling_sign(a ^ b, c, n)
        right = doubling_sign(b, c, n) * doubling_sign(a, b ^ c, n)
        assert associativity_ratio(a, b, c) == left * right


@pytest.mark.parametrize("n,expected", [(0, 0), (1, 0), (2, 0), (3, 168), (4, 1848), (5, 15960)])
def test_exhaustive_baselines(n, expected):
    result = measure(n, exhaustive_triples(n), exact=True)
    row = next(r for r in result["summary"] if r["stratum"] == "all")
    assert row["failures"] == expected
    assert row["total"] == 1 << (3 * n)
    assert row["low95"] == row["rate"] == row["high95"]
    assert sum(r["count"] for r in result["signatures"]) == row["total"]
    assert sum(r["total"] for r in result["structure"]) == row["total"]
    assert sum(r["failures"] for r in result["structure"]) == expected
    for feature in {r["feature"] for r in result["overlap"]}:
        rows = [r for r in result["overlap"] if r["feature"] == feature]
        assert sum(r["total"] for r in rows) == row["total"]
        assert sum(r["failures"] for r in rows) == expected


def test_classification_on_independent_triads():
    seen = set()
    for triple in exhaustive_triples(4):
        if independent(*triple):
            seen.add(type_signature(*triple))
    assert seen <= {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)}
    assert (1, 1, 1) in seen


@pytest.mark.parametrize("n", [32, 128, 1024])
def test_sampling_regimes_and_arbitrary_width(n):
    rng = random.Random(7)
    for regime in ("uniform", "dense", "sparse", "clustered"):
        for _ in range(20):
            triple = draw_triple(rng, n, regime)
            assert all(0 <= x < (1 << n) for x in triple)
            if regime == "sparse":
                assert all(x.bit_count() == 3 for x in triple)
            if regime == "dense":
                assert all(x.bit_count() == n - 3 for x in triple)
            if regime == "clustered":
                union = triple[0] | triple[1] | triple[2]
                if union:
                    trailing = (union & -union).bit_length() - 1
                    assert union.bit_length() - trailing <= 8
            assert associativity_ratio(*triple) in (-1, 1)
            a, b, c = triple
            pairs = ((a, b), (a ^ b, c), (b, c), (a, b ^ c))
            reference_ratio = 1
            for x, y in pairs:
                sign = iterative_doubling_sign(x, y, n)
                assert twist(x, y) == sign
                reference_ratio *= sign
            assert associativity_ratio(a, b, c) == reference_ratio


def test_neighbors_joint_counts_and_centered_correlation():
    result = measure(3, exhaustive_triples(3), neighbors_rng=random.Random(4))
    for kind in ("flip_a", "flip_b", "flip_c", "flip_shared", "increment_a"):
        assert sum(r["total"] for r in result["neighbors"]
                   if r["perturbation"] == kind) == 512
    for row in result["neighbors"]:
        assert row["failures"] == row["n01"] + row["n10"]
        assert row["total"] == sum(row[k] for k in ("n00", "n01", "n10", "n11"))
        assert row["correlation"] is None or abs(row["correlation"]) <= 1 + 1e-12
    degenerate = measure(3, [(0, 0, 0)], neighbors_rng=random.Random(4))
    assert all(r["correlation"] is None for r in degenerate["neighbors"])


def test_intervals_and_reproducibility():
    low, high = wilson(0, 100)
    assert 0 <= low < .001 and 0.03 < high < .04
    low, high = wilson(50, 100)
    assert low < .5 < high
    assert wilson(0, 0) == (None, None)
    assert seeded_rng(7, 128, "sparse").getrandbits(1024) == seeded_rng(7, 128, "sparse").getrandbits(1024)


def test_cli_outputs_and_configuration(tmp_path):
    args = ["--exhaustive-max", "2", "--levels", "32", "128", "1024",
            "--samples", "8", "--no-sweeps", "--no-studies", "--output", str(tmp_path)]
    main(args)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["arguments"]["samples"] == 8
    with (tmp_path / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len([r for r in rows if r["stratum"] == "all"]) == 15
    for name in ("overlap", "signatures", "neighbors"):
        assert (tmp_path / (name + ".csv")).stat().st_size > 0
    first = (tmp_path / "summary.csv").read_bytes()
    main(args)
    assert (tmp_path / "summary.csv").read_bytes() == first


def test_probe_positions_and_targeted_neighbors():
    assert all(x in (0, 1) for x in draw_triple(random.Random(1), 1, "uniform"))
    positions = probe_positions(1024)
    assert positions[:8] == list(range(8))
    assert positions[-1] == 1023
    assert len(positions) == len(set(positions))
    assert probe_positions(1) == [0]
    result = measure(4, exhaustive_triples(4), neighbors_rng=random.Random(1), positions=[0, 3])
    assert {r["bit_position"] for r in result["neighbors"]} == {-1, 0, 3}
    assert all(r["total"] > 0 for r in result["neighbors"])


def test_fixed_weight_and_sweep_parameters():
    rng = random.Random(42)
    for n in (1, 8, 32):
        for k in range(n + 1):
            value = fixed_weight(rng, n, k)
            assert value.bit_count() == k and 0 <= value < (1 << n)
    configs = list(sweep_configurations([8, 8], [1, 3, 16], [0, .01, .49, .5, 1], [1, 4, 16]))
    assert len(configs) == 9  # four weight, three unique effective density, two width
    assert all(c["level"] == 8 for c in configs)
    for config in configs:
        row = run_sweep(config, 50, 42)
        assert row == run_sweep(config, 50, 42)
        assert row["total"] == 50
        assert row["low95"] <= row["rate"] <= row["high95"]
        if config["sweep"] == "density" and config["parameter"] in (0, 1):
            assert row["failures"] == 0


def test_sweeps_cli_outputs(tmp_path):
    main(["--exhaustive-max", "0", "--levels", "8", "--samples", "5",
          "--sweep-levels", "8", "--weights", "1", "--widths", "2",
          "--densities", ".5", "--sweep-samples", "10", "--no-studies", "--output", str(tmp_path)])
    tables = json.loads((tmp_path / "results.json").read_text())
    assert len(tables["sweeps"]) == 4
    assert all(r["total"] == 10 for r in tables["sweeps"])
    assert tables["structure"]
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["sampled_triples"] == 60
    assert metadata["schema_version"] == 3


def test_plot_and_replot(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from scripts import associativity
    main(["--exhaustive-max", "3", "--levels", "8", "--samples", "30",
          "--sweep-levels", "8", "--weights", "1", "3", "--widths", "2", "4",
          "--densities", ".25", ".75", "--sweep-samples", "30",
          "--study-levels", "8", "--study-samples", "30", "--study-densities", ".25", ".75",
          "--order-max-weight", "3",
          "--plot", "--output", str(tmp_path)])
    for name in ("associativity.png", "associativity.pdf", "structure.png", "controls.png", "controls.pdf"):
        assert (tmp_path / name).stat().st_size > 1000
    def no_sampling(*args, **kwargs):
        pytest.fail("replot must not run an experiment")
    monkeypatch.setattr(associativity, "measure", no_sampling)
    main(["--replot", "--output", str(tmp_path)])


def test_parity_control_and_study_reproducibility():
    rng = random.Random(2026)
    for pattern in range(8):
        for _ in range(20):
            triple = draw_parity(rng, 128, pattern)
            assert all(x.bit_length() <= 128 for x in triple)
            assert sum((x.bit_count() & 1) << i for i, x in enumerate(triple)) == pattern
    configs = list(study_configurations([8], [.25, .75]))
    assert len(configs) == 17  # eight parity, five adjacent weights, four density
    for config in configs:
        row = run_study(config, 20, 2026)
        assert row == run_study(config, 20, 2026)
        assert row["total"] == 20


@pytest.mark.parametrize("threshold", range(5))
def test_bernoulli_bit_planes_exact(threshold):
    # Four positions encode uniform integers 0,1,2,3. Exactly the first
    # threshold positions must be selected by the parallel comparison.
    class Planes:
        def __init__(self):
            self.planes = iter([0b1100, 0b1010])
        def getrandbits(self, n):
            assert n == 4
            return next(self.planes)
    assert bernoulli_index(Planes(), 4, threshold, precision=2) == (1 << threshold) - 1


def test_exact_order_counts_and_sparse_strata():
    rows = exact_disjoint_orders(3)
    pooled = next(r for r in rows if r["lowest_pattern"] == 0)
    assert (pooled["failures"], pooled["total"]) == (1056, 1680)
    assert (pooled["numerator"], pooled["denominator"]) == (22, 35)
    assert {r["lowest_pattern"]: r["failures"] for r in rows if r["lowest_pattern"]} == {1: 332, 2: 392, 4: 332}
    measured = measure(3, exhaustive_triples(3), exact=True)
    assert sum(r["total"] for r in measured["sparse_order"]) == 512
    disjoint = next(r for r in measured["summary"] if r["stratum"] == "disjoint_nonzero")
    assert disjoint["total"] == 6 and disjoint["rate"] == 1


def test_unused_bit_gaps_preserve_order_signs():
    rng = random.Random(42)
    for _ in range(100):
        positions = sorted(rng.sample(range(128), 9))
        triple = tuple(rng.getrandbits(9) for _ in range(3))
        embedded = tuple(sum(((x >> j) & 1) << pos for j, pos in enumerate(positions)) for x in triple)
        assert associativity_ratio(*triple) == associativity_ratio(*embedded)
        assert twist(triple[0], triple[1]) == twist(embedded[0], embedded[1])


def test_high_bit_rules_and_uniform_reference():
    measured = measure(4, exhaustive_triples(4), neighbors_rng=random.Random(8), positions=[3])
    supported = [r for r in measured["rule_checks"] if r["prefix_independent"]]
    assert len(supported) == 4
    assert all(r["failures"] == 0 and r["total"] > 0 for r in supported)
    assert prefix_independence_probability(2) == 0
    assert prefix_independence_probability(3) == 168 / 512
    assert uniform_rate(3) == 168 / 512
    assert uniform_rate(4) == 1848 / 4096
    assert uniform_rate(3, independent_only=True) == 1
    assert uniform_rate(16384) == .5


@pytest.mark.parametrize("args", [["--samples", "0"], ["--levels", "0"],
                                  ["--levels", "2"], ["--exhaustive-max", "7"],
                                  ["--densities", "nan"], ["--densities", "1.1"],
                                  ["--sweep-samples", "0"], ["--widths", "0"],
                                  ["--study-samples", "0"], ["--study-levels", "0"],
                                  ["--study-densities", "nan"], ["--order-max-weight", "6"]])
def test_invalid_cli(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
