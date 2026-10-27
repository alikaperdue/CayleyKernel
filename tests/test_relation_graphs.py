"""CPU checks for the sparse graph experiment and its certificates."""

import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    "relation_graphs", Path(__file__).resolve().parents[1] / "scripts/relation_graphs.py")
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)
pytestmark = pytest.mark.cpu


def reference(x, y):
    """Independent dense doubling definition, only used in dimension 16."""
    if len(x) == 1:
        return [x[0] * y[0]]
    half = len(x) // 2
    a, b, c, d = x[:half], x[half:], y[:half], y[half:]
    conj = lambda z: z[:1] + [-v for v in z[1:]]
    ac, db = reference(a, c), reference(conj(d), b)
    da, bc = reference(d, a), reference(b, conj(c))
    return [u - v for u, v in zip(ac, db)] + [u + v for u, v in zip(da, bc)]


def test_sparse_products_and_relations_against_doubling():
    vertices, _ = rg.candidates(4, 14, random.Random(7), xor_mask=9)
    dense = []
    for x in vertices:
        row = [0] * 16
        for k, c in x:
            row[k] = c
        dense.append(row)
    graphs = rg.build_graphs(vertices, rg.RELATIONS)
    for i, x in enumerate(vertices):
        for j, y in enumerate(vertices):
            xy, yx = reference(dense[i], dense[j]), reference(dense[j], dense[i])
            assert rg.sparse_product(x, y) == {k: c for k, c in enumerate(xy) if c}
            expected = {"commute": xy == yx, "orthogonality": not any(xy) and not any(yx),
                        "zero-either": not any(xy) or not any(yx),
                        "dot": sum(a * b for a, b in zip(dense[i], dense[j])) == 0}
            for relation in rg.RELATIONS:
                assert (j in graphs[relation][i]) == (i != j and expected[relation])
    assert sum(map(len, graphs["orthogonality"])) > 0


def test_clique_search_against_exhaustive_subsets():
    rng = random.Random(42)
    for n in range(9):
        for _ in range(10):
            graph = [set() for _ in range(n)]
            for a, b in itertools.combinations(range(n), 2):
                if rng.random() < 0.5:
                    graph[a].add(b)
                    graph[b].add(a)
            result = rg.maximum_clique(graph, 10)
            expected = max(len(s) for size in range(n + 1)
                           for s in itertools.combinations(range(n), size)
                           if all(b in graph[a] for a, b in itertools.combinations(s, 2)))
            assert result["maximum_proven"]
            assert result["size"] == expected
            assert all(b in graph[a] for a, b in itertools.combinations(result["vertices"], 2))
    assert not rg.maximum_clique([{1}, {0}], 0)["maximum_proven"]


def test_distances_and_huge_indices(tmp_path):
    result = rg.graph_distances([{1}, {0, 2}, {1}, set()])
    assert result["diameter"] is None
    assert [c["diameter"] for c in result["components"]] == [2, 0]
    report = rg.main(["--levels", "600000", "--vertices", "6", "--clique-seconds", "0",
                      "--trials", "1", "--no-checkpoint", "--relations", *rg.RELATIONS,
                      "--output", str(tmp_path / "graphs.json")])
    run = report["runs"][0]
    vertices = [tuple((int(k, 16), c) for k, c in x) for x in run["vertices"]]
    assert max(k.bit_length() for x in vertices for k, c in x) > 599990
    for relation, result in run["graphs"].items():
        rg.verify_clique(vertices, result["clique"]["vertices"], relation)
    graphs = rg.build_graphs(vertices, rg.RELATIONS)
    for i, x in enumerate(vertices):
        for j, y in enumerate(vertices):
            if i != j:
                for relation in rg.RELATIONS:
                    assert (j in graphs[relation][i]) == rg.related(x, y, relation)


def test_basis_controls_and_no_zero_divisors():
    report = rg.main(["--levels", "3", "--family", "basis", "--vertices", "7",
                      "--trials", "1", "--no-checkpoint", "--relations", *rg.RELATIONS])
    graphs = report["runs"][0]["graphs"]
    assert graphs["commute"]["clique"]["size"] == 1
    assert graphs["dot"]["clique"]["size"] == 7
    assert graphs["orthogonality"]["clique"]["size"] == 0
    assert graphs["zero-either"]["clique"]["size"] == 0


def test_pasted_18_clique_is_reproducible():
    vertices, _ = rg.candidates(512, 96, random.Random("2026:512:0"))
    graph = rg.build_graphs(vertices, ["orthogonality"])["orthogonality"]
    assert sum(map(len, graph)) // 2 == 2112
    result = rg.maximum_clique(graph, 10)
    assert result["maximum_proven"]
    assert result["size"] == 18
    rg.verify_clique(vertices, result["vertices"], "orthogonality")


def test_checkpoint_contains_replayable_cliques_without_full_samples(tmp_path):
    path = tmp_path / "witnesses.jsonl"
    report = rg.main(["--levels", "4", "--vertices", "14", "--trials", "2",
                      "--checkpoint", str(path)])
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert report["runs"][0]["vertices"] is None
    for record in records:
        assert "vertices" not in record
        assert record["configuration"]["requested_vertices"] == 14
        for relation, support in record["certificates"].items():
            vertices = [tuple((int(k, 16), c) for k, c in x) for x in support]
            assert len(vertices) == record["graphs"][relation]["clique"]["size"]
            rg.verify_clique(vertices, list(range(len(vertices))), relation)


def test_coloring_bounds_and_basis_sample_count():
    # Complete multipartite graphs have large degrees but only one clique
    # vertex per independent part, so coloring must give a safe tight bound.
    graph = [{j for j in range(120) if j // 20 != i // 20} for i in range(120)]
    result = rg.maximum_clique(graph, 10)
    assert result["maximum_proven"] and result["size"] == 6
    vertices, _ = rg.candidates(4, 10, random.Random(3), "basis")
    assert len(vertices) == 10
    assert rg.DEFAULT_LEVELS[-1] == 2097152
    assert 1048576 in rg.DEFAULT_LEVELS
    assert rg.DEFAULT_VERTICES == 192 and rg.DEFAULT_TRIALS == 3


def test_general_sparse_graph_fallback():
    vertices = [((0, 1), (1, 2)), ((1, 1), (2, 3)),
                ((2, 1), (3, -1), (9, 2)), ((0, 2), (1, 4))]
    graphs = rg.build_graphs(vertices, rg.RELATIONS)
    for i, x in enumerate(vertices):
        for j, y in enumerate(vertices):
            for relation in rg.RELATIONS:
                assert (j in graphs[relation][i]) == (i != j and rg.related(x, y, relation))


def test_saved_69_clique_with_independent_doubling_oracle():
    from tests.test_sparse_zd import _doubling_sign

    path = Path(__file__).resolve().parents[1] / "scripts/results/relation_graphs_clique69.json"
    record = json.loads(path.read_text())
    vertices = [tuple((int(k, 16), c) for k, c in x)
                for x in record["certificates"]["orthogonality"]]
    assert len(vertices) == len(set(vertices)) == 69
    for i, x in enumerate(vertices):
        for j, y in enumerate(vertices):
            if i == j:
                continue
            out = {}
            for a, ca in x:
                for b, cb in y:
                    k = a ^ b
                    out[k] = out.get(k, 0) + ca * cb * _doubling_sign(a, b, record["level"])
            assert not any(out.values()), (i, j)


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-q", *sys.argv[1:]]))
