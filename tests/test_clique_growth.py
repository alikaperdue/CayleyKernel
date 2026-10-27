"""CPU regression tests for free searches, anchored extensions, and output."""

import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import clique_growth as growth
from scripts import relation_graphs as rg
from tests.test_relation_graphs import reference

pytestmark = pytest.mark.cpu


def small_anchor():
    vertices, _ = rg.candidates(4, 14, random.Random(1), xor_mask=9)
    graph = rg.build_graphs(vertices, ["orthogonality"])["orthogonality"]
    clique = rg.maximum_clique(graph, 5)
    return [vertices[i] for i in clique["vertices"]]


def test_oracle_matches_dense_doubling_and_rejects_nonclique():
    for a in range(16):
        for b in range(16):
            x, y = [0] * 16, [0] * 16
            x[a], y[b] = 1, 1
            result = reference(x, y)
            assert result[a ^ b] == growth.doubling_sign(a, b, 4)
    anchor = small_anchor()
    assert growth.certify(anchor, 4)["ordered_products_checked"] == 2
    with pytest.raises(AssertionError):
        growth.certify([((1, 1), (2, 1)), ((1, 1), (2, -1))], 4)
    with pytest.raises(ValueError):
        growth.certify(anchor + anchor, 4)


@pytest.mark.parametrize("anchor_size", [1, 2])
def test_extension_maximum_matches_exhaustive_compatible_pool(anchor_size):
    # Exhaust the available fixed-mask dyads in dimension 16, independently
    # classify compatibility using full products, and compare the extension.
    anchor = small_anchor()[:anchor_size]
    result, record = growth.search_extension(anchor, 4, 100, 2026, 0, 5)
    all_vertices, _ = rg.candidates(4, 14, random.Random(7), xor_mask=9)
    compatible = [x for x in all_vertices if x not in anchor
                  and all(rg.related(x, y, "orthogonality") for y in anchor)]
    graph = rg.build_graphs(compatible, ["orthogonality"])["orthogonality"]
    expected = rg.maximum_clique(graph, 5)
    assert record["compatible_with_entire_anchor"] == len(compatible)
    assert record["maximum_extension_proven_in_pool"]
    assert len(result) == len(anchor) + expected["size"]
    assert set(anchor) <= set(result)
    growth.certify(result, 4)


def test_free_search_reproduces_known_69():
    vertices, record = growth.search_free(32, 512, 2026, 2, 5)
    assert len(vertices) == 69
    assert record["edges"] == 60790
    assert record["sample_maximum_proven"]
    assert growth.certify(vertices, 32)["ordered_products_checked"] == 4692


def test_main_checkpoints_budget_and_unique_output_directories(tmp_path):
    common = ["--levels", "32", "--vertices", "8", "--trials", "1",
              "--clique-seconds", "0", "--output-dir", str(tmp_path)]
    summary = growth.main(common)
    assert summary["status"] == "plan_completed"
    assert summary["completed_phases"] == 2
    assert summary["best_size"] >= 69
    directory = Path(summary["output_directory"])
    rows = [json.loads(line) for line in (directory / "trials.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in rows] == ["free", "extension"]
    for row in rows:
        certificate = row["certificate"]
        assert certificate["verification"]["independent_doubling_verified"]
        growth.certify(growth.decode(certificate["vertices"]), certificate["level"])
    best = json.loads((directory / "best_clique.json").read_text())
    assert best["size"] == summary["best_size"]
    stopped = growth.main(common + ["--minutes", "0"])
    assert stopped["status"] == "budget_reached" and stopped["completed_phases"] == 0
    assert stopped["output_directory"] != summary["output_directory"]
    assert json.loads((directory / "summary.json").read_text())["status"] == "plan_completed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-q", *sys.argv[1:]]))
