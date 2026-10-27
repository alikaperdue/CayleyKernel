"""Verify balanced budgets, independent histories, and embedding controls."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import clique_level_comparison as comparison
from scripts import clique_growth as growth
from tests.test_clique_growth import small_anchor

pytestmark = pytest.mark.cpu


def test_uniform_pools_are_reproducible_and_obey_support_budget():
    anchor = growth.decode(growth.load_baseline(growth.BASELINE)["vertices"])
    low = comparison.uniform_pool(anchor, 32, 32, 32, 2026, 0, 0)
    assert low == comparison.uniform_pool(anchor, 32, 32, 32, 2026, 0, 0)
    high = comparison.uniform_pool(anchor, 32, 35, 32, 2026, 0, 0)
    assert len(low) == len(high) == 32
    assert not set(low) & set(anchor)
    assert max(k.bit_length() for x in low for k, c in x) <= 32
    assert max(k.bit_length() for x in high for k, c in x) == 35
    mask = anchor[0][0][0] ^ anchor[0][1][0]
    assert all(x[0][0] ^ x[1][0] == mask for x in low + high)
    assert low != comparison.uniform_pool(anchor, 32, 32, 32, 2026, 1, 0)


def test_native_high_support_edges_against_full_products():
    from scripts.relation_graphs import build_graphs, related

    anchor = growth.decode(growth.load_baseline(growth.BASELINE)["vertices"])
    pool = comparison.uniform_pool(anchor, 32, 35, 8, 2026, 0, 0)
    vertices = anchor + pool
    graph = build_graphs(vertices, ["orthogonality"])["orthogonality"]
    for i in range(len(anchor), len(vertices)):
        for j in range(len(vertices)):
            if i != j:
                assert (j in graph[i]) == related(vertices[i], vertices[j], "orthogonality")


def test_equal_budgets_no_cross_arm_carry_and_exact_embedding(tmp_path, monkeypatch):
    baseline = growth.witness(small_anchor(), 4, {"test": True})
    path = tmp_path / "baseline.json"
    growth.save_json(path, baseline)
    captured = []
    original = growth.extend_with_pool

    def spy(anchor, pool, seconds):
        captured.append(growth.encode(anchor))
        return original(anchor, pool, seconds)

    monkeypatch.setattr(growth, "extend_with_pool", spy)
    args = ["--baseline", str(path), "--levels", "4,5", "--vertices", "2",
            "--steps", "2", "--replicates", "2", "--output-dir", str(tmp_path)]
    summary = comparison.main(args)
    directory = Path(summary["output_directory"])
    rows = [json.loads(line) for line in (directory / "trials.jsonl").read_text().splitlines()]
    assert summary["status"] == "plan_completed"
    assert len(rows) == len(captured) == 12
    assert summary["complete_matched_blocks"] == 4
    assert summary["partial_blocks_excluded"] == 0
    previous = {}
    for row, anchor in zip(rows, captured):
        key = row["replicate"], row["arm"]
        assert anchor == previous.get(key, baseline["vertices"])
        assert row["anchor_size"] == len(anchor)
        assert row["candidate_pool"] == 2
        assert row["candidate_budget_total"] == (row["step"] + 1) * 2
        previous[key] = row["certificate"]["vertices"]
        growth.certify(growth.decode(row["certificate"]["vertices"]), row["level"])
    for replicate in range(2):
        assert previous[replicate, "embedded-5"] == previous[replicate, "native-4"]
    assert all(check["passed"] for check in summary["embedding_control_checks"])
    stopped = comparison.main(args + ["--minutes", "0"])
    assert stopped["status"] == "budget_reached"
    assert stopped["completed_phases"] == stopped["complete_matched_blocks"] == 0
    assert stopped["output_directory"] != summary["output_directory"]


def test_partial_blocks_are_excluded_from_endpoint_differences():
    arms = comparison.make_arms([32, 33], 32)
    rows = [{"replicate": 0, "step": 0, "arm": arm["id"],
             "size": 75 if arm["id"] == "native-33" else 70, "history_exact": True}
            for arm in arms]
    rows.append({"replicate": 0, "step": 1, "arm": "native-32", "size": 1000,
                 "history_exact": True})
    summary = comparison.comparison_summary(rows, arms, 32)
    assert summary["complete_matched_blocks"] == 1
    assert summary["partial_blocks_excluded"] == 1
    assert summary["endpoint_differences"]["33"]["differences"] == [5]
    assert summary["last_complete_step_by_replicate"] == {0: 0}


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-q", *sys.argv[1:]]))
