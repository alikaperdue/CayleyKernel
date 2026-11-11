# Research witness fixtures

[`relation_graphs_clique69.json`](relation_graphs_clique69.json) is the
orthogonality graph record for level 32, zero-based trial 2, seed 2026,
and 512 sampled vertices. The certificate lists the 69 dyads as zero-based
hexadecimal indices with signed integer coefficients. Its SHA-256 is
`acbf5f7f56dd79ebbe313518a5690262104aaaa06f727fe24a83c5c7163f3295`.

Regenerate the search from the repository root into fresh output paths.

```bash
python -m scripts.relation_graphs --levels 32 --vertices 512 --trials 3 \
    --seed 2026 --relations orthogonality --clique-seconds 5 \
    --output /tmp/cayley-clique-search.json --checkpoint /tmp/cayley-clique-search.jsonl
```

The fixture is the record for trial 2 under `runs` in the JSON report.
Compare its supports and graph facts when replaying the search. The reported
timings vary by machine. The graph regression verifies all 4,692 ordered
products of distinct clique members using the iterative doubling oracle.
Clique growth and level comparison share this file as their default anchor.
See the [relation-graph experiment](../../../scripts/RELATION_GRAPHS.md).

The root-level `relation_graphs_witnesses.jsonl` is a generated checkpoint.
Paper evidence has its own canonical location under
[CayleyValidation](https://github.com/alikaperdue/CayleyValidation/blob/main/docs/papers/README.md). Eigen workflow regressions
generate small synthetic archives in temporary directories.
