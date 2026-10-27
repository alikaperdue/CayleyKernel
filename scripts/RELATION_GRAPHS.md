# Exact sparse relation graphs

Run from the repository root with Python 3.10 or newer:

```bash
python3 scripts/relation_graphs.py
python3 scripts/relation_graphs.py --levels 4,8,128 --vertices 96 --trials 1 --output /tmp/graphs.json
python3 scripts/relation_graphs.py --levels 4,5,8,16,128,34284,600000 --vertices 64 --trials 3 --output /tmp/huge-graphs.json
python3 scripts/relation_graphs.py --levels 4 --vertices 14 --xor-mask 9 --relations orthogonality zero-either --clique-seconds 10
python3 scripts/relation_graphs.py --levels 4,8 --family basis --vertices 64 --relations commute orthogonality zero-either dot
# Push breadth and index size further than the defaults:
python3 scripts/relation_graphs.py --levels 512,1048576,2097152,4194304 --vertices 384 --trials 5 --clique-seconds 15 --checkpoint /tmp/graphs-deeper.jsonl
```

The defaults now cover levels **4, 5, 8, 16, 32, 64, ...,
2,097,152**, with **192 vertices, three independent trials per level, and
a five-second exact clique budget per graph**. That includes and exceeds
the pasted sweep through level 1,048,576. The default relation is mutual
annihilation (`orthogonality`). All four relations remain available using
`--relations commute orthogonality zero-either dot`.

The expanded default sweep completed all **63** distinct level/trial runs.
Its largest sampled clique had size **45** at level 131,072, trial 2. At
the final level 2,097,152, the three sampled maximum clique sizes were
**16, 9, and 25**. Counts and timings are saved in the
[extended summary](results/relation_graphs_extended_summary.json).

Every completed run appends a self-contained record to
`relation_graphs_witnesses.jsonl` in the current directory. Each record has
the run configuration, XOR mask, graph measurements, and full hexadecimal
supports of the verified cliques. Repeated invocations append further
records; they do not resume or skip earlier runs. Use `--checkpoint PATH`
to choose another file or `--no-checkpoint` to disable it. An interruption
preserves completed runs. A run still constructing its graph has not yet
written its record.

`--output PATH` additionally writes a single JSON document containing every
sampled support after the whole sweep finishes. Without it, full sample
tables are discarded after each run and the returned report's `vertices`
field is null. This keeps the default sweep's memory use proportional to
the current sample plus the saved clique certificates. Certificates are
under `certificates[relation]`, each an array of sparse vectors written as
`[[zero_based_index_hex, integer_coefficient], ...]`. Their order corresponds
to the original vertex IDs in `graphs[relation].clique.vertices`.

The script uses `CayleyKernel.twist` and `basis` with exact Python integer
coefficients. It needs no GPU, dense arrays, graph package, or recursive
doubling. A basis vector has zero-based index `k`; the library call receives
`k + 1` and returns a signed one-based index. Sparse multiplication expands
each pair of supports and sums coefficients by output index.

`--levels n` means an algebra of dimension **2^n** with indices below 2^n.
It does not mean dimension n or a power of a particular basis element.
High-level sampling draws full-width random indices, rather than simply
embedding the same low-dimensional examples. Reproduce a run with the seed,
level, trial, family, requested vertex count, and optional XOR mask.

## Relations

All graphs are simple undirected graphs. Distinct nonzero elements are
represented up to nonzero real scalar multiples. The dyad family samples
`e_a + s e_b`, with `a < b`, `s` equal to +1 or -1, and `a XOR b` fixed
within each trial. Fixing the XOR makes four-term product cancellation
possible; independent random supports almost never have this property at
large levels. This is a deliberately structured sample, not uniform sampling
of the algebra or its zero divisors. Each trial chooses another mask unless
`--xor-mask` is specified. At most `2^n - 2` such dyads exist per mask.

For this sampler, **all distinct dyads are Euclidean-orthogonal**: their
supports are disjoint, or they have the same support and opposite relative
signs. Thus the `dot` graph is always complete, and its clique size equals
the sample count by construction. For pure elements the CD identity
`xy + yx = -2<x,y>` then gives `yx = -xy`. Consequently `commute`,
`orthogonality`, and `zero-either` have the same edges in these samples;
the commutativity graph additionally retains isolated nonzero samples.
Those matching edge counts are expected, rather than three independent
discoveries. This is why the larger default sweep computes only the
mutual-annihilation graph.

The pasted 18-clique at level 512 and its 2,112 edges are reproducible with
`--levels 512 --vertices 96 --trials 1 --seed 2026`. A regression test checks
both results. The smaller cliques at larger levels do not establish a
ceiling or decline: each level changes the mask and random supports, and
each trial observes only a small structured part of the algebra. Additional
trials explore different masks; larger vertex counts explore each mask more
deeply. A run with no edges means no mutual annihilations were found in its
sample, not that the algebra has no zero divisors.

A broader search found a **69-clique at level 32**, trial 2, seed 2026,
with 512 sampled vertices. Its [saved certificate](results/relation_graphs_clique69.json)
includes every support. All **4,692 ordered products** of distinct clique
vertices were independently verified by the iterative doubling oracle in
`tests/test_sparse_zd.py`, without `twist` or `basis`. The regression
suite repeats that independent verification. Reproduce the search with:

```bash
python3 scripts/relation_graphs.py --levels 32 --vertices 512 --trials 3
```

This proves existence of such a clique in the standard real algebra of
dimension 2^32. The sample's maximum was 69; the ambient maximum remains
undetermined. The standard embedding `x -> (x, 0)` preserves multiplication,
so the same certificate supplies a 69-clique at every higher level as well,
regardless of the smaller cliques found in some high-level random samples.

| CLI relation | Edge condition |
| --- | --- |
| `commute` | xy = yx |
| `orthogonality` | xy = yx = 0 (mutual annihilation) |
| `zero-either` | xy = 0 or yx = 0 (symmetrized zero-product graph) |
| `dot` | Euclidean coefficient dot product = 0 |

For annihilation graphs, only samples with a witnessed neighbor are retained;
every retained vertex is thereby certified to be a zero divisor. The output
`vertices` field for each graph lists its vertex IDs in the run's common
support table. Missing vertices may still have annihilators outside the
sample. Commutativity samples have no scalar components and so exclude the
real center. Euclidean orthogonality is reported separately from algebraic
mutual annihilation. No directed zero-divisor graph is computed.

Basis-only sampling is a useful control: distinct imaginary basis units
anticommute, have nonzero products, and are Euclidean orthogonal. Finding
zero divisors requires sums, which is why dyads are the default.

## Certificates and limits

Each graph gets a greedy clique followed by an iterative bitset exact search with
`--clique-seconds` as its time budget. Every reported clique is checked again
by recomputing its pairwise relations. `maximum_proven: true` means the
maximum of **this finite sampled graph** was proved by exhaustion and safe
pruning using proper graph colorings as safe upper bounds. Otherwise the
clique size is a lower bound. Set the budget to zero
for greedy search only. Rechecking the same product formula verifies the
certificate's edges; the small-dimensional tests additionally compare to
an independent recursive doubling definition.

`--clique-bound K` flags verified cliques larger than the user-supplied
clique-size bound. A flag only contradicts a conjecture if its algebra,
vertex restrictions, equivalence conventions, and edge definition agree.
No published conjecture or numerical bound is assumed by the script.
Failure to find a large clique proves no ambient upper bound or scaling law.
Fixed sample size also limits every observed clique size.

All-source BFS computes exact component diameters of the sampled graph.
Disconnected graphs have JSON `diameter: null`; an empty graph has no
components. Paths outside the sample can shorten distances and connect
components, so these measurements cannot disprove ambient diameter bounds.
A large clique does not violate a diameter bound.

The cited [2023 article](https://doi.org/10.1007/s10958-023-06285-5)
concerns **split** algebras; its abstract describes graph results for
split-complex numbers, split-quaternions, and split-octonions. This library's
standard real multiplication does not implement those split parameters.
These experiments do not claim to reproduce or refute that paper. Matching
its precise definitions and split multiplication would require additional
implementation and access to its full statements.

Graph construction uses quadratic many vertex pairs. For canonical dyads
with a common XOR mask, it classifies zero products from two cancellation
conditions using at most four twist evaluations; the second condition can
be skipped if the first fails. The other algebraic relations coincide for
this sample. General sparse samples use full products in both orders.
Clique certificates always use full `basis` multiplication in both
orders, even when their graphs used the optimized classifier. All-source
BFS is O(V(V+E)); exact
maximum-clique search is exponential in the worst case. Arbitrary-width
Python bit operations also cost time proportional to integer size; a
constant number of bit operations is not constant bit complexity. Huge
symbolic dimension avoids allocating 2^n coefficients but does not make
enumerating that many vertices feasible. JSON stores full hexadecimal
indices, which avoids Python's decimal string digit limit but may produce
large files. The time budget covers clique search; graph construction,
certificate verification, BFS, and file writes take additional time. The
per-run graph-build timing distinguishes large-index product cost from
clique-search cost.

Validation:

```bash
python3 -m pytest -q tests/test_relation_graphs.py
```
