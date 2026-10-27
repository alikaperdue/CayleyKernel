# Exact structural scaling experiments

Run from the repository root (Python 3.10+):

```bash
python3 scripts/structure_scaling.py --output /tmp/structure-results
python3 -m unittest tests.test_structure_scaling
```

Edit `MAX_DEPTH = 2**18` near the top of the script to change the default
maximum depth. The default schedule includes depths 1, 2, 3, 4, then powers
of two through 262144. It includes the endpoint even when it is not a power
of two. These are selected depths, not every intervening depth.
`SAMPLES_PER_DEPTH = 100` controls trials per sampled design at each depth.
Override either setting with `--max-depth 524288 --samples 1000`, or supply
specific depths with `--levels 4 18 262144`.

Depth means index bit width: the default endpoint represents dimension
`2**262144`. For a maximum algebra dimension of `2**18` (262144 components),
set `MAX_DEPTH = 18` or use `--max-depth 18`.

The experiment uses `basis` with its actual signed, one-based API.
Internally, sparse dictionaries use mathematical zero-based indices. All
coefficients and residuals are exact integers. No dense vector or recursive
evaluation is needed at large levels. A small independent recursive oracle
calibrates all signed products through dimension 32 before experiments begin.

`--levels n` means dimension `2**n`, with basis indices below `2**n`. It does
not mean dimension `n` or exponentiation of a single algebra element.
Arbitrary-size integer operations are not constant time in their bit length.
Full tables still require `D**2` entries; `--table-level 8` streams a bounded
256-by-256 CSV table. Large levels use samples and sparse expressions.

The JSONL report separates uniform basis samples, uniform sparse samples,
exhaustive small basis triples, and an embedded sedenion witness. Sparse
samples draw independent indices and signs; repeated indices merge. Failure
counts are counts of nonzero residuals, without statistical extrapolation.
Each failing identity retains its first exact witness, with hexadecimal
indices to avoid Python's decimal conversion limit. Metadata records the
seed, arguments, script hash, sources, convention and calibration result.

The identities include conjugation as an anti-involution, scalar norm,
associativity, both alternative laws, flexibility, the middle Moufang law,
norm composition, and consistency with the classical doubling formula.
Alternativity requires testing sums: repeated single basis units can all
pass even when sums fail. Passing exhaustive basis triples does not establish
the non-multilinear identities on arbitrary elements.

[Kingdon algebras](https://arxiv.org/abs/2310.09972) are alternative algebras.
An alternativity counterexample prevents identifying that whole standard CD
algebra with a Kingdon algebra. Passing these necessary identities does not
establish the defining generator relations, a universal property or an
isomorphism. The script does not construct a general Kingdon algebra.

The [dimagma paper](https://doi.org/10.3390/math12070996) is from 2024.
It permits freely selected connecting maps and four actions. The script uses
one explicit specialization of equations (12)-(16): `X=B=A`, `h=t=0`,
`x tensor x' = x*x'`, with the other three actions zero. This gives the
ordinary direct product `(a,b)*(c,d)=(ac,bd)` with componentwise addition.
Its componentwise conjugation remains an anti-involution. Product differences
from CD are counted separately and are expected, not violations of the paper.
This is not the paper's different Section 4.1 specialization. A comparison
against another model requires specifying its maps, actions, involution and
embedding; neither a sign table nor this experiment establishes physical
validity or invalidity.

Optional dense backend comparison:

```bash
python3 scripts/structure_scaling.py --levels 4 --gpu-level 4 --gpu-samples 8 --output /tmp/structure-gpu
```

This calls `multiply` on signed int64 arrays and compares every coefficient
against sparse exact products. It requires the compiled kernel and CUDA/CuPy;
backend errors are reported, not silently skipped. GPU validation is capped
at level 20 and defaults to disabled. Large dense products can be expensive
even with sparse inputs, so start at a small level.
