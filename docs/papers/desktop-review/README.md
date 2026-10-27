# Desktop paper verification collection

The historical collection classified 26 supplied PDFs: 16 were reviewed and
10 were archived outside the source tree. Nine reviews preceded a stricter
relevance filter; seven later reviews directly concern Cayley-Dickson
algebras, their multiplication or applications. Original PDFs are not
required for running the committed tests and are not distributed here.
The [historical manifest](manifest.json) records the original dispositions.

The 16 per-paper JSON snapshots linked below report **118 tests: 118 passed,
0 skipped, and no failures or errors** on 26 October 2026 (UTC). They record
CuPy as available and include successful device tests. This summarizes those
saved snapshots, not a new validation of the current checkout.

The separate [historical aggregate](summary.json) describes an earlier run
with **102 passed and 16 GPU skips**. It was not refreshed when the individual
snapshots were replaced. Its counts and archived PDF hashes are preserved as
historical provenance; it is not the aggregate for the snapshots below.
Passing counterexample regressions reproduce discrepancies rather than prove
the printed claims. These finite computations do not prove every assertion.

## Reviews and results

Each review explains conventions, tested ranges, limitations and current
commands. The table counts are derived from the linked saved JSON snapshots.
A saved source hash identifies historical bytes; it does not assert that the
current checkout matches them. Counts below are passed / skipped.

| Paper | Review | Tests | Review stage | Results |
| --- | --- | --- | --- | --- |
| [0311008v1](../0311008v1/README.md) | Differential characters on orbifolds and string connections I | 6 / 0 | Prior review | [JSON](../0311008v1/results/results.json) |
| [12050977](../12050977/README.md) | Representation of an ordered group of symbols by hypervectors | 6 / 0 | Prior review | [JSON](../12050977/results/results.json) |
| [1474295358](../1474295358/README.md) | An alternate Cayley-Dickson product | 5 / 0 | Prior review | [JSON](../1474295358/results/results.json) |
| [1602.02843v3](../1602.02843v3/README.md) | Periodicity of the Cayley-Dickson twists | 6 / 0 | Prior review | [JSON](../1602.02843v3/results/results.json) |
| [1703.00144v4](../1703.00144v4/README.md) | Theoretical properties for neural networks with weight matrices of low displacement rank | 7 / 0 | Prior review | [JSON](../1703.00144v4/results/results.json) |
| [1707.07318v4](../1707.07318v4/README.md) | The Cayley-Dickson doubling products | 8 / 0 | Prior review | [JSON](../1707.07318v4/results/results.json) |
| [2001.11797v4](../2001.11797v4/README.md) | A comparison of vector symbolic architectures | 7 / 0 | Prior review | [JSON](../2001.11797v4/results/results.json) |
| [2002.01113v1](../2002.01113v1/README.md) | Efficient Riemannian optimization on the Stiefel manifold via the Cayley transform | 6 / 0 | Prior review | [JSON](../2002.01113v1/results/results.json) |
| [20220019441](../20220019441/README.md) | Circuits, methods, and articles of manufacture for hyper-dimensional computing systems and related applications | 11 / 0 | Prior review | [JSON](../20220019441/results/results.json) |
| [2205.07986v2](../2205.07986v2/README.md) | The twisted group algebra structure of the Cayley-Dickson algebra | 8 / 0 | After relevance filter | [JSON](../2205.07986v2/results/results.json) |
| [2306.15889v4](../2306.15889v4/README.md) | Alternating roots of polynomials over Cayley-Dickson algebras | 9 / 0 | After relevance filter | [JSON](../2306.15889v4/results/results.json) |
| [2311.02269v1](../2311.02269v1/README.md) | All Hurwitz algebras from 3D geometric algebras | 8 / 0 | After relevance filter | [JSON](../2311.02269v1/results/results.json) |
| [2407.00449v5](../2407.00449v5/README.md) | Fully tensorial approach to hypercomplex-valued neural networks | 7 / 0 | After relevance filter | [JSON](../2407.00449v5/results/results.json) |
| [2503.12402v1](../2503.12402v1/README.md) | A calibration in R16 and Federer's product question | 7 / 0 | After relevance filter | [JSON](../2503.12402v1/results/results.json) |
| [2505.11747v3](../2505.11747v3/README.md) | Structure of the Cayley-Dickson algebras | 10 / 0 | After relevance filter | [JSON](../2505.11747v3/results/results.json) |
| [2609.11588v1](../2609.11588v1/README.md) | Quasilinear multiplication in the real Cayley-Dickson tower | 7 / 0 | After relevance filter | [JSON](../2609.11588v1/results/results.json) |

## Reproduce the checks

Run the committed collection tests from the repository root:

```bash
python3 -m unittest discover -s tests/papers/desktop_batch -t . -v
```

Or run one paper:

```bash
python3 -m unittest tests.papers.desktop_batch.test_2609_11588v1 -v
```

NumPy is required for the numerical companions. GPU methods additionally
require CuPy, the compiled kernel and a working CUDA device; they skip when
prerequisites are missing. Some experiments, especially the dense level-20
quasilinear checks, can take substantial time and memory.

These commands report current results to the terminal. They do not refresh
the historical JSON snapshots, their PDF/source hashes, or the aggregate.
The former custom report/collection runners have been retired. Tests do not
require PDFs, OCR helpers or machine-specific PDF locations. See the
[saved-evidence policy](../README.md).

Test sources are in
[tests/papers/desktop_batch](../../../tests/papers/desktop_batch/).
The shared adapter uses signed one-based `basis()` and zero-based `twist()`;
sparse products expand those basis products. Device checks call `multiply()`
when their prerequisites are available.

## Deeper checks and significant findings

- [Explicit twist formula](../2205.07986v2/README.md): all 1,048,576 ordered
  basis pairs in dimension 1024, plus sampled indices up to 4096 bits;
  discrepancies in the worked degree example and a displayed equation
  are recorded alongside the independent recursive comparison.
- [Alternating polynomial roots](../2306.15889v4/README.md): numeric root
  construction and exact friend-polynomial identities; a sedenion linear
  system of rank 12 with augmented rank 13 gives a concrete limit on the
  broad odd-degree root statement.
- [Clifford construction](../2311.02269v1/README.md): all eight signatures
  and both modified products, with explicit basis correspondence and a
  counterexample to the printed odd-part identity.
- [Tensorial networks](../2407.00449v5/README.md): dense layers and 1D/2D/3D
  convolution, plus a dual-number counterexample to a general
  nondegeneracy implication. This does not refute the separately tested
  Cayley-Dickson component reconstruction.
- [Calibration](../2503.12402v1/README.md): exact exterior forms, Clifford
  relations, self-duality and norm 294; sampled plane bounds are kept
  distinct from a proof of the global comass claim.
- [Algebra structure](../2505.11747v3/README.md): triad and zero-divisor
  support counts through dimension 128, subalgebra counts, all printed
  prime/mode products and split-algebra comparisons. Several prose/table
  count discrepancies and sign-counting conventions are documented.
- [Quasilinear multiplication](../2609.11588v1/README.md): an independent
  [NumPy CPU implementation](../../../scripts/quasilinear_cd.py) reproduces
  the recursion and arithmetic-count table. Actual dense calculations
  reach level 20 (1,048,576 coefficients). Four coordinates of each fully
  dense product have independent full signed-XOR-sum checks; a separate
  dense-by-four-term case checks every coordinate. CPU timings are
  observations, with no GPU speedup claim.

Sparse tests using 4096-bit indices address selected basis elements in a
level-4096 algebra; they do not allocate a vector of dimension 2^4096.
The level-20 quasilinear experiments do allocate dense vectors. Each
review identifies its own tested range and unimplemented work, including
proofs, training studies and optimized benchmark replications where
applicable.

## Coefficient-type coverage

All 16 companions now replay their existing CD device fixtures in float16,
complex64 and complex128, with independent recursive expectations and dtype
assertions. Holonomy, split signatures and quasilinear multiplication also
have the explicit supplements listed in their reviews. See
[coefficient types and numerical scope](../COEFFICIENT_TYPES.md). The saved
source hashes and pass counts above describe historical snapshots; they do
not establish execution of later coverage additions. The separate aggregate
retains the older CPU-only run described above.
