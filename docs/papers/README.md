# Paper verification and saved evidence

These companions connect published Cayley–Dickson results with reproducible
computational checks. Each review records the supplied paper version,
arithmetic conventions, tested cases, findings, and limits of coverage.
Finite checks provide evidence; they do not prove every statement in a paper.
A passing counterexample regression confirms that a discrepancy was reproduced,
not that the printed claim holds.

## Review index

The [16-paper collection](desktop-review/README.md) lists its individual
reviews, test modules, and saved results. It includes work on doubling
products, twist formulas, algebra structure, polynomial roots, tensorial
networks, and quasilinear multiplication, alongside earlier reviews of
adjacent topics. Consult that index for the collection's scope and history.

The other companions are listed below. Follow each review for findings,
version details, reproduction commands, and saved evidence.

| Paper or article | Review and computational focus |
| --- | --- |
| [math/0702075v2](math.0702075/README.md) | *Large Annihilators in Cayley-Dickson Algebras II*: exact annihilator and subspace calculations |
| [0704.0112v3](0704.0112/README.md) | *Placeholder Substructures III*: bit-rule tables, zero divisors, and recipe constructions |
| [0907.2047v3](0907.2047/README.md) | *The Basic Subalgebra Structure of the Cayley-Dickson Algebra of Dimension 32*: basis products and subalgebra tables |
| [Ünlü and Yalçın (2010)](quasilinear-actions-products-spheres/README.md) | *Quasilinear actions on products of spheres*: algebraic identities and finite frame examples |
| [1505.01670v1](1505.01670/README.md) | *On the Radius in Cayley-Dickson Algebras*: powers, radius, and numerical inequalities |
| [1912.01351v1](1912.01351/README.md) | *Function Theories in Cayley-Dickson algebras and Number Theory*: exact identities and counterexample regressions |
| [617-0702/2024](617-0702-2024/README.md) | *Graphical and Computational Analysis of Cayley-Dickson Algebra Elements over Z_p*: finite-field products and counts |
| [2607.23613v3](2607.23613/README.md) | *General polynomials and eigenvalues over Cayley-Dickson algebras*: polynomial roots, eigenvalues, and numerical search workflows |
| [2608.28176v1](2608.28176/README.md) | *Orthogonality graphs of real Cayley–Dickson algebras. Part I*: zero-divisor graphs and exact discrepancy checks |
| [Furey, Physics Letters B 882 (2026) 141078](physletb-2026-141078/README.md) | *Standard model symmetries and the nested embeddings of R ⊂ C ⊂ H ⊂ O*: multiplication operators, Clifford relations, and symmetry constraints |

See [coefficient types and numerical scope](COEFFICIENT_TYPES.md) for the
shared FP16 and complex replay conventions. Individual reviews distinguish
checks of printed statements from extensions beyond the paper's scope.

## Running current tests

Run commands from the repository root. A CPU-only source installation and
Python test dependencies can be prepared with:

```bash
python3 -m pip install --no-deps -e .
python3 -m pip install pytest numpy scipy sympy
```

SciPy is required to collect the current eigenvalue workflow tests. SymPy is
used by symbolic companions. Some individual suites have fewer dependencies;
see their review pages for focused commands.

Run one companion first, for example:

```bash
python3 -m pytest -v tests/papers/arxiv_2608_28176/test_sections.py
```

Run the full paper test collection with:

```bash
python3 -m pytest -v -m "not slow" tests/papers/
```

The `not slow` selection matches the default `make test` policy. Several
paper-specific exhaustive and large-array experiments are not marked `slow`
and can still take substantial time and memory. Consult the relevant review
before running the entire collection. For the 16-paper collection's standalone
`unittest` commands, see [its reproduction instructions](desktop-review/README.md#reproduce-the-checks).

CPU basis operations use the Python implementation or the optional Rust
backend. See [CPU backend setup](../../rust/README.md) for backend selection.
GPU checks additionally require CuPy, a compiled kernel, and a usable CUDA
device. Missing prerequisites produce skips in the paper suites; a skip does
not validate GPU multiplication. A numerical mismatch or missing required
entry point after prerequisites succeed is a failure, not an expected skip.

Tests run without the original PDFs and do not refresh the committed result
snapshots. Some companions retain a `run` module for generating reports; its
`--paper` option hashes a local PDF for provenance rather than supplying test
inputs. Use `--help` and the companion's instructions: `--output` accepts a
directory in some runners and a JSON filename in others. Choose a new output
location to preserve earlier evidence.

## Reading results and coverage

Each companion identifies its arithmetic: exact integers, rational numbers,
finite fields, symbolic expressions, or numerical arrays with tolerances.
Agreement in one setting does not establish a claim in another. CPU checks
using `twist()` or `basis()` and independent references are distinct from
device checks of `multiply()`.

`twist()` takes zero-based nonnegative indices. `basis()` takes nonzero,
signed one-based labels. A doubling level `n` describes dimension `2^n`;
sparse checks at large levels store selected terms and large integer indices,
whereas dense experiments allocate coefficient arrays. Large sparse levels
do not imply that a dense vector of that dimension was constructed.

Counterexample tests, controls, sampled cases, and exhaustive finite ranges
have different meanings. Consult the review and findings before interpreting
a pass count as support for a paper's general conclusions.

## Historical snapshots

Committed JSON, CSV, SVG, and retained text logs describe particular runs.
Their environments, outcomes, timestamps, source hashes, and PDF hashes apply
to those runs. Later source changes and added tests can change coverage and
counts. A recorded GPU pass does not establish GPU execution in a new
environment. A test that was not selected is distinct from a test that ran
and reported a skip.

Saved source hashes may refer to retired report-generation helpers or use
paths relative to the original test package. Preserve those hashes as
provenance; do not recompute them or relabel old results as a new run merely
to match the current checkout.

For the 16-paper collection, the individual per-paper snapshots and the older
[aggregate](desktop-review/summary.json) describe different runs. The
[collection index](desktop-review/README.md) explains their differing counts.
Use the linked per-paper evidence for the outcomes summarized there.

## What belongs in Git

Keep explanations, findings, compact reference JSON/CSV/SVG evidence, and
required test fixtures. Downloaded PDFs, OCR text, rendered pages, and new raw
console logs remain local. Existing tracked logs remain historical evidence;
adding an ignore rule does not remove them from Git. Save new console logs
outside the repository or in an ignored location, and use separate output
paths rather than overwriting historical results.
