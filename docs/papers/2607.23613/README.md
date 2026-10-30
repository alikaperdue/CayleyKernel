# General polynomials and eigenvalues over Cayley-Dickson algebras

Companion to Adam Chapman, Ilan Levin and Solomon Vishkautsan,
**General polynomials and eigenvalues over Cayley-Dickson algebras**, supplied
`2607.23613v3.pdf`, 15 pages, Communications in Mathematics 35 (2027), no. 2,
Paper 4, DOI `10.46298/cm.18874`. This review follows the supplied version,
including its October 9, 2026 acceptance date and 2027 publication label.
Page references below are printed/PDF page numbers. Source PDF SHA-256 and
all executed source hashes are in [results/results.json](results/results.json).
The reviewed PDF is not distributed in Git; obtain the cited version separately if you want to check its provenance hash.

## Outcome

The recorded run on October 28, 2026 has **22 tests: 20 passed, 2 skipped,
no failures or errors**. Both skips are GPU tests: the environment has CuPy
and a loaded kernel, but CUDA reports no device. These skips do not verify
`multiply()`. The CPU backend was Python. See [analysis](ANALYSIS.md) for
results, actual counterexample vectors, and interpretation.

No contradiction to the paper's stated claims was found in these checks.
The suite does find counterexamples to extending octonion-only arguments to
sedenions. Numerical root samples are evidence, not proofs of existence or
completeness for arbitrary inputs.

## What this paper is doing

A usual real quadratic has two roots at most. Here a root is a vector of
coefficients in an algebra, and an equation can have a whole sphere of roots.
For example, every unit pure imaginary octonion squares to -1.

The paper studies `f(x) = x*x + L(x) + r`, where `L` is any sum of bracketed
products containing exactly one `x`. Multiplication by fixed coefficients is
linear in the coordinates, even when multiplication is nonassociative.
Writing `x = t + v`, with `v` pure imaginary, makes all imaginary equations
linear once the real coordinate `t` is fixed. A determinant reduces the
remaining real equation to an ordinary one-variable polynomial. Singular
values of `t` must be treated separately: they can produce root spheres.

For a 2-by-2 octonion matrix, the paper turns an eigenvector of the form
`(1,x)` into just such a quadratic problem. This finds some right eigenvalues;
it does not claim that normalizing the first coordinate this way finds the
entire right spectrum. Triangular matrices receive a separate back-substitution
argument, including the important distinction between left and right eigenvalues.

## Reproduction

From the Git repository root, use a Python environment with NumPy and SymPy:

```bash
python -m pip install -r tests/papers/arxiv_2607_23613/requirements.txt
python -m tests.papers.arxiv_2607_23613.run
python -m pytest -v tests/papers/arxiv_2607_23613/test_sections.py
```

The saved run used a Python environment with NumPy and SymPy installed. Pytest is optional; the runner uses `unittest`. GPU tests execute
automatically if prerequisites are available, and then kernel failures are
failures rather than skips.

To create a **new** evidence file (the runner refuses to overwrite one):

```bash
python -m tests.papers.arxiv_2607_23613.run \
  --paper /path/to/2607.23613v3.pdf \
  --output /tmp/2607.23613-new-results.json
```

The GPU deep quadratic test defaults to levels 4, 8 and 12. To request level
20 (dimension 1,048,576) on a working device:

```bash
CK_PAPER_GPU_LEVEL=20 python -m tests.papers.arxiv_2607_23613.run
```

This is a correctness run, not a timing benchmark. No GPU speedup is inferred
from the CPU results. Symbolic determinants, root isolation and linear solves
have costs separate from the module's multiplication performance.

## Code and arithmetic

- [test_sections.py](../../../tests/papers/arxiv_2607_23613/test_sections.py) contains the numbered tests below.
- [method.py](../../../tests/papers/arxiv_2607_23613/method.py) implements the paper's coordinate matrix, exact determinant polynomials, regular candidates, and singular affine-sphere intersections.
- [run.py](../../../tests/papers/arxiv_2607_23613/run.py) saves statuses, environment, hashes, numerical residual data and counterexample witnesses.

Paths above are relative to this documentation directory. Sparse arithmetic
reuses the existing `tests.papers.desktop_batch.common` adapter and the
independent recursive doubling oracle from `arxiv_2608_28176.algebra`.
The module's current signed one-based API is `basis()`, exposed in the
adapter as `basis_product`. Mathematical indices in all reports are **zero
based**, so `i=e1`, `j=e2`, `ell=e4`. `twist()` uses zero-based indices.
Test 02 independently checks both API conventions against recursive doubling.
The deep tests use the module's basis products with full-width integer indices.

## Tests in paper order

| Test | Paper location | What is coded and what is checked |
| --- | --- | --- |
| 01 | Section 1, p. 2 | For dimensions 4, 8, 16, 32, every basis column of `ix-xi` has zero scalar part. Linearity therefore prevents `ix-xi+1=0`; illustrates why the monic hypothesis matters. |
| 02 | Section 2, pp. 3-4 | All 5,460 ordered basis products at levels 1-6 match the independent recursive convention. Check `twist`, positive labels, and a negative first label. |
| 03 | Section 2, p. 4 | 12 seeded sparse pairs per level 1-8: conjugate norm, quadratic identity, flexibility and the fourth-power parenthesizations. Integer comparisons are exact. |
| 04 | Section 2, p. 4 | Explicit noncommutativity/nonassociativity, plus 100 samples each showing octonion alternativity and failures in sedenions. |
| 05 | Section 3.1, p. 5 | Left-polynomial evaluation need not preserve multiplication. For `(a*x)(x-b)`, the evaluated right factor implication holds in 50 octonion samples and fails in sedenion samples. This is a degree-two test, not general factorization. |
| 06 | Sections 3.2-4, pp. 5-6 | Bracketed expressions give different functions; reproduce the section's deeply bracketed linear term and its coefficient matrix in dimension 16. |
| 07 | Section 4, equations E0-Em, p. 7 | For dimensions 2, 4, 8, 16, 32, compare all coordinate equations with original bracketed evaluation, exactly, using seeded coefficients and inputs. |
| 08 | Claim 4.1 / Lemma 4.2, pp. 7-8 | Consistent singular trace for `x²+2x+1`: determinant degree/leading coefficient, common-factor removal, reduced degree bound and unique root. A second fixture has nonzero Cramer minors, all divisible by the singular-trace factor. |
| 09 | Section 4, pp. 7-8 | `x²+i`: singular trace zero is inconsistent and must be excluded; recover two regular roots and check original residuals. Check the degree bound. |
| 10 | Lemma 4.3 / Theorems 4.4-4.5, pp. 8-9 | `x²+r0` has an empty/point/sphere singular slice for `r0=-1/0/1`, dimensions 2-64. `x²+2ix+r0` in H tests an offset affine line, including the two-point (zero-dimensional sphere) case. Verify representatives by substitution. Empty slice does not mean the whole polynomial is rootless. |
| 11 | Theorem 4.6, p. 10 | Two independently sampled general quadratics in each of C, H, O. Use bracketed linear terms, exact univariate elimination and numerical root representatives; verify existence in these six samples, degree/singular-trace bounds, and original residuals. No roots are planted. |
| 12 | Remark 4.7, pp. 10-11 | Symbolically verify that `x²+1` has no real root at level 0, and check `i` is a root at level 1. |
| 13 | Example 5.3, p. 12 | Reproduce the printed `y`, `x`, and right eigenvalue exactly. Also solve its transformed quadratic and substitute each computed candidate into the original matrix equation. |
| 14 | Theorem 5.1 / Remark 5.2, pp. 11-12 | Two seeded arbitrary octonion 2-by-2 matrices: construct the bracketed transformed polynomial and recover right eigenpairs. Exercise the `b=0`, vector `(0,1)` branch. Jou's cited theorem is not independently proved. |
| 15 | Propositions 6.1-6.2, pp. 12-14 | Upper and lower 3-by-3 octonion triangular matrices, distinct and repeated diagonal classes: construct left/right eigenvectors, solve Sylvester systems, test singularity for conjugate diagonal values, and substitute into every matrix row. Both repeated cases choose the earliest matching diagonal. |
| 16 | Section 6 correction, pp. 13-14 | `[[0,1],[0,i]]`: the second standard vector fails, while `(-i,1)` is a left eigenvector for `i`. |
| 17 | Higher-level extension of Section 4.1 | Levels 4, 8, 16, 32, 64, 128, 1024, 16 samples each: exact pure-imaginary and translated scalar quadratics; also planted roots for bracketed linear terms. Full-width randomly selected indices, at most a few nonzero coefficients. |
| 18 | Limits of extending Sections 5-6 | 100 sedenion samples test the Artin identity used in the eigenvalue transformation. Search signed basis pairs for an exact zero divisor, yielding a 1-by-1 triangular matrix with eigenvalue 0 different from its diagonal. |
| 19 | Additional root finding beyond the paper's examples | 32 unplanted `x²+2x+r` problems per level, levels 4 through 1024 as above. Solve in the complex subalgebra generated by independently sampled `r`; record maximum original-equation residual per level. |
| 20 | Section 4 general-field/signature supplement | Exhaust all 2,824 input vectors over F3/F5 for one bracketed quadratic in each of four +/- signatures in dimension 4. Independent recursive reference verifies E0-Em and counts roots/traces. This is not native kernel support for other fields/signatures. |
| 21 | Optional device replay | Dimensions 2, 4, 8, 16, 64, 256, 12 pairs each: `multiply()` versus sparse expected products. Shared typed replay additionally checks float16 and complex64/128 with its independent recursive oracle. |
| 22 | Optional device paper/deep checks | Direct `multiply()` evaluation of Example 5.3 and its eigenvector equation; quadratic identity at levels 4, 8 and configurable 4-20 (default 12). Float64 integer fixtures. |

## What is not coded or not established

| Content | Boundary of this companion |
| --- | --- |
| Sections 1 and 3's historical/categorical statements | No verification of cited literature, free-product universal properties, formal polynomial algebra construction, or proof assistant formalization. Concrete evaluation behavior is tested. |
| Section 2 arbitrary doubling parameters/base fields | Kernel checks use real gamma=-1. Finite-field supplement handles only primes 3 and 5, dimension 4, gamma +/-1; arbitrary parameters, extension fields and characteristic 2 are not implemented. |
| Section 3.1 arbitrary-degree roots/right factors | The degree-two identity is tested; no arbitrary-degree polynomial factorization engine. |
| Claim 4.1 and bounds in Lemma 4.2 / Theorem 4.4 | Selected exact fixtures and degree checks, not a universal proof or exhaustive extremal classification. The finite-field trace bound is weak here since each field has fewer than eight elements. |
| Section 4 a fully general symbolic zero-set solver | Exact rational determinant construction followed by numerical real candidates. Singular slices use float64 SVD with tolerance 1e-9; roots use exact rational isolating intervals then float64 solves. Ill-conditioned cases, near-coincident traces, irrational singular data, and certified completeness are not resolved. |
| Theorem 4.5 whole spheres | Centers/radii/nullities plus finite representatives, not enumeration of an infinite sphere. Dimension-one nullspaces yield pairs of points. |
| Theorem 4.6 all monic general quadratics | Six small unplanted general samples; deep unplanted tests have scalar linear coefficient. Deep planted bracketed roots check evaluation only, not general root discovery. No inference from sample success to a universal proof. |
| Section 5 all eigenvalues or diagonalization | Constructed `(1,x)` eigenvectors give some right eigenvalues. No complete arbitrary-matrix right spectrum, full left spectrum, or invertible eigenvector matrix U is computed. |
| Section 6 all fields/division algebras/matrix sizes | Real octonions, 3-by-3 finite fixtures (plus the 2-by-2 correction). No general division-algebra or characteristic-two implementation; repeated conjugacy coverage uses identical diagonal entries. |
| Device performance and very high dense dimensions | Both GPU tests skipped in the recorded environment. CPU sparse level 1024 is not a dense allocation or benchmark. Bit-operation cost grows with index size. |

Potential follow-ups: certified singular algebraic-number arithmetic; larger
generic unplanted quadratics with structured sparse linear solvers; distinct
but conjugate repeated diagonal entries; and the level-20 device run on your
CUDA host. These are extensions, not completed checks hidden in the counts.

## Further experiments

See [FRONTIER.md](FRONTIER.md) for a separate matrix-free numerical runner:
unplanted quadratics with non-scalar bracketed linear terms, and direct
right-eigenpair searches beyond octonions. It supports actual GPU products
through configurable level 20, records unsuccessful searches as unresolved,
and saves vectors and residuals. These additions have their own tests and
snapshots; the original 22-test report above remains unchanged.

## Batched GPU survey report

See [BATCHED_RESULTS_REPORT.md](BATCHED_RESULTS_REPORT.md) for the October 29
GPU batching benchmarks, the 8,192-matrix sparse survey and targeted continuation,
and the 1,024-matrix denser survey. These later experiments are separate from
the original paper-companion test counts above.

## Public evidence and local artifacts

A GitHub clone includes the companion source code, tests, documentation, and
curated JSON/CSV evidence under `results/`. Historical result files are retained
unchanged, including recorded environment paths, source hashes, and PDF hashes.
Those paths describe the original machine; they are not setup instructions or
files guaranteed to exist in a clone. Saved source hashes describe the source
at the time of each run, not necessarily the current revision.

Downloaded PDFs, extracted text, rendered pages, console logs, and bulk search
archives are local artifacts and are not distributed here. Commands using
`/path/to/...` require your own files. The optional `--paper` and `--reference`
arguments hash locally supplied PDFs for provenance; ordinary tests do not
require the PDFs. Fresh `/tmp/...` output paths in examples are local scratch
locations; copy new evidence to durable storage if you need to retain it.

Full historical solver archives, including binary NPZ vectors, are not included.
Compact survey and audit summaries cannot substitute for the complete archives
when continuing a saved search or rechecking every original residual. The
[portable eigenpair witnesses](results/frontier-eigen-witnesses.json) provide a
small self-contained selection of coefficients and vectors. Use the recorded
configurations and workflow instructions to generate new full archives;
hardware and library differences may change numerical results and timings.
