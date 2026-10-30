# Results and interpretation

The source of truth for this run is [results.json](results/results.json).
The runner stores per-test outcomes, source/PDF hashes, versions and seeded
experiment data. No original PDF or raw console log is added to Git.

## Sections 1-3: conventions and loss of structure

The ordinary CayleyKernel signs match the paper's doubling convention on all
ordered basis pairs through dimension 64. This is essential: borrowing the
paper's octonion formulas under a different basis convention could otherwise
produce misleading failures.

The tests distinguish three things that can look alike on paper: formal left
polynomial multiplication, substitution into those polynomials, and evaluation
of explicitly bracketed general polynomials. Evaluation of left polynomials
is not multiplicative. The octonion right-factor example works because of
alternativity; sedenions provide failures. These are expected changes in
structure, not module defects.

## Section 4: regular and singular roots

The implementation builds `G[:,j]=L(e_j)` using actual bracketed products.
It then builds `M(t)=2tI+G[1:,1:]` and its Cramer determinants exactly with
SymPy. Clearing denominators in E0 produces a polynomial of degree at most
`2*dimension`. Denominator factors are removed from the regular candidate
polynomial and handled by separate singular slices. They must not be treated
as ordinary isolated roots merely because they appear in a cleared numerator.

The singular-slice implementation finds an affine solution space, projects
the sphere center onto it, and subtracts the squared projection distance
from the sphere's squared radius. It exercises inconsistent linear systems,
empty intersections, tangent points, zero-dimensional spheres (two points),
and higher-dimensional spheres. Exact fixtures include a consistent singular
trace with nonzero Cramer minors, so common-factor tests are not exclusively
checks of zero polynomials.

The six unplanted general examples in dimensions 2, 4 and 8 all produced
roots satisfying their original equations. That confirms these cases of the
method; it does not certify completeness of every reported numerical root
set. Exact isolation of scalar roots does not eliminate conditioning issues
in subsequent floating-point linear solves.

The finite-field supplement exhausts all vectors in eight algebra/field
configurations, a total of `4*(3^4+5^4)=2824` inputs. Root counts for gamma
pairs `(-1,-1),(-1,1),(1,-1),(1,1)` are respectively `4,6,4,2` over F3 and
`4,4,4,4` over F5 for the fixed polynomial `x²+e1*(x*e2)+x*e3+1`.
All coordinate reductions agree. These counts describe that one polynomial;
Theorem 4.6 asserts existence over the locally-complex real sequence, not
these finite fields.

## Section 5: the printed example works

With zero-based basis notation, Example 5.3 gives exactly

- `y = e5 + e7`,
- `x = e4 - e6`,
- `lambda = e0 + e5 + e7`.

Both the transformed polynomial and the original right-eigenvector equation
vanish exactly on the printed vectors. Independently solving the transformed
polynomial also gives candidates that pass the original equations. Two
additional independently sampled octonion matrices pass the same pipeline.

The brackets in `b*((b_inverse*y)*a) - b*(d*(b_inverse*y))` are kept exactly
as printed. Ordinary associative matrix shortcuts would not justify this
calculation.

## Section 6: why back substitution matters

Upper/lower triangular fixtures include repeated diagonal entries and both
left and right eigenvalue equations. The earliest matching diagonal selects
a valid pivot; Sylvester systems above it are solved as real linear systems.
Every row is checked afterward.

For the correction discussed on pages 13-14, take `B=[[0,1],[0,i]]`.
The vector `(0,1)` fails because its first output coordinate is 1. The vector
`(-i,1)` succeeds: `B*(-i,1)=(1,i)=i*(-i,1)`. A diagonal entry being an
eigenvalue does not imply its standard coordinate vector is an eigenvector.

## Extensions: what survives at high levels

Two distinct experiments reach doubling level 1024 using sparse coordinates:

1. Exact integer families, including translated pure-imaginary quadratics and
   bracketed polynomials with planted roots, at 16 samples per level.
2. Actual unplanted root finding for `x²+2x+r`, with 32 independently sampled
   constants at each of levels 4, 8, 16, 32, 64, 128, 1024 (224 problems).

For the second experiment set `w=1-r`. In the complex subalgebra generated
by `r`, a square root has scalar part
`a=sqrt((sqrt(w0²+||Im(w)||²)+w0)/2)` and imaginary part `Im(w)/(2a)`.
Then `x=-1+sqrt(w)` solves the equation. Fixtures always have nonzero
imaginary part, so the formula's zero-denominator branch is not exercised.
Every answer is checked with sparse CayleyKernel products in the full ambient
algebra, rather than by the scalar formula alone.

The largest recorded coefficient residual is approximately **7.11e-15**;
the acceptance tolerance is `1e-10`. The level-1024 maximum is approximately
`6.22e-15`. There is no increasing-error claim: the tested supports remain
small and coefficient magnitudes bounded. The JSON records every level's
maximum. These sparse calculations are neither dense dimension-`2^1024`
products nor timing evidence for the GPU.

## Extensions: what fails in sedenions

The exact pair

```
a = e1 - e10
x = e4 + e15
```

has `a*x=0` with both vectors nonzero. Consequently the 1-by-1 matrix `[a]`
has left eigenvalue zero, which differs from its sole diagonal entry. This
is a concrete failure of extending Proposition 6.2 to ordinary sedenions;
its division-algebra hypothesis is essential. It does not refute the paper.

Also, **95 of 100** seeded sedenion pairs violate the particular identity
`b*(x*(b*x)) = (b*x)*(b*x)` used in Section 5. The JSON preserves the first
exact witness and its nonzero residual. This rules out carrying the paper's
proof over unchanged. It does **not** prove that sedenion 2-by-2 matrices
lack right eigenvalues, nor that some other existence argument is impossible.

## Execution boundary

The suite completed 20 CPU checks successfully and skipped both device checks
because CUDA reported no device. Runtime is about seven seconds on the recorded
environment, including symbolic work; consult JSON for the actual duration.
No GPU timings, dense level-20 results, or typed GPU residuals were observed.
The device code is available for reproduction on a CUDA host.

The companion also passed repository pytest collection and execution using
`python -m pytest -q tests/papers/arxiv_2607_23613/test_sections.py` in a separate Python environment:
20 passed, 2 skipped in 7.36 seconds. The saved JSON is the separate unittest
run, not a relabeling of this integration check. A repository-wide whitespace
check reported an existing change in `cuda/amd_kernel.cu`; this review did not
modify that file. The new companion files passed their whitespace checks.
