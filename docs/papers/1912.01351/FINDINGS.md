# Findings for the supplied 1912.01351v1 PDF

These findings apply to the supplied version, with the conventions verified against its complete octonion multiplication table. They are not claims about subsequent versions or the cited literature. The [test inventory](README.md) and [saved data](results/results.json) distinguish exact counterexamples from numerical diagnostics. Passing regression tests below mean the discrepancies were reproduced.

## 1. Section 2 convention and notation cautions

The octonion table on page 4 is consistent with CayleyKernel after the permutation

```text
paper:   0 1 2 3 4 5 6 7
kernel:  0 1 2 4 3 5 6 7
```

All 49 imaginary table entries pass. A comparison without this translation could produce false failures. `basis()` additionally uses signed one-based labels; these must not be confused with either row of the table above.

The coordinate expression for N on page 5 appears to omit squares. Its interpretation as `z*conj(z)=|z|²` and all tests here use `sum(xi²)`.

## 2. Section 3: closure needs arithmetic conditions

The definition on page 6 uses a commutativity equation where its surrounding discussion calls for closure under multiplication. These are different: the ordinary integer octonion lattice is closed under products but contains units with `e1*e2=-e2*e1`.

Likewise, rational coefficients in a radical field do not by themselves make the Z-span of a chosen basis closed under multiplication. With `omega0=1`, `omega1=sqrt(2)*e1/2`, the square of omega1 is `-1/2`, outside the scalar Z-span.

If the alpha coefficients on page 7 are read as independently chosen integers, extra divisibility conditions are still needed. In kernel numbering take the radical weights `sqrt(1),sqrt(2),sqrt(3),sqrt(6),sqrt(5),sqrt(10),sqrt(15),sqrt(30)` and multiply only the e1 weight by 2. The product of the e2 and e3 generators is then `3/2` times the e1 generator. This does **not** refute the narrower construction in which composite generators are actual products of integer-scaled primitive generators. The unscaled canonical radical lattice passes all 64 structure-constant checks.

An arbitrary rational combination of field-basis vectors remains in the field's rational vector space; this weaker fact is not the same as remaining in a specified integer lattice.

## 3. Section 4.1: the unrestricted harmonicity assertion fails in sedenions

The first paragraph on page 8 states harmonicity for all left/right Cayley-Dickson regular functions. The octonion case has the expected property; the unrestricted statement does not hold for the usual sedenions.

Here all indices use **kernel zero-based numbering**. Consider the polynomial

```text
f(x) = x1*x10*e4 + x0*x1*e14 - x0*x10*e5 - x0²*e15.
```

The coefficient calculations give `Df=0` identically, whereas

```text
Laplacian(f) = -2*e15.
```

The test constructs the coefficients, checks every coefficient of Df, verifies the nonzero Laplacian with independent recursive multiplication, then evaluates the polynomial at a nonzero integer point and uses exact central differences. Since the polynomial is quadratic, those differences are exact derivatives, without a step-size error. This is a genuine regular, nonharmonic polynomial in the standard 16-dimensional algebra. It does not concern a failure of octonion composition.

The isolated-singularity condition in Definition 4.4 also displays a power of the fixed point a where distance to the singularity, `|z-a|`, is presumably intended. At a=0 the literal fixed-point condition degenerates. The implemented pole-order test uses distance scaling for the Cauchy kernel and does not silently implement the displayed definition as a general classifier.

## 4. Section 4.2: the printed zeta correction has the wrong sign for the stated primitive

Write `qj=partial_j q0` and `Vj(z)=xj-x0*ej`. The printed zeta summand on page 10, repeated on page 12, is

```text
q0(z+w) - q0(w) + sum_j Vj(z)*qj(w).
```

Its xi derivative for i>0 is

```text
qi(z+w) + qi(w),
```

while the displayed wp definition requires

```text
qi(z+w) - qi(w).
```

The difference is **2*qi(w)**, not zero in general. The single-summand check at `w=(2,1,0,1,0,0,0,0)` is exact. A finite anisotropic lattice sum confirms that this discrepancy need not disappear by finite-set symmetry. On a fully symmetric cubic cutoff the derivative sums can cancel, which is why testing only that special lattice could conceal the issue.

Using a minus sign before the Fueter correction gives the desired derivative for every finite partial sum, and removes its first-order Taylor term. The code therefore includes both the literal printed expression (`correction=+1`) and an explicitly proposed variant (`correction=-1`). It does not present the latter as a published correction.

For one far-field ray, doubling radius from 32 to 64 reduces the norm of the printed summand by about 252.34, consistent with radius^-8. The proposed variant gives about 501.32, consistent with radius^-9. A ray is not a proof of normal convergence on an eight-dimensional lattice. No certified infinite-series claim is made for either variant here.

## 5. Proposition 4.9: exact octonion counterexample

This finding uses the **paper's** basis labels and multiplication table. Take

```text
lambda=e1, mu=e2, z=e3.
```

All three have N=1. The printed left side is `(mu*q0(lambda*(z*mu)))*lambda`. The table gives

```text
z*mu               = -e6
lambda*(z*mu)      =  e7
q0(lambda*(z*mu))  = -e7
mu*(-e7)           =  e5
e5*lambda          =  e3.
```

The printed right side is `q0(z)=-e3`. Thus the two sides have opposite signs. In kernel numbering the same triple is `(e1,e2,e4)`. The test verifies the result with both public-basis arithmetic and independent recursive doubling, after separately verifying the paper's table.

Exhausting all 512 triples of positive octonion basis units, including 1, gives **344 equalities and 168 failures**. Every triple in the selected four-dimensional quaternion subalgebra passes. This is consistent with the problem being the unjustified reassociation in the two-sided step, rather than the separately tested conjugation or one-sided identities.

The proof on page 12 substitutes `f(lambda*z)` for an expression whose inner argument is `lambda*(z*mu)`. The former uses `(lambda*z)*mu`. Those arguments need not agree. Alternativity does not allow arbitrary three-factor reassociation.

## 6. Proposition 4.11: the general two-sided regularity claim also fails

Again take kernel units lambda=e1 and mu=e2. The transformed kernel has the explicit real-coordinate form

```text
F(z) = A*z / |z|^8,
A = diag(1,-1,-1,-1,1,1,1,1).
```

This follows by applying the bracketed real-linear map to all eight basis units, since multiplication by these units preserves norm. At z=1 its x0 derivative is -7, its imaginary derivatives are the remaining diagonal entries times their basis units, and both Dirac residuals are

```text
D F(1) = F(1) D = -8.
```

The exact test differentiates this expression and verifies both residuals. Consequently the general two-sided part of Proposition 4.11 is false under the verified convention. Its two one-sided special cases pass sixteen seeded dense exact checks and are not implicated by this example. Each scalar component of F can still be harmonic away from zero; harmonicity is weaker than regularity.

## 7. Theorem 4.14: failed cancellation and an adverse trace diagnostic

For the integer octonion lattice, lambda=e1 and mu=e2 are legitimate non-real lattice multiplicators. Their norms are one and their period map is an integer signed permutation. Hence its determinant index is one, the division set in the fundamental cell is `{0}`, and the division trace excluding zero is an empty sum, exactly zero.

The proof's singularity cancellation does not work: on `z=t*e4` in kernel numbering,

```text
(mu*q0(lambda*(z*mu)))*lambda - q0(z) = 2*t^-7*e4.
```

This is an uncancelled pole, not a remainder harmonic across the origin. The exact regression uses t=1/2,1/4,1/8. Thus the main proof cannot be accepted as written. Even after fixing the zeta correction sign, this bracket-sensitive leading term remains wrong.

There is also a separate numerical diagnostic of the **final** formula. On a cubic lattice, signed coordinate symmetry relates the eight Legendre constants to eta0. Using finite symmetric sums for the proposed zeta variant gives:

| Cutoff | Nonzero lattice points | eta0 scalar component | Formula's predicted trace component along kernel e4 | Exact division trace |
| --- | --- | --- | --- | --- |
| 1 | 6,560 | 4.54490053 | -9.08980107 | 0 |
| 2 | 390,624 | 4.21871449 | -8.43742898 | 0 |

The printed-plus and proposed-minus corrections have the same finite cubic sum up to rounding, because the first-derivative lattice sums cancel by cubic symmetry. The diagnostic does not depend materially on choosing between them on this lattice.

These nonzero predictions are concerning, but the infinite Legendre constants have not been enclosed with rigorous tail bounds. The table is therefore numerical evidence against the final formula for this example, **not** an exact disproof of the final formula on its own. The preceding counterexample to its singularity-cancellation step is exact. The cofactor elimination used later in the proof passes an independent synthetic-data check; that linear algebra is not what fails here.

## 8. What should be trusted from this run

The module's basis algebra agrees with the printed octonion table and independent doubling on the tested domains. The canonical integral radical lattice, one-sided Cauchy identities, period-map scaling and elementary division counts have successful finite checks. The paper's warning about failure of octonion module closure also has a concrete confirming example.

The two-sided function formulas require correction or additional hypotheses. No replacement general trace theorem is proposed here. The documentation explicitly retains unimplemented analytic results and the open questions in Section 4.4. Direct GPU replay remains unperformed because CuPy is missing; the two device tests are included for an environment with a working CUDA stack.
