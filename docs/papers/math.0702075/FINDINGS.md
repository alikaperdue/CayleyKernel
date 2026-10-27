# Findings for the supplied math/0702075v2 PDF

The core splitting and annihilator computations agree with the paper on the tested domains. Several supporting statements need missing hypotheses or corrected wording. These are preserved as exact regressions, rather than silently repaired and reported as passes of the original statements.

See the [complete inventory](README.md), [results](results/results.json) and [exact annihilator certificates](results/annihilators.json). All indices in the examples below are mathematical zero-based basis indices, matching `twist()`; signed `basis()` labels have a separate one-based convention.

## 1. The principal multiplication formula is certified in four fixed dimensions

Let

```text
B(a,b) = (a+b, i_n*(-a+b)) = sqrt(2)*{a,b}_paper.
```

Theorem 4.5 becomes the exact rational identity

```text
B(a,b)*B(x,y) = 2 * [
    B(pi_perp(a*x), pi_perp(b*y))
    + tilde(pi_C(x*a+b*y))
    + tilde(pi_C(a*y-x*b))*(0,i_n)
].
```

Every ordered basis pair of the combined input space C_n-perp × C_n-perp was checked:

| Target algebra dimension | Number of ordered basis pairs |
| --- | --- |
| 16 | 144 |
| 32 | 784 |
| 64 | 3,600 |
| 128 | 15,376 |

Both sides are bilinear in the two combined inputs. Their agreement on a basis certifies the equality for every real vector in these fixed subspaces, not just for sampled integer points. The three terms land in mutually orthogonal coordinate subspaces. No general arbitrary-n proof is claimed by the code.

The normalization factor is handled explicitly throughout. Annihilators are unaffected by the common nonzero scalar relating B to the paper's braces, but products, norms and the relative coefficients in Theorem 6.12 are not allowed to drop that factor.

## 2. Lemma 2.15 and Corollary 2.18 need a restriction on a

Lemma 2.15 on page 6 is stated for arbitrary a,b in A_n. Its proof uses Lemma 2.6, which requires a in C_n-perp. That hypothesis matters.

In A3, C3 is the span of 1 and e4. Take

```text
a=b=1+e1.
```

The C-linear projection of b onto the C-linear span of a is b itself, so b'=a. But

```text
a*b=2*e1,
pi_C(a*b)=0,
a*b'=2*e1 != 0.
```

This contradicts the printed equality `pi_C(a*b)=a*b'`.

The same example contradicts Corollary 2.18 as worded: b belongs to the C-span of a (indeed b=a), but a*b does not belong to C3. The element a has zero annihilator, so adding Ann(a) does not change the example.

With a in C_n-perp, the projection identities pass the sampled tests, and the entire preimage of C_n under left multiplication equals `span_C(a)+Ann(a)` on the specified fixtures. Later multiplication calculations use precisely this restricted setting. These regressions do not refute the central Theorem 4.5, which is independently certified above.

## 3. Zero and one-zero-component exceptions must be excluded

### Theorem 2.11

The upper bound `2^n-4*n+4` applies to annihilators of **nonzero** elements. The literal phrase “any annihilator” includes

```text
Ann(0)=A_n,
```

whose dimension 2^n is larger than the bound. This is checked in levels 3,4,5. All substantive upper-bound and maximality checks use nonzero elements.

### Proposition 5.11

The nonzero requirement on a is omitted from its displayed assumptions. At a=0 in the source A3, the target is zero in A4, with annihilator dimension 16. The printed expression gives

```text
dim Ann(0 in A3) + 8 - 4 = 12.
```

Moreover zero is not a zero-divisor under the paper's explicit definition, which requires the element itself to be nonzero. Both left/right formulas pass full-subspace checks for the tested nonzero a.

### Lemma 6.4

This lemma assumes only that the combined element {a,b} is nonzero and outside D. It needs both components nonzero for the stated dimension formula.

Take a=e1 in A3 and b=0. The combined element is nonzero and not in D because a cannot be orthogonal to Ann(0)=A3. Its annihilator in A4 lies in H4-perp and has dimension 4, while the printed sum is

```text
dim Ann(e1 in A3) + dim Ann(0 in A3) = 0+8 = 8.
```

The test explicitly checks the intersection with H4-perp as well as the full kernel. Theorem 6.7 does require both components nonzero; its tested off-D cases pass.

### Lemma 7.1

The literal statement also permits b=0 and alpha=0 while a is nonzero. Its second braced argument is then zero, whose annihilator is all A4. The nonzero first braced argument cannot be orthogonal to that whole space. The nonzero zero-divisor uses needed by Proposition 7.3 avoid this degenerate exception and pass the checked cases.

## 4. Lemma 6.3's proof drops a projection

The statement correctly contains `pi_C(a*y-x*b)`. The proof on page 14 says its vanishing includes requiring `a*y=0` for all y in Ann(b). The corresponding condition is only

```text
pi_C(a*y)=0.
```

The tests retain a concrete D pair and a vector y in Ann(b) for which a*y is nonzero but has zero C projection. The complete parameter-basis vanishing criterion of the **statement** agrees with D membership on the tested examples. This is a proof-text issue, not failure of the tested lemma statement.

## 5. Lemma 6.9 names the wrong annihilator for uniqueness

The lemma asks for a unique x satisfying a*x=b and x perpendicular to **Ann(b)**. The following definition, 6.10, instead uses **Ann(a)**, which is the correct source-kernel complement.

An exact example in A4 is

```text
a = e1+e13,
b = e1,
x = (1+e12)/2,
x' = (1+e12)/2 + e6+e10.
```

Direct multiplication gives

```text
a*x = b,
a*x' = b,
x != x'.
```

Ann(a) has dimension four; b is perpendicular to it. Ann(b) is zero because b is a basis unit. Therefore **both** x and x' are perpendicular to Ann(b), contradicting uniqueness under the printed condition.

The implementation solves a*x=b with x perpendicular to Ann(a), following Definition 6.10. This gives a unique rational answer whenever the target belongs to the image. It also rejects a target outside the image. It does not attempt to obtain the quotient by blindly multiplying by a conjugate inverse, which is insufficient when a is a zero-divisor.

## 6. Theorem 6.12's full decomposition passes

For seven tested nonzero D pairs in source levels 3,4,5, the three predicted C-subspaces span exactly the directly computed annihilator and are mutually C-orthogonal. The dimension is the sum of the component dimensions plus four.

In rational B coordinates, the third generator is

```text
B(b/a,-a/b) + 2*(0,i_n).
```

The factor two is the relative coefficient after multiplying the paper's generator by sqrt(2). Dropping it would spoil cancellation. The exact full-subspace tests preserve it.

The 316-pair finite census also agrees with the D/non-D dimension formula:

- 36 source A3 basis pairs, giving target nullity 0 or 4;
- 196 source A4 basis pairs, giving target nullity 0 or 4;
- one fixed sedenion zero divisor against 84 signed basis-frame zero divisors, giving nullity 12 on six D cases and 8 on 78 non-D cases.

This finite census does not exhaust all real component pairs. Full kernel data for selected D and off-D examples are exported independently of the prediction formulas.

## 7. Section 7 eigenspaces need unit normalization

The paragraph before Corollary 7.4 begins with an arbitrary zero-divisor a in A4, defines a geometric four-plane using `(y,c*y)`, and states the equation

```text
a*(a*x)=-2*x.
```

For general scale, the equation is instead

```text
a*(a*x)=-2*|a|^2*x.
```

On the tested basis-frame elements a=(e_i,+/-e_j), i!=j, the squared norm is 2. Direct products on the stated four-plane give `-4*x`, not `-2*x`. Dividing a by sqrt(2) gives the intended unit-length eigenvalue 2.

The geometric four-plane and the six-plane membership criterion are unchanged by rescaling a. Thus the computed Corollary 7.4 criterion still passes on all 504 tested pairs: 36 in D and 468 out. The normalization issue affects the unqualified eigenvalue characterization, not that span criterion.

The following Stiefel identification also needs the usual rescaling: two orthogonal unit octonion halves form a vector of squared length 2 in A4, not 1. The map to the unit zero-divisor locus includes division by sqrt(2). This is a fixed rescaling diffeomorphism, not a counterexample to the bundle statement.

The code checks the local algebraic six-plane and the second-half sign-flip isomorphism between the four-dimensional fibers. It does not verify global bundle charts, G2 topology or the claimed diffeomorphism in Theorem 7.5/Remark 7.6.

## 8. Section 8 constructions and thresholds agree on exact examples

The recursively constructed maximal examples have annihilator dimensions

```text
A3:0, A4:4, A5:16, A6:44, A7:104, A8:228.
```

The alternative-element family of Proposition 8.2 gives exact full kernels outside H-perp of dimensions 4,12,28,60,124 in target levels 4–8, four below half the corresponding algebra dimension.

Both-component D examples in target levels 4–8 have nullities 4,12,36,92,212. In levels at least five these are exactly four below Proposition 8.6's one-component forcing bound. This checks the advertised sharpness construction on the named levels.

The stability cutoff is

```text
n >= c/4+4,
```

not n>=4*c+4. The fraction was checked in the rendered pages. All 1,981 allowed integer parameter pairs through source level 12 satisfy the expected partition of the two arithmetic conditions.

Actual two-component nonstable witnesses are constructed at source levels n=5,6,7 with c=8,12,16 and target nullities 36,92,212. Both B coordinates are nonzero, so these points are outside both one-slot images. Dividing by the square root of their saved squared norms gives unit points without changing kernels.

At the stable boundary, both one-slot lifts give target nullities 16,40,96,216 from source nullities 4,12,36,92 in levels 4–7. These are concrete points satisfying the predicted formulas. They do not establish equality/diffeomorphism of entire geometric spaces for arbitrary n.

## 9. Corollary 8.13 needs its unit-length context

The literal wording describes all zero-divisors in A5 with 16-dimensional annihilator as two copies of V2(R7). Without a length condition, every nonzero real scalar multiple of such an element has the same annihilator, so this space contains unbounded rays and is noncompact. Two copies of the Stiefel manifold V2(R7) are compact.

The intended statement concerns the **unit-length** locus T5^0 introduced earlier in the section. The tests verify invariance of kernels under scaling and the expected 16-dimensional examples, while recording this scope distinction. They do not use a finite numerical test to claim a topology theorem.

Similarly, the A6 nullity-40 examples from A5 nullity-12 components agree with Corollary 8.14 at the level of dimensions and the two lift maps. Its global diffeomorphism still requires the paper's mathematical argument.

## 10. What the saved certificates establish

Every one of the 27 exported examples includes an exact kernel basis. Each listed vector has zero product with the element. Independent modular ranks certify both the multiplication-operator rank and independence of the proposed kernel, giving matching lower/upper rational-rank bounds. These are finite, reproducible annihilator certificates, including the A8 nullity-228 example.

The suite does not solve the three open classification questions, construct every real zero divisor, or prove the global smooth topology assertions. Three GPU replays are provided but were skipped because CuPy is unavailable. No GPU result is inferred from the CPU checks.
