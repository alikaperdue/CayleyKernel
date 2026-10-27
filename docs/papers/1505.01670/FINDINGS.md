# Findings for the supplied 1505.01670v1 PDF

The central radius and single-element power identities pass the finite checks in this review. The discrepancies below concern later statements or numerical evaluation. They refer to the supplied PDF only. The [complete inventory](README.md), [tests](../../../tests/papers/arxiv_1505_01670/test_sections.py) and [recorded observations](results/results.json) specify the bounds and evidence.

A passing counterexample regression means that the discrepancy was reproduced. A symbolic test on a restricted additive function is not an everywhere-defined implementation on R.

## 1. Sections 1–2: the core algebra agrees

The paper uses the same doubling formula as the module. Complete independent basis checks through dimension 64 agree with both public sign routines. The paper uses one-based coordinate names; its e4+e11 and e7-e16 become zero-based kernel e3+e10 and e6-e15.

For that pair, both module-based and independent products give the zero product, the nonalternativity witness, and the expected squared lengths. The quadratic annihilator and norm-power identity also pass exact seeded checks in dimensions 1–64. All parenthesizations through exponent 7 agree on the explicitly tested inputs.

For a nonreal element a, the minimal polynomial is

```text
t² - 2*a0*t + |a|².
```

Its roots are `a0 +/- i*|imaginary_part(a)|`, both of modulus |a|. For a scalar element the minimal polynomial is linear, including t at zero. The numerical root tests preserve this distinction instead of assuming the quadratic is always minimal. No discrepancy with the main radius theorem was found in the tested domain.

### The radius is not the spectrum of the left-multiplication operator

A useful independent diagnostic uses the printed sedenion zero divisor a. The real matrix of `x -> a*x` is singular, of rank 12, and has spectral radius 2. The element's polynomial radius is sqrt(2). These quantities must not be confused.

The paper's comparison with matrix algebras concerns matrices as elements of an associative algebra. It does not identify an arbitrary nonassociative element's radius with that of its left-action matrix. This diagnostic is a clarification, not a discrepancy in the paper.

## 2. Section 3: scope of the pathological-function model

The paper allows a nonzero additive, pi-periodic real function phi. To make concrete symbolic checks, prescribe on the rational span of pi and 1:

```text
phi(p*pi+r) = r,    p,r rational.
```

Pi and 1 are rationally independent. This prescription therefore extends to a Q-linear map on R by extending a Hamel basis. Every such extension has phi(pi)=0 and phi(1)=1. It is pi-periodic and cannot be continuous: a continuous additive real function is linear, and vanishing at pi would force it to vanish at 1 as well.

This extension argument is mathematical, not executable. The code stores the two rational coefficients and computes phi only on this restricted domain. It never approximates phi by applying a discontinuous oracle to a floating-point atan2 result.

On C, with the **oriented** argument in [0,2*pi), subtracting multiples of 2*pi leaves phi unchanged. Thus `phi(arg(z^k))=k*phi(arg(z))`. Multiplication by a negative real shifts the angle by pi and also leaves phi unchanged. The complex construction behaves as advertised for these exact prescribed values, and for any global extension with those values. The tested Gelfand expressions also include non-Euclidean lp factors.

The explicit angle approximants in the tests illustrate zero and nonzero phi values near the same targets. For any target t and prescribed rational r, choose a rational p of denominator D nearest `(t-r)/pi`. Then

```text
|p*pi+r-t| <= pi/(2*D),    phi(p*pi+r)=r.
```

This bound explains how the restricted domain already supports dense zero/nonzero examples and arbitrarily large positive and negative prescribed phi values near an angle. The finite examples alone are not a topological proof on all R.

## 3. Theorem 3.3(a): folded quaternion arguments break homogeneity

The paper defines a quaternion argument by replacing its imaginary part with its **nonnegative Euclidean length** and then taking a complex argument. Consequently its value lies in [0,pi]. It loses the orientation that made the complex construction work.

Take the unit quaternion

```text
q = cos(1) + i*sin(1),
```

and the base subnorm f=Euclidean length. With the permitted phi prescription above,

```text
arg(q)  = 1,       phi(arg(q))  =  1,
arg(-q) = pi-1,    phi(arg(-q)) = -1.
```

The displayed quaternion h therefore satisfies

```text
h(q)=e,    h(-q)=e^-1.
```

A real subnorm must have h(-q)=h(q). The stated formula thus fails absolute homogeneity and cannot be the claimed subnorm. This is an exact sign contradiction for every Hamel extension with the prescribed values. Numerical module powers and coordinate arguments confirm the selected branches; the phi values are symbolic rational values throughout.

## 4. Theorem 3.3(c) and Theorem 5.2(b): power compatibility also fails

For the same q,

```text
arg(q^4) = 2*pi-4,
phi(arg(q^4)) = -4,
h(q^4) = e^-4,
h(q)^4 = e^4.
```

So the quaternion power equation in Theorem 5.2(b) fails already at exponent 4. The exact log defect is -8. The function is not a submodulus as printed, independently of its earlier homogeneity failure.

More generally, at integer exponent k the folded argument gives phi equal to either k or -k, according to the sign of sin(k). The kth-root values are therefore e or e^-1. The finite test records both behaviors through k=64, including k=4 and k=7. The fact that the irrational rotation by 1 radian visits both semicircles infinitely often then shows the claimed limit in Theorem 3.3(c) does not exist for this q. That last infinite-return assertion is part of the mathematical argument; it is not proved by the finite numerical list.

The complex version retains the oriented angle and does not have this failure. No replacement quaternion argument or corrected quaternion theorem is silently introduced in the code.

## 5. Corollary 4.2: continuity is missing from the unrestricted wording

Theorem 4.1 assumes a **continuous** subnorm when identifying stability with radial dominance. Corollary 4.2 states the equivalence for a subnorm on An without repeating continuity. Read literally, that is too broad; the paper's later complex examples already demonstrate why.

For f=Euclidean norm on C, use the valid oriented complex construction g and the same phi. Additivity makes g satisfy the power equation exactly, so g is strongly stable. Real absolute homogeneity follows from pi-periodicity. But at the unit complex point with angle pi-1,

```text
phi(pi-1)=-1,
g(z)=e^-1 < 1=|z|.
```

It is stable and fails radial dominance. The code checks the prescribed angle and its powers exactly; the global counterexample uses the stated Hamel-extension argument and oriented complex formula. This is not a full numerical implementation of g on all complex inputs.

The lp and weighted norm examples remain continuous, so their use of the equivalence is unaffected. The corrected scope of the corollary should include continuity; it should not be interpreted as overriding the later discontinuous examples.

## 6. Theorem 4.3: the quaternion formula has no uniform stability bound

There is a stronger failure than a single power mismatch. Let m be a positive integer and define

```text
M = floor(2*m*pi)-4,
theta = m*pi-M/2,
q_m = cos(theta)+i*sin(theta).
```

Since `4 <= 2*m*pi-M < 5`, theta lies in [2,2.5), inside (0,pi). Therefore

```text
phi(arg(q_m)) = -M/2.
```

The doubled angle is in [4,5), so quaternion argument folding gives

```text
arg(q_m²) = 2*pi-2*theta,
phi(arg(q_m²)) = M.
```

All points are units. The stability ratio at the **fixed exponent 2** is

```text
h(q_m²) / h(q_m)² = exp(2*M).
```

M grows without bound. There can be no uniform sigma as required by the paper's definition of stability, even if one overlooks that h has already failed the subnorm definition.

| m | M | log stability ratio at k=2 |
| --- | --- | --- |
| 1 | 2 | 4 |
| 2 | 8 | 16 |
| 10 | 58 | 116 |
| 100 | 624 | 1248 |

The test uses exact rational phi values and actual module quaternion squares to confirm the angle branches. Logs avoid overflowing exp(1248). The argument for unboundedness uses the displayed family for all m, rather than interpreting four examples as a proof by sampling.

## 7. Page 10: strict boundary needed for stable but non-strong weighted l1 norms

The preceding two criteria are consistent:

```text
stable:          u>=1 and v>=1,
strongly stable: v²>=u>=1.
```

Their difference is

```text
stable but not strongly stable: u>v²>=1.
```

The printed final sentence includes the boundary u=v² using >=. For example, u=4,v=2 is on that boundary and is strongly stable. This can be seen without a large computation: the conditions u>=1 and v²>=u make the weighted l1 norm submultiplicative by applying the triangle inequality to a complex product. Submultiplicativity implies strong stability. All 625 tested coordinate-grid products agree.

For contrast, u=4,v=1 is truly stable but not strongly stable: on i, f(i²)=4 while f(i)²=1. A uniform sigma=4 suffices, so failure of sigma=1 does not imply instability.

The sentence that the lp criterion holds only for p<=2 also implicitly needs dimension at least two. On A0 every lp function is just absolute value and is strongly stable for every positive p. This dimension exception is checked in the code.

## 8. Numerical issues are distinct from mathematical counterexamples

### Repeated floating-point powers beyond octonions

A seeded 32-dimensional unit input, repeatedly multiplied using the CPU bilinear adapter, shows squared-norm errors about 1.55e-6, 7.41e21 and 1.68e77 at exponents 64,128,256. Exact integer powers followed by normalization retain unit length and satisfy the Gelfand bounds.

This reflects amplification of floating-point perturbations outside the two-dimensional subalgebra generated by the exact element. A unit multiplier need not be length-preserving on arbitrary elements in these higher algebras. It is not evidence against exact power-associativity or the radius theorem. The saved record retains both computations so numerical artifacts cannot masquerade as a theorem failure.

The observation concerns CPU floating-point arithmetic through the module's basis signs. The GPU tests were skipped, so no GPU numerical behavior is claimed.

### Lemma 5.1 near the negative real axis

The literal square-root formula involves `sqrt(2*a0+2*|a|)`. For a=(-1,1e-12), float64 rounds |a| to 1, making the denominator zero. This is cancellation in evaluation, not a failure of existence of the square root.

The test-only stable reformulation gives b=(5e-13,1), whose square reproduces a to the stated precision. Exact rational generated-target tests and ordinary dense numerical inputs also pass. The implementation does not modify CayleyKernel or claim a general production-quality square-root API over every floating-point scale.

## 9. Overall scope

The core radius result, exact powers, quadratic relation, printed zero divisors and square-root construction have successful finite checks in the ordinary real Cayley-Dickson sequence. Later quaternion pathological-function claims are incompatible with the explicitly displayed folded argument. Two stability statements also need qualifications as detailed above.

The suite does not prove universal identities, construct an everywhere-defined pathological additive oracle, characterize every strongly stable subnorm, or prove uniqueness among all continuous submoduli. These limits are recorded in the main inventory rather than hidden behind a green test result.
