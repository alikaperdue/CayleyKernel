# Findings for version 1

These examples concern the supplied version 1 PDF. The relevant formulas were checked visually on printed pages 8 and 25, since extracted text loses conjugation bars and other notation. No claim is made about later versions or an author's intended corrections.

## Corollary 3.7: six positions can represent fewer than six vertices

The statement on page 8 assumes the hypotheses of Notation 3.1 and condition (*) for both factors, then says the displayed six positions give a 6-cycle. In a graph cycle, the six projective vertices must be distinct.

Use the split quaternions A2{−1,+1}, with component algebra A1{−1}=C, and choose

```text
a = b = c = d = e1
last doubling parameter gamma1 = +1
n(a) = n(b) = n(c) = n(d) = 1
chi = +1
```

The component algebra is associative, so all component strong-alternativity assumptions hold. Both factors satisfy (*). In the full split quaternion basis,

```text
(a,b) = (c,d) = e1 + e3
(e1 + e3)^2 = 0
ac = da = -e0
conjugate(ac) = -e0
```

The six positions in Corollary 3.7 are

```text
e1 + e3,
e1 + e3,
-e0 + e2,
e1 - e3,
e1 - e3,
-e0 - e2.
```

Every consecutive product, including the last times the first, is zero. There are only **four distinct projective lines**. Some consecutive positions are the same vertex, which does not give an edge under Definition 2.3's distinct-vertex requirement.

Thus the zero-product formulas hold in this example, but the conclusion of a genuine 6-cycle needs an additional distinctness condition or a restriction excluding this case. The ordinary-sequence examples in Sections 3–4 have six distinct lines and pass the cycle checks. This example does not contradict the separately hypothesized chordless ordinary hexagons tested in the suite.

Implemented in `PaperSections.test_03_01_reference_only_degenerate_cycle_and_zero_norm_branch`. This uses an independent parameterized reference because CayleyKernel's public routines implement the ordinary sequence only.

## Lemma 6.14(1): the printed y-list fails

On page 25, part (1) includes both `(c,d)` and `(c,gamma_n*d)` in the permitted choices of y. For gamma_n=−1 these are `(c,d)` and `(c,−d)`. Changing just that sign generally turns a zero product into a nonzero one, as also seen in Lemma 4.5.

Take n=4, all parameters −1, and four independent signed basis units in E3:

```text
a = e1
b = e2
c = e4
d = -e7
```

They satisfy the lemma's independent-component hypothesis. In the reduced algebra A4°=M4 (dimension 16),

```text
(a,b) = e1 + e10
(c,d) = e4 - e15
(a,b)(c,d) = (c,d)(a,b) = 0.
```

In A5=M5 (dimension 32), choose the explicitly listed choices x=`(a,b)` and y=`(c,gamma4*d)`:

```text
x = e1 + e18
y = e4 + e23
xy = 2*e5 - 2*e22 != 0.
```

The required orthogonality equivalence therefore fails for the list as printed. This product is reproduced with **both** `CayleyKernel.basis()` accumulation and independent recursive doubling; the counterexample is within CayleyKernel's supported ordinary algebras.

A plausible replacement is the Proposition 6.7 orbit:

```text
(c,d), (d,gamma_n*c),
(tilde(c),tilde(d)), (tilde(d),gamma_n*tilde(c)).
```

The suite checks that proposed replacement for every ordered quadruple of four distinct pure units from the seven lower-half units at the first nonvacuous tested dimension. It also checks part (2) as printed. Passing these finite checks does not establish an author's intended correction or prove a repaired general statement.

Implemented in `PaperSections.test_06_02_05_lemma_6_14_1_printed_counterexample`, with the proposed replacement and part (2) in `test_06_02_04_lemma_6_14_reduction`.

## Reading the test result

The counterexample tests are designed to pass when the documented zero/nonzero products and vertex counts are reproduced. Consequently, an overall successful suite does **not** mean every statement in the paper was verified as printed. It means the finite agreement checks and the documented discrepancy regressions behaved as described.

The optional GPU tests were skipped in the saved run because CuPy was unavailable. CPU results establish no execution claim about `multiply()` on this machine.
