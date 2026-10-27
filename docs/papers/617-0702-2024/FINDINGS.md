# Findings and corrections for article 617-0702/2024

These findings concern the eight-page supplied PDF, *Graphical and Computational Analysis of Cayley-Dickson Algebra Elements over Z_p*. See [README.md](README.md) for all tests, conventions, exact code and reproduction. Page numbers here refer to PDF pages.

The reported successes concern the explicitly defined valid standard and split Cayley–Dickson models. They do not validate every printed table or proof. In particular, the actual Table 1 agrees with CayleyKernel; the selected Table 2 entries do not define the composition algebra needed by the argument.

## Summary of the main numerical comparisons

All counts below include zero in the nilpotent and intersection sets. A zero divisor follows the paper's explicit nonzero definition.

| Quantity, dimension 8 | p=3: paper | p=3: verified | p=5: paper | p=5: verified |
| --- | --- | --- | --- | --- |
| Idempotents, Theorem 1 | 758 | 758 | 15,752 | 15,752 |
| Nilpotents, Theorem 3 | 729 | 729 | 15,625 | 15,625 |
| Common-coordinate nilpotents, Theorem 5 | 33 | **297** | 145 | **3,625** |
| Zero divisors, Corollary 6 | 2,295 | **2,240** | 78,875 | **78,624** |
| Units, Theorem 7 | 4,266 | **4,320** | 311,750 | **312,000** |

These small-prime results use complete enumeration in both standard and split models, with actual modular products. Independent quadratic-form convolution agrees. Unit/zero-divisor classification uses separately checked composition identities; it is not assumed for all higher-dimensional algebras.

The CSV includes additional primes and separately distinguishes norm-zero elements including zero from nonzero zero divisors. The figures show formula evaluations, not newly collected experimental observations.

## 1. Introduction and multiplication conventions

### Standard multiplication is verified; Table 2 has a concrete conflict

All 64 entries transcribed from Table 1 agree with CayleyKernel, including the signs. The supplied PDF places many table symbols in tiny separate images, so enlargement is needed; text extraction alone does not recover this table reliably.

For the intended split norm, `N(e1)=N(e2)=1` and `N(e4)=-1`. Table 2 prints `e1*e2=e4`, with diagonal entries `e1^2=e2^2=-1` and `e4^2=1`. Consequently its selected entries give

`N(e1*e2)=-1`, but `N(e1)*N(e2)=1`.

These differ in every odd characteristic. This is an internal conflict with the split composition norm, rather than merely a different basis ordering. Only the selected Table 2 cells are transcribed; no claim is made to audit every remaining cell or to classify the algebra obtained by taking the whole table literally.

The code therefore specifies the valid split model by doubling parameters `(-1,-1,+1)` and tests it independently. It does not substitute the problematic Table 2 into the enumeration without warning.

### Introductory sign and terminology issues

The complex-pair formula on p. 1 uses `ac+bd` instead of `ac-bd`. Its generator then squares to +1, so the displayed formula is the split-complex product rather than the standard complex one.

The split-quaternion relations on p. 2 combine `i^2=1`, `j^2=k^2=-1`, `ij=k` and anticommutation. Associativity would require `(ij)^2=-i^2*j^2=+1`, contradicting the displayed `k^2=-1`. These equations are not used as a consistent model by this suite.

The real-quaternion prose on p. 1 also calls the same algebra a division algebra and “not a division ring.” This wording is inconsistent. Over finite prime fields the tested quaternion models do have nonzero zero divisors, but that is a different setting from the real algebra discussed there.

### The valid split and Zorn models really are connected

The Zorn multiplication printed on p. 3 is implemented with its upper cross term `+w cross w'` and lower cross term `-v cross v'`. For split coordinates `(q0,...,q7)`, the verified map to Zorn coordinates is

```
a = q0+q4                     b = q0-q4
v = (q1-q5, q2-q6, q3-q7)
w = (-q1-q5, -q2-q6, -q3-q7).
```

Its inverse uses division by 2 and therefore is applied only in odd characteristic. All basis products commute with this map in the tested fields, and the determinant `ab-v dot w` equals the split norm.

For each of the eleven plotted odd primes, the standard and split octonions are also connected by an exported explicit algebra isomorphism. Choose a quaternion u with `N(u)=-1`; then `(a,b) -> (a,u*b)` maps standard multiplication to split multiplication. This explains the matching element-type counts over these finite fields. It is not a claim that the real octonion division algebra is isomorphic to the real split algebra.

## 2. Methodology and what was actually computed

The suite enumerates every element of dimensions 4 and 8 over F3/F5, for both standard and split signatures: 795,784 elements across eight cases. It computes actual squares and conjugate products, rather than only plugging p into formulas. Norm-residue convolution provides an independently implemented count.

No MATLAB source, executable listing or full pseudocode algorithm is present in the supplied eight-page PDF. The code and charts in this repository are new implementations. The paper's data-availability statement says supporting data are available from the corresponding author; no such external data were obtained for this review.

For dimensions 16–64, exact dynamic counting is possible for idempotents and nilpotents because their conditions reduce to a scalar coordinate and a quadratic form. This does not require enumerating `p^64` elements or inventing a higher-dimensional 3-vector cross product. The total count of assignments in the convolution is exact, although only the listed dimensions/primes are run here.

## 3. Numbered results

### Theorem 1: correct count, incomplete case argument

The final formula `p^6+p^3+2` agrees with complete standard/split enumeration at p=3 and p=5 and with independent counting at all plotted primes.

The proof's Cases II and III, however, each omit a factor of two: when one vector is zero and the other nonzero, `(a,b)` can be either `(0,1)` or `(1,0)`. Each case contributes `2*(p^3-1)`.

At p=3 the actual Zorn cases are:

| Case | Verified count |
| --- | --- |
| I: both vectors zero | 4 |
| II: v=0, w nonzero | 52 |
| III: v nonzero, w=0 | 52 |
| IV: v,w nonzero | 650 |
| Total | 758 |

Using the two printed values 26 instead would sum to 706, not 758. Case IV also cannot exclude `a=0,1`: nonzero orthogonal vectors exist. One exported idempotent is `a=0,b=1,v=(0,0,1),w=(0,1,0)`.

The Case IV total `(p^3-1)(p^3-2)` can be recovered correctly by keeping those endpoints: for each nonzero v, endpoints contribute `2*(p^2-1)` choices of w, and the other p-2 values of a contribute `(p-2)*p^2` choices. Their sum is `p^3-2`.

### Corollary 2: higher idempotent count supported by the quadratic identity

For the specified models,

`x^2 - 2*x0*x + N(x)*1 = 0`.

A nontrivial idempotent must have trace 1 and norm zero. In odd characteristic this fixes `x0=1/2` and leaves an imaginary quadratic equation with target `-1/4`. Convolution gives the printed `p^(d-2)+p^(d/2-1)+2` in every tested prime/dimension/signature case through d=64.

This supports the numerical corollary by an appropriate method. The proof's arbitrary-dimensional Zorn-vector extension is not implemented: its 3-dimensional cross product does not supply such an extension, and `p^(d/2-1)` is a number of possible vectors, not the vector dimension `d/2-1`.

### Theorem 3 and Corollary 4: odd-prime count agrees; matrix proof and p=2 need care

For a nonzero nilpotent in an odd-characteristic quadratic algebra, the minimal polynomial is `t^2`, so trace and norm vanish and the element has square zero. With `x0=0`, the remaining odd-dimensional quadratic form has `p^(d-2)` zeros in the tested cases. Theorem 3's `p^6` therefore agrees.

The ordinary 2-by-2 matrix argument does not establish the octonion result. Ordinary matrices over Fp have `p^4` elements and `p^2` nilpotents. The Zorn space has `p^8` elements, a different multiplication, and nonassociativity. Reusing an ordinary-matrix nilpotency probability with a changed denominator is not justified merely by writing both objects as 2-by-2 arrays.

Corollary 4 says “p a prime,” omitting the odd-prime restriction. At p=2, complete enumeration gives 8 nilpotents in dimension 4 and 128 in dimension 8, whereas the printed formula gives 4 and 64. In this characteristic the standard model has `x^2=(sum xi)^2*1`; its square-zero locus has `2^(d-1)` elements.

### Theorem 5: a missing ternary factor

Use the shared coordinates and the two norm forms printed in the proof. Common nilpotents satisfy `q0=0` and

```
A+B=0,  A-B=0,
A=q1^2+q2^2+q3^2,
B=q4^2+q5^2+q6^2+q7^2.
```

For odd p this is equivalent to **both** `A=0` and `B=0`. The proof keeps only the second block's count. There are `p^2` choices for the first block and `p^3+p^2-p` choices for the second, so the full common-coordinate count is

`p^2*(p^3+p^2-p) = p^5+p^4-p^3`.

Both counts include zero. Excluding zero subtracts one; it cannot explain away a factor of p squared. Complete direct squaring in both models gives 297 versus 33 at p=3 and 3,625 versus 145 at p=5. The regenerated Figure 3 comparison makes this difference visible.

An intersection requires an identification of the two underlying coordinate spaces. The result above uses exactly the natural first-block/last-block norm identification in the paper's proof; an arbitrary algebra isomorphism could give a different transported intersection.

### Corollary 6 and Theorem 7: wrong sign and a zero convention

For the tested dimension-4 and dimension-8 composition algebras, let Z0 count norm-zero elements **including zero**. The verified formula is

`Z0 = p^(d-1)+p^(d/2)-p^(d/2-1)`.

The paper prints a plus sign before the last term. Its nonzero-zero-divisor definition requires `Z0-1`, and the composition-algebra unit count is `p^d-Z0`. For p=3,d=8:

```
Z0                       = 2241
nonzero zero divisors    = 2240
units                   = 4320
zero + nonzero ZDs + units = 1+2240+4320 = 6561.
```

Thus neither including nor excluding zero rescues the printed value 2295. The unit formula inherits the sign error and gives 4266 instead of 4320. The prose preceding Corollary 6 also gives the same cubic zero-divisor count for quaternions and octonions; complete enumeration distinguishes their dimensions immediately.

The proof's “every element is a zero divisor or unit” must account for zero even in composition dimensions. Its higher-dimensional problem is more substantial, as the next example shows.

### Beyond octonions: units and zero divisors can overlap

In dimension 16 over F3, set

`a=e1+e10`, `b=e4-e15`.

The actual standard Cayley–Dickson products give

`a*a=1`, `a*b=b*a=0`, with `a!=0` and `b!=0`.

So a is its own two-sided inverse **and** is a zero divisor. Its norm is 2, not zero. A verified four-dimensional annihilator basis is saved. Analogous inverse/annihilator certificates are checked over F5, F7 and F11.

There is no contradiction because associativity fails: `(a*a)*b=b`, whereas `a*(a*b)=0`. Under the paper's definition using a left or right inverse, a certainly qualifies as a unit; it even satisfies both inverse equations.

Consequently, subtracting a zero-divisor count from the total cannot be used to count all higher-dimensional units. The repaired composition-dimension formulas above are not claimed as a repair of all sedenion and higher counts. A complete higher-dimensional unit/zero-divisor classification is not performed by this package.

## 4. Causal language in the conclusion

Norm zero is a well-defined finite-field condition. Positive and negative norm are not intrinsic ordered-field conditions over Fp: the representatives 1 and `1-p` name the same residue but have opposite integer signs. No finite field admits an ordering compatible with field addition and multiplication.

A real split algebra is different: its norm has positive, negative and isotropic examples, all illustrated by the tests. Finite-field enumeration alone does not assign real time-/space-/light-like categories or establish a physical application.

## Execution and reproducibility

All 23 local CPU tests pass. Several passing tests intentionally reproduce printed discrepancies. Four actual `multiply()` tests are provided for the user's CUDA machine and run automatically under normal pytest when the prerequisites are present. They were not selected or executed locally.

The [results](results/results.json) and [certificates](results/certificates.json) record the exact model, field, source hashes, complete enumeration totals, explicit algebra maps, kernel bases and counterexamples. The [CSV](results/figure-data.csv) keeps printed and corrected numerical series separate. Original MATLAB programs and the external cited literature were not independently audited.
