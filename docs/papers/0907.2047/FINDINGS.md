# Findings for the supplied 0907.2047v3 PDF

The paper's central finite classification is reproduced. The [test inventory](README.md), [saved run](results/results.json) and [complete lattice with certificates](results/subloops.json) give the checks and data. The main discrepancy concerns exchanged table images/captions, rather than a failure of the subloop counts.

## 1. Main finite claims agree

The exact computation gives the following proper nontrivial subloops:

| Order | Count | Isomorphism classes |
| --- | --- | --- |
| 32 | 31 | S:16, S_alpha:7, S_beta:7, S_gamma:1 |
| 16 | 155 | O:50, O_quasi:105 |
| 8 | 155 | Q8:155 |
| 4 | 31 | C4:31 |
| 2 | 1 | C2:1 |

The 31 maximal members have the claimed maximal octonion-type compositions: (8 ordinary,7 quasi) occurs 17 times; (2,13) occurs seven times; (0,15) occurs seven times. The four representative rows of Table 6 agree. All 155 octonion-type members have seven Q8, seven C4 and one C2 proper nontrivial subloops.

The signed basis multiplication agrees with independent recursive doubling on every pair and sign combination. Normality is checked through all 8,256 standard inner generators; every such map preserves each absolute basis index. Each signed-support subloop is therefore invariant. All seven identities in Table 7 pass on their full finite domains.

Class counts are not inferred from matching one weak statistic. Distinct representative invariants establish separation, and every class membership has an explicit bijection whose entire signed multiplication table is checked directly against `basis()`.

## 2. Tables 3 and 4 have exchanged image content relative to captions

On PDF page 5:

| Table | Header actually inside the image | Class described by caption |
| --- | --- | --- |
| 3 | 0,1,2,3,12,13,14,15,20,21,22,23,24,25,26,27 | S_alpha, whose listed support uses 8,9,10,11,28,29,30,31 instead |
| 4 | 0,1,2,3,8,9,10,11,20,21,22,23,28,29,30,31 | S_beta, whose listed support uses 12,13,14,15,24,25,26,27 instead |

Thus image 3 is the beta representative and image 4 is the alpha representative, as defined in Section 3.1. The captioned support and the identity row within each image cannot both describe the same labelled table.

All 256 entries in each image agree with CayleyKernel when interpreted using that image's actual header. The tests retain the literal tables and separately reproduce the caption exchange. They do not silently rewrite the image to fit the caption or count the inconsistent labels as verified.

Table 2 and Table 5 also pass full 256-entry literal checks. Four rows of the low-resolution Table 1 image pass a 128-entry check. The other Table 1 image cells are not claimed to have been individually transcribed; the complete mathematical product is independently checked against doubling.

A smaller notation issue occurs in Section 3.1's fourth bullet: the superscript is delta while its description and subsequent class list/Table 5 use gamma. The explicitly printed support identifies the intended representative without ambiguity. The code consistently calls it S_gamma.

## 3. Why S and S_gamma are different despite the same composition

Both have eight ordinary and seven quasi-octonion maximal subloops. That count alone cannot distinguish them. The following exact invariant does:

For a fixed positive basis representative a, count positive basis pairs (b,c) with `(ab)c != a(bc)`. Record the distribution of these counts across all positive a in the support. A signed-loop isomorphism can permute a and change its sign, but cannot change this profile: central input signs cancel from associativity.

| Representative | Associator-failure profile: failure count : number of a values | Positive-triple Moufang failures |
| --- | --- | --- |
| S | 0:1, 120:14, 168:1 | 672 |
| S_alpha | 0:1, 72:1, 84:8, 88:6 | 1248 |
| S_beta | 0:1, 56:3, 68:3, 72:4, 84:5 | 1440 |
| S_gamma | 0:1, 84:1, 120:7, 132:7 | 672 |

These profiles are pairwise different. They give a direct independently computed reason for four signed-loop classes, including the S/S_gamma distinction. The explicit representative-to-member maps then establish the stated multiplicity of each class.

The octonion profiles differ too: ordinary O has 0:1,24:7 and no Moufang failures; O_quasi has 0:1,8:3,12:4 and 96 positive-triple Moufang failures. A smaller number of failed associativity instances does not imply Moufang behavior; these are different properties.

## 4. Exact algebra checks support the loop/algebra distinction

All signed basis elements of the full 64-element loop satisfy alternative and inverse laws. Arbitrary sums in the 32-dimensional algebra do not.

One recorded pair, in mathematical zero-based basis coordinates, is

```text
a = e1+e10,
b = e4-e15.
```

Both are nonzero and `ab=0`. The quadratic relation gives `a²=-2`, so

```text
a*(a*b)=0,    (a*a)*b=-2*b != 0.
```

This demonstrates why basis-loop alternativity does not imply algebra alternativity. Inverse-property and C-loop equations also fail on the same kind of sums. With `q=1+b`, products `a*q` and `q*a` equal a and are nonzero, yet the algebra's anti-automorphic and weak inverse equations still fail. Thus these failures do not depend on attempting to invert zero.

In contrast, exact symbolic coefficient expansions certify flexibility and the quadratic relation on the entire real 32-dimensional algebra. No sampled tolerance is involved. The quadratic relation supplies the usual single-element power-associativity argument, with seeded degree-three/four products as additional checks.

For every one of the 50 ordinary octonion coordinate spans, all coefficients of `N(ab)-N(a)N(b)` vanish. This is an all-real polynomial certificate within the stated spans. Every quasi-octonion span has a nonzero defect and a saved two-term zero-divisor pair. All 31 sedenion-type coordinate spans also have saved two-term zero-divisor pairs, checked independently.

## 5. Complete lattice constructed beyond Figure 1

Every nontrivial subloop contains -1 and both signs of each included basis unit. Its unsigned indices form a subspace of F2^5 under XOR. Conversely every such subspace lifts to a signed subloop. This gives a compact exhaustive construction rather than searching all subsets of a 64-element set.

There are 374 such subspaces, including {0} and the full five-dimensional space. Add the singleton loop {+1} to obtain 375 total subloops. The paper's 373 are the proper nontrivial ones, excluding the singleton and the ambient loop. This counting distinction is not a contradiction in the paper.

The exported complete lattice has:

- 375 nodes, all with explicit supports;
- 2,078 immediate inclusions;
- exact intersection and generated-span operations;
- 2,304,000 verified modular-law instances, exhausting every A<=C and every B.

The class-level cover counts are:

| Upper class | Lower class | Number of actual cover edges |
| --- | --- | --- |
| T | S / S_alpha / S_beta / S_gamma | 16 / 7 / 7 / 1 |
| S | O / O_quasi | 128 / 112 |
| S_alpha | O / O_quasi | 14 / 91 |
| S_beta | O_quasi | 105 |
| S_gamma | O / O_quasi | 8 / 7 |
| O | Q8 | 350 |
| O_quasi | Q8 | 735 |
| Q8 | C4 | 465 |
| C4 | C2 | 31 |
| C2 | singleton | 1 |

The existence of these block edges agrees with Figure 1. The paper explicitly says that determining the complete lattice remains work in progress; the JSON here supplies it for this fixed signed basis loop.

### Aggregating by isomorphism class loses the lattice operation

Figure 1 is a valid block-form description of containments, but the phrase “lattice of isomorphy classes” should be read informally. The graph of classes alone is not a lattice in the strict order-theoretic sense. For example, S and S_gamma have two incomparable maximal common lower classes, O and O_quasi. Neither is a greatest lower class.

Individual concrete subloops still have well-defined intersection and join, and their full lattice is modular. The block graph merges different subloops and forgets which particular intersections occur. The suite checks this distinction rather than using a class graph to stand in for the actual lattice.

## 6. Scope of “basic subalgebra” and algebra isomorphism

This study exhausts the coordinate subalgebras spanned by subsets of the selected 32 basis elements. It does not enumerate every real subalgebra of T.

For example, let v=e1+e2. Then `v²=-2`, so `span_R{1,v}` is a two-dimensional closed real subalgebra. It is not the coordinate span of a selected pair of basis units. Its union of coordinate supports has three positions, but the subalgebra itself has dimension two. The test includes an explicit product inside this span.

Consequently the final statement about 16-dimensional subalgebras is verified here for all **31 basic coordinate** cases, not for every possible real 16-dimensional subalgebra. No counterexample outside that scope is claimed.

A signed-loop isomorphism extends by bilinearity to an algebra isomorphism of the corresponding spans; those extensions are certified. The reverse implication is not automatic: arbitrary real-linear algebra isomorphisms could mix basis directions. The signed-loop invariants distinguish signed loops, not all possible real algebra tensors under GL(16,R). No full real-linear algebra-isomorphism classification is asserted by this implementation.

## 7. What remains outside the result

No independent literature-priority review, physics model, GAP/FINITAS software audit, general automorphism-group classification, or classification of all dense zero divisors is included. Our deterministic node IDs are not the original FINITAS enumeration IDs.

Three GPU replays call `multiply()` on the full signed product table, all 373 dense subalgebra examples and all 136 zero-divisor witnesses when prerequisites are available. CuPy is missing here, so they are explicitly skipped. CPU `twist()` and `basis()` checks were performed; GPU execution and timing were not.
