# Findings, interpretation and higher-level experiments

## The paper's algebraic route is reproduced

The subsequent [GPU run](results/gpu-2026-10-29.json) has **31 passes, no skips or failures**, completing in 25.05 seconds with the Rust CPU backend. Both GPU checks passed, including full arrays through level 20 (1,048,576 components). The original [CPU-environment run](results/results.json), with 29 passes and two GPU skips, is retained as historical evidence. The [coverage inventory](README.md) lists all 31 methods in paper order and separates finite experiments from symbolic identities and unimplemented physical proposals.

The initial multiplication algebras have real dimensions:

| Base | Left | Right | Full | Left/right intersection |
| --- | ---: | ---: | ---: | ---: |
| R | 1 | 1 | 1 | 1 |
| C | 2 | 2 | 2 | 2 |
| H | 4 | 4 | 16 | 1 |
| O | 64 | 64 | 64 | 64 |

These numbers explain the special role of H in the paper. Its left and right actions each give a small operator algebra, and both are needed to make the full 16-dimensional operator algebra. For O, either side's multiplication sequences already span every 8-by-8 real matrix. An algebra's elements and its multiplication operators are different objects.

The exact commutation-constraint solver then obtains the claimed real centralizer dimensions: H with one volume gives 8; O with one gives 32; O with both gives 20. Restricting H by a second volume gives 4. Taking skew parts and imposing the paper's annihilation and trace conditions leaves 12 or 9 dimensions, with commutator spans of dimension 11 or 8. The remaining single phase is precisely the charge generator constructed from the projectors.

No failing printed algebraic identity was found within this suite's stated coverage. This is not a review of all referenced papers or proof that the proposed model accounts for the Standard Model.

## Charges are relative operator weights

The extended 16-real-dimensional representation has these exact spectra:

| Operator | Eigenvalue : real multiplicity |
| --- | --- |
| Y | 0:4, 1/3:6, 1/2:4, 1:2 |
| Q | 0:6, 1/3:6, 1:4 |

Both real traces are 6; the corresponding complex trace is 3 on complex dimension 8. Thus the trace divided by dimension is 3/8 in either convention, as printed. Subtracting 3/8 times the identity leaves every matrix commutator unchanged. Also, Q-Y is supported in the quaternion block and has the weak-generator eigenvalues +/-1/2.

On a map from block j to block i, the commutator charge is `weight_i - weight_j`. The exact complex matrix-unit spectra are:

| Absolute nonzero charge | Y multiplicity at each sign | Q multiplicity at each sign |
| --- | ---: | ---: |
| 1/6 | 6 | 0 |
| 1/3 | 6 | 9 |
| 1/2 | 6 | 0 |
| 2/3 | 3 | 6 |
| 1 | 2 | 6 |
| Zero charge | 18 total | 22 total |

Each column totals 64 complex matrix units after including both signs. This is a reproducible algebraic spectrum, not an assignment of all those entries to physical particles. Reference [7]'s full spin, chirality and generation interpretation was not reconstructed from the diagram alone.

The inverse-dimension form in Eq. (43) means each nonzero block separately is a maximally mixed density matrix on that block. Its trace is one and entropy is log(n). Their direct sum has complex trace three; it is not itself a normalized density matrix on the entire eight-dimensional space.

## What survives beyond the sedenions

Two separate experiments distinguish keeping the paper's construction as a small embedded model from replacing its octonions by a whole larger algebra.

1. **Keep the small model.** Low-index Cayley-Dickson embeddings preserve the multiplication of the existing 16-dimensional subalgebra. All 256 of its basis pairs were checked at levels 4,8,16,20,32,64,128,256,1024. At those same levels, 64 random high-bit pairs per level agree with an independent iterative doubling oracle. These are exact sparse/index checks. They do not construct full higher-dimensional operator spaces.
2. **Use new high-bit generators.** At levels 4,8,20,64,128,1024, 32 sampled pairs of distinct imaginary basis units each generated a quaternion table. All 64 basis associators vanish in each constructed four-dimensional span. Trilinearity extends this to arbitrary real coefficients in each tested span. This finds associative islands far beyond the division algebras; it does not make the ambient algebra associative.
3. **Enlarge the projector construction.** For the doubling unit `u=E_(2^(n-1))`, every column of `L_u R_u` was evaluated through level 12 (dimension 4096), without allocating a dense operator matrix. The projectors `(I-L_u R_u)/2` and its complement have real ranks 2 and `2^n-2`.

For commuting complex structures with these ranks, elementary complex block linear algebra predicts a joint-centralizer real dimension of

```text
2 + (d-2)^2/2,  d = 2^n.
```

At d=8 this is 20, consistent with C+M3(C). At d=16 it is 100, consistent with C+M7(C). The suite independently solves all the d=16 commutation equations for binary unit E15 and obtains 100. The sampled projector family uses the doubling unit E8 at that level; this separate solve also checks the dimension at a different basis unit. Larger centralizer dimensions in the JSON are **predictions from the block formula**, not results of enormous full-matrix elimination.

This exposes a useful boundary: extending the ambient algebra does not automatically preserve the same color-sector size. Keeping the embedded octonion construction retains its small block structure; applying the full centralizer construction to a larger space increases the available matrix block. No new physical interpretation is asserted.

## Why the unchanged Hopf formula stops at O

The paper restricts Eq. (55) to A=R,C,H,O. The symbolic tests confirm norm composition in all four algebras. It gives the homogeneous identity

```text
(N(a)-N(b))^2 + N(2*a*conjugate(b)) = (N(a)+N(b))^2.
```

For normalized input, the right side is one, so the output lies on a sphere. The north-pole differential is `(0,2*conjugate(b))`, with full rank dim(A).

The following exact counterexample uses **kernel zero-based coordinates**, not the paper's Fano labels:

```text
a = (E1 + E10)/2
b = (-E4 + E15)/2
N(a) = N(b) = 1/2
```

The search finds `(E1+E10)*(E4-E15)=0`. Consequently `a*conjugate(b)=0`; the scalar component `N(a)-N(b)` also vanishes. Thus a **norm-one input maps to the zero vector**, whose norm is zero. This disproves the unit-sphere property of the unchanged formula for sedenionic halves. It would concern an attempted map out of the unit sphere in a 32-dimensional doubled space, not the paper's legitimate octonion Hopf map out of S15.

The low-index witness persists in every larger ordinary Cayley-Dickson algebra. The suite records it at levels 4,5,8,16,20,32,128,1024. Those repeated checks establish admissibility of the embedded witness and reuse its exact multiplication; they are not new independent high-index counterexamples. This limitation comes from loss of norm composition, so faster hardware or a larger GPU cannot remove it.

This result does not disprove every possible higher-level geometric construction. It rules out this particular unmodified sphere-map formula on the full higher algebra.

## What remains open or outside this companion

- The tests do not derive particle masses, a Higgs potential, interactions, experimental predictions, baryon asymmetry or a physical Fock space. The paper does not specify enough of those constructions to implement a verification.
- The e8 paragraph supplies dimensions and a proposed direction, rather than a chosen embedding and complete bracket/charge prescription. Only its dimension bookkeeping was checked.
- Global Hopf-fibration topology and global representation-theoretic identifications cannot be established by the norm and differential checks alone. They would require additional topology or proofs; there is no finite sample count that substitutes for them.
- Other 16-dimensional embedding spaces were tested as real vector spaces with transported actions. This suite does not claim their products are sedenion products. In particular, C tensor O has a central scalar imaginary unit, whereas a sedenion doubling unit has different multiplication rules.
- CPU sparse/index checks completed through level 1024; column-complete projector checks reached level 12. Full-array GPU replay through level 20 **passed in the subsequent GPU run**, using sparse coefficients in batched full-length arrays. The original environment skipped those checks. The 25.05-second suite duration is not an isolated multiplication benchmark or a speedup measurement.

These follow-on investigations are now implemented in a [separate section](extensions/README.md): exact XOR-subspace classification, comparisons of other basis units, and [explicit internal particle representations](extensions/PARTICLES.md) using the acquired reference [7]. The new work has its own tests and saved evidence; the original run and its original scope above are preserved. The reconstruction also exposes a printed sign discrepancy in reference [7], documented separately from the main paper.
