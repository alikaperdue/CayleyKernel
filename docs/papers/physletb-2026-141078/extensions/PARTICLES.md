# Explicit particle-block reconstruction from reference [7]

This calculation extends the main paper's Section 9.3. It uses [arXiv:2505.07923v4](https://arxiv.org/abs/2505.07923v4), the main paper's reference [7], to reconstruct internal irreducible representations. It is part of the [separate follow-up suite](README.md), not a relabeling of the original review's dimension-only checks.

## A source discrepancy must be handled first

On PDF page 6, Eq. (26) prints the third color generator in the form

```text
A = alpha2† alpha3† vO alpha3 alpha2
B = alpha1† alpha3† vO alpha3 alpha1
printed:   i Lambda3 = -i A - i B
```

With the paper's ladder operators, A and B are orthogonal rank-one projections. Each has trace one. Thus the printed operator has **trace -2i**, whereas an su(3) generator in the stated three-dimensional color block must be traceless.

The exact implementation reproduces this nonzero trace. Together with the other printed generators and their commutators, it spans nine dimensions in that block, rather than the claimed eight-dimensional su(3).

A minimal correction is

```text
repaired:  i Lambda3 = -i A + i B.
```

This repaired operator is traceless. The repaired eight generators close as an eight-dimensional algebra and agree in span with the **independently supplied color generators in Eq. (8) of the main Physics Letters B paper**, projected to the relevant complex color block. Thus the repair is supported by a second construction, not merely selected to make a trace test pass.

The literal printed formula and the repaired formula are both retained in [particle_reconstruction.py](../../../../tests/papers/physletb_141078/particle_reconstruction.py). `test_08` passes only when the literal discrepancy is reproduced and the separate repair succeeds. Subsequent color-irrep tests explicitly use the repaired set. No claim is made that the literal Eq. (26) passed.

This is an apparent typographical sign error in reference [7] v4. It does not invalidate the original review's successful test of the main paper's different Eq. (8), and no author confirmation or correction has been assumed.

## Construction rather than assigned labels

The calculation begins with octonion left and right multiplication matrices assembled from CayleyKernel basis products. The paper-to-kernel Fano convention is the same one verified in the original companion.

From reference [7] it constructs:

- the three octonion annihilation operators and their adjoints, Eq. (24);
- the octonion vacuum vO, Eq. (25);
- the two complex volume projectors, Eq. (23), and the quaternionic-volume refinement, Eq. (37);
- all five orthogonal projectors, Eq. (38), with internal complex ranks **1,3,2,1,1**;
- the repaired color generators, Eq. (26), weak generators, Eq. (43), and Y, Eq. (44);
- the fourth ladder operator beta4, Eq. (30), in a faithful 2-by-2 complex quaternion representation tensored with the octonion operators;
- the full rank-one vacuum and four left/right beta excitations for every off-diagonal particle family on pages 10-12.

All four annihilation operators obey the exact canonical anticommutation relations on the 16-complex-dimensional representation. The vacuum is Hermitian, idempotent and rank one. The projectors are Hermitian, mutually orthogonal, and sum to the identity. Including the quaternion factor, their ranks become **2,6,4,2,2**.

The tensor operations used here are those of C tensor H tensor O and its multiplication operators. They are not replaced by a single 32-dimensional Cayley-Dickson product; those algebras have different products.

## Actual restricted generator matrices

For an operator X mapping block j into block i, the infinitesimal internal action is

```text
delta X = G_i X - X G_j.
```

For each directed block `i<j`, the code constructs an orthonormal basis from the actual projectors and computes this action on its basis matrices. It verifies that the restricted action intertwines with the original 8-by-8 octonion operator commutator.

It then independently checks:

1. Y is the claimed scalar on the entire module.
2. The normalized color quadratic Casimir is 4/3 for triplets and zero for singlets.
3. The normalized weak quadratic Casimir is 3/4 for doublets and zero for singlets.
4. The color Cartan cubic trace changes sign between 3 and 3bar, distinguishing conjugate triplets that the quadratic Casimir alone cannot distinguish.
5. The complex commutant of all color and weak generator matrices has dimension **one**. Since these representations are unitary, this certifies complex irreducibility of the internal module.
6. `Q=Y+T3` has the electric-charge eigenvalues listed below.

The expected labels were transcribed independently from the reference's particle tables. Matrix calculations are compared against those labels; the labels are not inferred from dimensions alone.

## Reconstructed internal irreps

A row describes one internal complex irrep. The full corresponding directed block contains four multiplicity copies supplied by the 2-by-2 quaternion matrix factor. Hermitian conjugation supplies the reverse block with conjugate color representation and opposite charges.

| Block | Reference assignment | SU(3), SU(2), Y | Electric charges in one internal copy | Internal complex dimension |
| --- | --- | --- | --- | ---: |
| H12 | Anti-bottom sector | (3bar,1,1/3) | 1/3, three colors | 3 |
| H13 | Anti-lepton doublet | (1,2,1/2) | 0, 1 | 2 |
| H14 | Sterile anti-neutrino | (1,1,0) | 0 | 1 |
| H15 | Charged anti-lepton | (1,1,1) | 1 | 1 |
| H23 | Left quark doublet | (3,2,1/6) | 2/3 and -1/3, three colors each | 6 |
| H24 | Right down-type quark | (3,1,-1/3) | -1/3, three colors | 3 |
| H25 | Right up-type quark | (3,1,2/3) | 2/3, three colors | 3 |
| H34 | Higgs times (anti)-neutrino interpretation | (1,2,-1/2) | 0, -1 | 2 |
| H35 | Anti-tau doublet | (1,2,1/2) | 0, 1 | 2 |
| H45 | Anti-tau singlet | (1,1,1) | 1 | 1 |

These are the reference's directed-block conventions, including its use of anti-particle labels in some blocks. The equivalent SU(2) conjugate doublet is also two-dimensional. The calculation does not choose an interpretation for H34 beyond checking its stated internal representation.

### Explicit operator bases and multiplicities

For example, the internal H23 basis is generated by

```text
alpha_i† alpha_j† vO alpha_a,
(i,j) in {(1,2),(2,3),(3,1)},  a in {1,2}.
```

It has six independent elements. The four beta left/right excitations produce 24 independent complex operators in the full directed block; adding their Hermitian conjugates gives 48 real parameters. This agrees with the reference's H23 count.

The same procedure is implemented for every off-diagonal family, not just H23. Their total is **192 real dimensions** after Hermitian conjugation. The six outer families account for 128 real dimensions; the four inner families H12,H34,H35,H45 account for 64.

On the diagonal, exact adjoint-action Casimirs give

```text
M3(C): singlet + color adjoint 8,  Casimir eigenvalues 0 and 3
M2(C): singlet + weak adjoint 3,   Casimir eigenvalues 0 and 2.
```

Together with the smaller diagonal blocks and quaternion factor, the full diagonal Hermitian sector has **64 real dimensions**. Therefore the operator-space count is

```text
64 diagonal + 128 outer off-diagonal + 64 inner off-diagonal = 256 real.
```

This reproduces the algebraic partition in reference [7]. It does not by itself derive a physical generation or spin assignment for each multiplicity copy.

### Charge conventions versus the main paper

Reference [7] uses internal block weights

```text
(1/2, 1/6, 0, 1/2, -1/2).
```

The simplified main-paper Y has weights `(0,1/3,1/2,1,0)` in its own block ordering. After exchanging the last two singlet labels and using the conjugate charge convention, the relation is exactly

```text
Y_reference = (1/2) I - Y_simple_reordered.
```

The scalar shift cancels in commutators; the overall minus reverses the charge convention. The test checks this relation explicitly rather than equating differently labeled matrices entry by entry. In particular, the agreement of physical charge patterns requires attention to particle/anti-particle orientation.

## What this adds beyond the initial review

The original companion calculated block sizes and relative charge spectra. This follow-up additionally supplies **explicit operator bases and generator matrices**, verifies their actual action, distinguishes conjugate color triplets, and certifies internal irreducibility. It therefore replaces an earlier untested irrep interpretation with a reproducible reconstruction, subject to the documented source repair.

It also locates the original model's limitation precisely: among the inner off-diagonal blocks there is no `(3,2,1/6)` top-containing doublet or `(3,1,2/3)` up-type singlet. Those occur only in the outer blocks H23 and H25. This reproduces the reference's stated absence of the top-containing internal representations in the proposed third-generation sector. It does not establish that a top quark is composite.

The following remain outside this calculation:

- Lorentzian dynamics, Weyl spin statistics and an independently justified identification of every multiplicity copy with a physical generation;
- a Higgs potential, mass spectrum, interactions or anomaly cancellation;
- experimental evidence for the model;
- a demonstration that the reference's particle labels are the only possible interpretation of these internal representations.

## Inspect or reuse the matrices

The [saved JSON](results/results.json) contains:

- `experiments.explicit_operators`: the projectors, octonion annihilation matrices, beta4, orthonormal internal states and every internal ladder-operator basis;
- `experiments.particle_irreps`: the exact restricted color, weak, Y and Q matrices for each block, labels, commutant dimension and electric-charge multiplicities;
- `experiments.reference_discrepancy`: both the literal trace failure and the separately verified correction;
- `experiments.explicit_ladder_basis_counts` and `particle_scope`: the complete degree-of-freedom counts and limits.

Matrices are stored as `shape` plus nonzero `[row,column,expression]` entries. Expressions are exact SymPy strings, including I and square roots; omitted entries are zero. Generator order is Lambda1 through Lambda8 divided by two, then tau1 through tau3 divided by two. Color and weak matrices are anti-Hermitian. Y and Q are Hermitian charge-action matrices; multiply them by i when forming anti-Hermitian infinitesimal phase generators.

The [source module](../../../../tests/papers/physletb_141078/particle_reconstruction.py) also returns these matrices directly through `model()` and `restricted_module(i,j)`. The full 16-by-16 beta-excited particle bases can be regenerated from `model()['full_ladder_basis']`.
