# Findings: 0704.0112v3

See [README.md](README.md) for all 32 coded tests, their order in the paper, reproduction commands, and uncoded claims. This file distinguishes successful finite verification from discrepancies and proposed repairs. All indices below are mathematical, zero-based indices; direct calls to `basis()` require labels one greater.

## The main finite structure is reproducible

The paper's Cayley–Dickson indexing agrees with CayleyKernel without a basis permutation. The independent recursive sign implementation agrees at every basis pair through dimension 256. The ET cell decision is then independently checked against full sparse products in both orders for all strut constants through dimension 64.

The finite central, corner, shutter and label-line recursions of Theorems 11–13 agree with exact products in the specified complete table transitions, including the signed marks. The singleton-high-bit rule and its recursive harmonic targets also agree in the tested ranges.

The S=15 balloon ride gives:

| N | Algebra dimension | Filled directed cells | Box-Kites |
| --- | --- | --- | --- |
| 5 | 32 | 72 | 3 |
| 6 | 64 | 456 | 19 |
| 7 | 128 | 2,184 | 91 |
| 8 | 256 | 9,480 | 395 |
| 9 | 512 | 39,432 | 1,643 |
| 10 | 1,024 | 160,776 | 6,699 |

The table counts were computed from products, not populated by the closed formula. Selected levels also have an explicit edge partition into enumerated XOR-sail Box-Kites. The [new table diagram](results/balloon-ride.svg) shows the first three stages.

For the worked hide/fill example `S=57,N=7`, exact enumeration supplies the count the paper declines to calculate by hand: **63 Box-Kites**. All 756 undirected edges are covered exactly once; all 63 six-vertex index sets are saved in [certificates.json](results/certificates.json). The corresponding 1,512 filled directed cells sit among 3,720 admissible off-diagonal positions. The ET itself has `62*62=3,844` positions, of which 124 are on the two forbidden diagonals.

## A real octonion subalgebra inside a hidden configuration

The discussion on p. 22 says hidden configurations can house zero-divisor-free octonion copies. We verify four concrete examples at `S=9,N=5`, based on low triples

`(2,4,6), (2,5,7), (3,4,7), (3,5,6)`.

For example, the first subalgebra has index set

`{0,2,4,6,25,27,29,31}`.

The data give a signed basis map from the standard octonions; every one of its 64 multiplication-table entries is checked. Independently, all coefficients in the real norm-composition identity are checked. This establishes a genuine eight-dimensional unital copy with no nonzero zero divisors for arbitrary real coefficients, rather than simply finding eight basis units that multiply nontrivially. The verification concerns these four examples, not the proposed general “bramble” structure or the speculative physical applications.

## Cookbook preparation needs a qualification at multiples of 16

Preparation on p. 25 says to right-shift `S` three times, or four times if `S` is a multiple of 8, before counting high bits. The recipe then alternates fill/hide rules and uses the parity of that count for cells untouched by a rule.

With that preparation and ordinary modular residues, the recipe agrees for every valid cell at levels `N=4,5`. A complete audit through `N=8` finds failures at multiples of 16. Power-of-two values are useful controls but are excluded by the final proposition; the relevant non-power-of-two failures include **48, 80, 96 and 112**.

The smallest non-power-of-two example is `S=48,N=7`. Binary `48=0b110000`. Shifting four times gives two high bits, 32 and 16. The two rules do not touch `r=1,c=2,P=3`; even parity therefore fills this cell. Actual products give

```
(e1 + e113)(e2 - e114) =  2 e3
(e1 + e113)(e2 + e114) = -2 e115.
```

Neither slope makes zero, so the actual cell is blank. Both nonzero products are saved explicitly. There are 3,360 mismatching positions in this 62-by-62 table. For comparison, there are 360 genuinely filled directed cells, or 15 Box-Kites, while the literal rule paints every off-diagonal admissible cell.

The text uses a special `mod*` notation, described only informally as handling multiples of 8. This implementation reports its ordinary-residue interpretation explicitly. It does not pretend that the notation was fully formalized by the paper or that the discrepancy disproves every conceivable interpretation. The high-bit parity at the displayed S=48 cell makes the printed preparation itself the central issue.

### A proposed correction, separately tested

Instead of always dropping exactly four low bits for multiples of 8, exclude the actual least significant set bit when it occurs at 8 or above. In exponent notation, ignore all set bits below

`cutoff = max(3, v2(S)+1)`,

where `v2(S)` is the number of trailing zero bits. Then keep the other set-bit exponents in descending order. The unchanged first-match fill/hide recipe uses the usual residues `S mod 2**gamma`.

For odd S and for S with low set bits, this gives the printed preparation. For `S=24` it drops bit 8, as the paper intends. For `S=48` it drops bit 16 and keeps only bit 32, making the correct singleton rule. Powers of two have no retained bits and yield a full non-diagonal table, as the earlier full-table theorem requires.

This is **our proposed correction**, implemented as an explicit separate option, not silently substituted for the printed instructions. The complete audit checks every valid ordered cell for every strut constant through dimension 256:

| N | Strut constants checked | Ordered cells, including diagonals | Corrected mismatches |
| --- | --- | --- | --- |
| 4 | 7 | 252 | 0 |
| 5 | 15 | 2,940 | 0 |
| 6 | 31 | 27,900 | 0 |
| 7 | 63 | 242,172 | 0 |
| 8 | 127 | 2,016,252 | 0 |
| Total | 243 level/constant cases | 2,289,516 | 0 |

There are also 9,000 deterministic sampled queries, including the paper's S=613 example and a 201-bit S at formal level N=203. This does not prove the corrected rule for every integer or an infinite-dimensional limit.

## The band-boundary corollary is too broad if read literally

The corollary following Theorem 16 says multiples of 8 that are not powers of two belong in the spectral band of the preceding integers. At dimension 128, products give:

- `S=47`: 1,896 filled directed cells, 79 Box-Kites;
- `S=48`: 360 filled directed cells, 15 Box-Kites.

Thus their counts differ at a multiple-of-16 endpoint. The worked dimension-64 bands still reproduce exactly, including the endpoint S=24. The preparation qualification explains why extending that endpoint convention indiscriminately is problematic.

## Theorem 15's off-switch needs a label-range restriction

The proof on p. 21 changes an upper index by **adding** a new bit L. The correct general assessor formula changes it by XOR with L. Addition agrees with XOR when that bit was absent, as in old labels strictly below L; it does not agree for arbitrary labels from the already larger ambient algebra.

At `N=5`, the cell `r=2,c=8` remains filled when S changes from 1 to 9:

```
S=1: (e2 + e19)(e8 - e25) = 0
S=9: (e2 + e27)(e8 - e17) = 0.
```

The reversed products are also zero. This is an exact counterexample to the unrestricted statement that all previous DMZ cells switch off. All old cells in the specifically tested range `r,c<L` do switch off. The subsequent Hat-Trick restoration is tested on such low-label families, including non-adjacent high-bit choices.

## A units issue in the Whorfian count equation

On p. 17, omega and delta are introduced as **cell counts**, each carrying a factor of 24. The next displayed line equates the Box-Kite count with `omega+delta`, but the polynomial and the numerical sequence are actually

`Box-Kites = (omega_cells + delta_cells)/24`.

The code counts the near-solid-line and remaining-cell contributions independently, confirms both factors of 24, and confirms the final Box-Kite polynomial. This is a units/notation issue; the printed numerical sequence itself is correct in the tested valid levels.

## Fractal, scale-free and orthogonality claims need definitions beyond the tables

The finite masks recur as shown. That is not by itself a proof that a particular limiting metric space is a classical fractal, or that a specified graph has a power-law degree distribution.

For the assessor adjacency graph with S=15, the actual degree histograms are:

| N | Degree : number of vertices |
| --- | --- |
| 5 | 4 : 12; 12 : 2 |
| 6 | 12 : 24; 28 : 6 |
| 7 | 28 : 48; 60 : 14 |
| 8 | 60 : 96; 124 : 30 |

These two-degree finite graphs do not establish the scale-free claim for any proposed infinite or alternative network. The closed formula gives a filled-area fraction tending to 5/8; that density calculation is separate from a fractal-dimension or power-law argument.

The final proposition's “one mutually orthogonal line per cell” also needs an ambient-space interpretation. A filled cell represents a **relationship between two existing assessor lines**. A single row line participates in four cells already in a sedenion ET. The 24 filled directed cells cannot simply be treated as 24 distinct orthogonal directions inside the original 16-dimensional algebra. An explicitly defined larger product/tensor space might encode independent relationships, but the code does not invent that construction or claim to verify it.

The concluding discussions of chaos, Wolfram Class 4 behavior, neuroscience, semantic architectures and adaptive parsers specify no executable dynamics or benchmark. The finite trijoin relation given on p. 29 is implemented and checked; the broader analogies remain uncoded rather than being presented as verified applications.

## CUDA execution is reserved for the user

All `multiply()` calls are confined to the GPU class, which runs automatically under pytest when CUDA is available. The recorded CPU-only report excludes that class and performs no CUDA-device probe. The tests compare full outputs, including nonzero control cases, with the independent recursive oracle. See the commands in [README.md](README.md) to run them on your CUDA machine and preserve a separate report.
