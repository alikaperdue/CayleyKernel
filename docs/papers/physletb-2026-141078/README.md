# Verification of Furey, Physics Letters B 882 (2026) 141078

N. Furey, **Standard model symmetries and the nested embeddings of R ⊂ C ⊂ H ⊂ O**, DOI `10.1016/j.physletb.2026.141078`, ten PDF pages, published 17 October 2026. The reviewed input was saved as `CERN-main.pdf`; obtain the cited article separately to check its provenance hash.

SHA-256: `edb6827bddf686b609d459495ce44e9faa2b8549b31ce55f677b03971583a40c`. The PDF, extracted text, rendered inspection pages, and console logs are not duplicated in Git. Printed and PDF page numbers agree. Pages 3 and 6-9 were also inspected as rendered images, including the convention, charge, embedding and Hopf diagrams.

## Result and what it means

The subsequent [GPU run](results/gpu-2026-10-29.json), supplied by the user on 29 October 2026, contains **31 tests: 31 passed, no skips, failures or errors**, in 25.05 seconds. It used the Rust CPU backend and successfully executed both GPU tests, including batched full arrays through level 20. The original [CPU-environment run](results/results.json) is preserved separately: 29 passed and two GPU skips because CUDA reported no device. CPU experiments use exact integers, rational numbers, symbolic polynomials and rational matrix elimination. No counterexample to a tested printed identity was found. The negative higher-level Hopf result below is an extension beyond the paper's stated division-algebra scope.

In practical terms, the paper starts with four number systems: real numbers, complex numbers, quaternions and octonions. It studies operators that multiply their elements, including sequences of multiplications. Selecting operators that commute with particular rotations leaves a block structure. Two further constraints reduce its skew operators to dimensions **12** and **9**, matching the dimensions of the proposed pre- and post-breaking symmetry Lie algebras. This verifies the stated algebraic construction; it does not establish that nature uses this model.

The tests reproduce the multiplication-algebra dimensions, all seven conversion formulas between right and left octonion multiplication, all eight gluon-generator derivation checks, Clifford anticommutation and volume elements, centralizers, symmetry constraints, charges, embeddings, and the Hopf norm formula. [Findings and extensions](FINDINGS.md) explain the higher-level results and the remaining limits.

## Reproduce

From the repository root, using Python 3.10 or later with NumPy and SymPy:

```bash
python3 -m tests.papers.physletb_141078.run
python3 -m pytest -q tests/papers/physletb_141078
```

Record a **new** evidence file without replacing the historical result:

```bash
python3 -m tests.papers.physletb_141078.run \
  --paper /path/to/CERN-main.pdf \
  --output /tmp/cern-verification-new.json
```

This runner executes `Sections` first in paper order, then `GPUVerification`. Plain `unittest` discovery also works, but sorts the GPU class first. The recorded run used a Python environment with NumPy, SymPy, CuPy and the loaded kernel; repository pytest integration was checked in a separate Python environment. Tests carry the `cpu` marker so that the repository's general GPU fixture does not skip their CPU work. The GPU class checks its own prerequisites; an error or mismatch after those checks succeed is a failure, not a skip.

GPU test 27 runs full arrays through level 20, dimension 1,048,576. It uses sparse coefficients in those arrays; the actual GPU `multiply()` still executes. This can be expensive and is a correctness experiment, not a timing baseline. These GPU products passed in the subsequent GPU run; they did not run in the original no-device environment. Sparse CPU level 1024 means arithmetic on 1024-bit indices, **not** allocation or multiplication of arrays of length 2^1024.

## Files and arithmetic conventions

- [test_sections.py](../../../tests/papers/physletb_141078/test_sections.py): every test listed below.
- [algebra.py](../../../tests/papers/physletb_141078/algebra.py): sparse-to-matrix conversion, exact centralizer and constraint solvers, basis convention, embeddings and Hopf formula.
- [run.py](../../../tests/papers/physletb_141078/run.py): ordered runner with environment, PDF/source hashes, outcomes and experiment data.
- [results.json](results/results.json): compact recorded evidence, including matrix columns, spectra, sample counts and high-level ranges.

Sparse multiplication reuses the existing [Desktop companion adapter](../../../tests/papers/desktop_batch/common.py). It calls `CayleyKernel.basis_product` if available, otherwise the public `CayleyKernel.basis` routine. The current public name is `basis`; the adapter is named `basis_product`. Signed one-based API labels are converted to zero-based sparse coordinate indices. `twist()` is tested separately against an independent iterative implementation of the four Cayley-Dickson doubling cases. Agreement of `basis` with `twist` alone would not be independent evidence.

The paper's octonion labels are converted as follows, where `E` denotes CayleyKernel's zero-based basis:

| Paper | Kernel coordinates |
| --- | --- |
| 1 | E0 |
| e1 | E1 |
| e2 | E2 |
| e3 | E4 |
| e4 | E3 |
| e5 | E6 |
| e6 | -E7 |
| e7 | E5 |

All 64 paper basis products verify the seven oriented Fano triples in Section 3.2. Matrix rows and columns use **kernel coordinate order**, including the octonion source coordinates of the embedding. Quaternion source coordinates are `1, epsilon1, epsilon2, epsilon3`; their embedded images use paper `1,e2,e6,e7`.

Operator composition uses ordinary exact matrix multiplication: `L_a L_b x = a*(b*x)`. Replacing it by `L_(a*b)` would invalidate the paper's construction. This suite does not modify the module or substitute CPU execution for GPU `multiply()`.

## Coverage in paper order

Test names below give the suffix following `test_`. Every coded test, including supplements and skipped GPU tests, appears in this inventory.

### 1. Nature's feat of efficiency (pages 1-2)

`01_section1_clifford_0_8_and_bott_dimensions`: build eight real 16-by-16 gamma matrices from octonion left multiplication. Check all anticommutators and squares. Their 256 products have exact Gram matrix `16 I`, establishing independence and hence the full matrix algebra in this representation. Check tensor-power dimension counts for zero through four factors in Eq. (1).

**Not established:** a Fock space with particle statistics, all real Clifford signatures in Eqs. (1)-(2), or the physical meaning of the component counts. The paper itself leaves (anti-)symmetrization to further work. The numerical coincidence between 256 and a physical counting convention is not an empirical test.

### 2. Connected works (page 2)

Historical and priority statements are not computational tests. Eqs. (3)-(4) are checked at their detailed derivation in Section 8; Eq. (5) is checked at the Hopf construction in Section 9.6. Cited papers are not treated as additional input PDFs or silently added to the review scope.

### 3. Multiplication algebras (pages 3-4)

| Section / statement | Test | Actual check |
| --- | --- | --- |
| 3.1, Eqs. (6)-(7) | `02_section3_1_left_right_maps_and_nonassociativity` | All basis columns in dimensions 1,2,4,8 agree with sparse products. Exact rank of `a -> L_a-R_a` establishes the entire scalar center, including arbitrary linear combinations. Operator composition is associative, while a concrete octonion example rejects `L_a L_b=L_(ab)`. |
| 3.1, Eq. (8) | `03_section3_1_equation8_su3_derivations` | Eight independent real skew matrices annihilate e7 and satisfy the Leibniz rule on all 64 octonion basis pairs. Every pairwise commutator lies in their eight-dimensional span. Nonzero normalization factors, including sqrt(3), are omitted; the real Lie algebra span is unchanged. Normalized structure constants are not asserted. |
| 3.2 conventions | `04_section3_2_fano_table_and_api_oracle` | All 64 Fano-table entries after convention transport. Exhaust all basis pairs at levels 0-6 against independent doubling, `twist`, and all four input-sign choices of `basis_product`. |
| 3.2, Eq. (9) | `05_section3_2_direct_sum_componentwise` | Four mutually orthogonal block identities sum to the 15-dimensional identity; component products agree with multiplication matrices in R,C,H,O. Cross summands have zero product in this direct sum. |
| 3.3, Eqs. (10)-(11),(13) | `06_section3_3_multiplication_algebra_dimensions` | Exact left/right/full dimensions are (1,1,1), (2,2,2), (4,4,16), (64,64,64). Intersections have dimensions 1,2,1,64. Six octonion generators provide all 64 independent operator words; all 16 quaternion left/right tensor products are independent. |
| 3.3, Eq. (12) | `07_section3_3_equation12_all_seven_units` | All seven printed right-in-left formulas, including cyclic index wraparound, and their order-reversed left-in-right counterparts. |

### 4. Clifford algebras and imaginary volume elements (page 4)

`08_section4_clifford_signatures_and_volumes`: every anticommutator and the complete volume product for quaternion left and right Cl(0,2), full quaternion Cl(3,1), and octonion left and right Cl(0,6). Confirm the printed signs of the volume elements and that each squares to -I. Explicit real 2-by-2 generators also verify the optional Cl(2,0) realization of Eq. (15).

### 5. Centralizers (page 4)

`09_section5_centralizers_and_even_words`: solve **all matrix commutation constraints**, not just examples. Real dimensions are 2 for C, 8 for H with one volume, 32 for O with one volume, and 20 for O with both volumes. The one-volume quaternion and octonion centralizers equal the spans of even Clifford words. The trivial R case is identity multiplication. Dimensions, complex structures, and projector ranks agree with Eq. (16)'s C, M2(C), M4(C), and C+M3(C) descriptions. This is a fixed-dimension exact calculation rather than a classification of arbitrary Clifford algebras.

### 6. Electroweak symmetry breaking (page 4)

`10_section6_second_quaternion_centralizer`: imposing both quaternion volume constraints leaves exactly four real dimensions. The matrices I,L3,R3,L3R3 span that solution. Orthogonal rank-two real projectors split it into two complex lines, consistent with C+C. This is the proposed **algebraic restriction**; it does not simulate a dynamical Higgs field or calculate a mass.

### 7. Lie-Jordan splitting (page 5)

`11_section7_lie_jordan_splitting`: exact skew centralizer bases give dimensions 15 and 13 before conditions (32); exhaustive commutators close in those spans. All four bilinear rules of Eq. (27) are checked on complete Hermitian and anti-Hermitian bases of M3(C), which also contains the smaller complex block cases. The real block is immediate. Positive-state dynamics and quantum field theory are not inferred from these identities.

### 8. Extracting symmetries (pages 5-7)

| Subsection | Test | Actual check |
| --- | --- | --- |
| 8.1, Eqs. (30)-(31) | `12_section8_1_xor_grading` | Exhaust basis products through level 7: output grade is input XOR. Explicitly distinguish paper labels from binary grades. Clifford word/parity calculations are also present in tests 01,08,09. |
| 8.2, Eq. (32) | `13_section8_2_full_symmetry_constraints` | Solve the full annihilation and two trace constraints on all skew centralizers. Dimensions fall 15 -> 12 and 13 -> 9. Exhaust commutators; derived subalgebra dimensions are 11 and 8, leaving one abelian phase in each. These match su(3)+su(2)+u(1) and su(3)+u(1). |
| 8.3, Eqs. (34),(36) | `14_section8_3_projectors` | Idempotence, symmetry, orthogonality, complementarity, ranks (2,6,2,2), and action on the distinguished highest elements. |
| 8.4-8.5, Eqs. (37)-(42) | `15_sections8_4_8_5_trace_constraints_and_charges` | Solve phase equations independently with gamma=1; obtain (1/3,1/2,1) and (1/3,1,1). Verify the resulting skew generators belong to the full constrained algebras and all three real sector traces equal -2. Verify Q-Y is a traceless weak generator with real eigenvalues +/-1/2. |
| 8.6(b,c), 8.7, Eqs. (43)-(45) | `16_sections8_6_8_7_scalar_shift_spectra_and_entropy` | Exact real spectra on the extended 16-dimensional space; traces 6 and trace/dimension=3/8. The shifted operators are traceless, and every one of 256 matrix-unit commutators is unchanged by the shift. Each nonzero normalized block has trace one and entropy log(n). The full Y or Q is a sum of three such blocks, not a trace-one density matrix. |
| 8.6(a) | `16b_section8_6_left_right_complex_structures` | L7 and R7 agree on the octonion singlet plane and are negatives on its six-real-dimensional complement; both square to -I. This verifies the orientation change underlying the conjugate complex interpretation. It supplies no baryogenesis mechanism. |

**Not established:** equality to every particle block or operator convention of reference [7], experimental hypercharges, three physical generations, or the suggestion about baryon asymmetry. We calculate the consequences of the model's stated operators, rather than assuming these establish all observed particle assignments.

### 9. Nested Cayley-Dickson models (pages 7-9)

| Subsection | Test | Actual check |
| --- | --- | --- |
| 9.1 | `16c_section9_1_other_vector_space_embeddings` | Explicit orthogonal embeddings for O direct-sum O and C tensor O as real vector spaces, with rank 15 and an extra real direction. Transport the constrained SM actions and verify equivariance. These are vector-space tests; their full algebra multiplications are not conflated with sedenion multiplication. |
| 9.2, Eqs. (46)-(49) | `17_section9_2_embedding_and_nested_subalgebras` | Build every column of the sedenion embedding using actual basis products, including all signs. Prove rank 15 and an orthogonal 16-column completion. Exhaust closure of the specified nested C7,H267,O subalgebras. A shifted unit squares out of its shifted summand, confirming why the embedding is not an algebra homomorphism. |
| 9.2, Eqs. (50)-(51) | `18_section9_2_equivariance_global_complex_structure` | Every basis generator of both constrained Lie algebras satisfies equivariance under transport, annihilates the extra real direction, and commutes with the global sedenion L_e7. That global operator squares to -I. |
| 9.3, Figs. 5-6 | `19_section9_3_peirce_blocks_and_grading` | Complex block dimensions sum to eight and all block homomorphism dimensions sum to 64. Check the nonzero matrix-unit composition grade rule for all 512 triples. Verify both the five-bit SM grading and the four-bit version mentioned in endnote 3; also check the refined LE blocks. Record the entire commutator charge spectra and multiplicities. |
| 9.4-9.5, Eqs. (52)-(53) | `20_section9_4_tensor_clifford_and9_5_e8_dimensions` | Real/complex endomorphism tensor dimension counts, triality count 28+28+3*64=248, and the 8*(4+2+1+1)=64 split. **Dimension checks only:** no e8 Lie bracket, hypercharge choice inside e8, or Tenfold Way classification is supplied or verified. |
| 9.6, Eq. (55) | `21_section9_6_hopf_norm_identity` | Forty seeded integer input pairs for each dimension 1,2,4,8 verify the homogeneous sphere norm identity exactly; normalization then gives the unit-sphere identity. |
| 9.6, Eq. (55) | `21b_section9_6_symbolic_norm_composition` | Expand the norm-composition polynomial in independent symbolic coefficients for each of R,C,H,O. Every coefficient of the defect vanishes. This establishes the required norm identity for **all real inputs in these fixed algebras**. |
| 9.6, Eq. (56) | `22_section9_6_hopf_north_pole_derivative` | Exact north-pole differential columns are 2*conjugate(b); their ranks are 1,2,4,8. The four tangent-space dimensions total 15. |

**Not established:** global topology, identification of all fibres or local trivializations, sphere parallelizability, a complete independent irrep reconstruction of Fig. 1/reference [7], spin/chirality, physical particle counts, e8 unification, or Bott-periodic particle dynamics. The Peirce charge spectrum is recorded without attaching unverified particle names. Eq. (56)'s tangent-space identifications are checked by the differential, not by a global topology algorithm.

### 10. Outlook (page 9)

The future nested-Clifford/particle interpretation is not developed into a testable construction in this paper. The Clifford representations and even-subalgebra checks above cover its supplied algebraic ingredients. A future paper would be needed to test the promised particle decomposition in that approach. Declarations, acknowledgments and references add no numerical tests.

## Extensions beyond the paper

| Test | Scope and result |
| --- | --- |
| `23_extension_higher_hopf_obstruction` | Find a sedenion zero product deterministically. A norm-one pair of halves maps to zero under the unchanged Hopf formula. Check the low-index witness remains admissible in levels 4,5,8,16,20,32,128,1024. This is an embedded obstruction; it is not a full-array computation at those levels. |
| `24_extension_high_bit_twists_and_embedded_model` | 64 independent high-bit index pairs per level 4,8,16,20,32,64,128,256,1024, each against recursive-doubling logic. Also all 256 basis products of the embedded sedenion model per level. |
| `25_extension_basis_projectors_up_to_level12` | Every basis column of L_u R_u for u=2^(n-1), levels 1-12. Projector ranks remain 2 and 2^n-2. Record consequent joint-centralizer dimension predictions; independently solve the entire d=16 joint centralizer for unit E15 to obtain real dimension 100. The larger predicted dimensions are not solved by allocating huge matrices. |
| `25b_extension_high_bit_quaternion_subalgebras` | 32 distinct-unit quaternion constructions per level 4,8,20,64,128,1024. Verify the cyclic multiplication rules and all 64 basis associators for each constructed span. Trilinearity establishes associativity for every real linear combination in each tested span. |

## GPU checks retained for a suitable machine

| Test | Implementation and recorded outcome |
| --- | --- |
| `26_multiply_all_paper_basis_pairs_and_nested_operators` | All 64 octonion basis pairs through `multiply()` for float32, float64, complex64, complex128, then every column of all seven nested Eq. (12) identities using actual repeated GPU products. Passed in the subsequent GPU run; skipped in the original no-device run. |
| `27_multiply_sparse_witness_dense_arrays_through_level20` | Batched full float64 arrays at levels 4,8,12,16,18,20 replay a sedenion zero-product pair and a pair with high-bit support against exact sparse expectations. Passed in the subsequent GPU run; skipped in the original no-device run. |

Complex GPU dtypes here carry real basis coefficients; these tests do not claim coverage of arbitrary complex coefficient distributions or every supported module dtype. GPU mismatches would fail exact array comparison. The suite's exact CPU algebra does not depend on these optional device checks.

## Separate follow-up: classification, centralizers and explicit particle irreps

The three further investigations are implemented in a [separate section with its own tests, runner, results and write-up](extensions/README.md). It includes exhaustive XOR-subspace classification through level 9, comparisons of other basis units, and an [explicit reconstruction from reference [7]](extensions/PARTICLES.md). The latter documents a literal sign discrepancy in reference [7] separately from its tested repair. The original suite and its historical results remain unchanged.

```bash
python3 -m tests.papers.physletb_141078.run_extensions
```

## Public evidence and local artifacts

A GitHub clone includes the companion source code, tests, documentation, and
curated JSON/CSV evidence under `results/`. Historical result files are retained
unchanged, including recorded environment paths, source hashes, and PDF hashes.
Those paths describe the original machine; they are not setup instructions or
files guaranteed to exist in a clone. Saved source hashes describe the source
at the time of each run, not necessarily the current revision.

Downloaded PDFs, extracted text, rendered pages, console logs, and bulk search
archives are local artifacts and are not distributed here. Commands using
`/path/to/...` require your own files. The optional `--paper` and `--reference`
arguments hash locally supplied PDFs for provenance; ordinary tests do not
require the PDFs. Fresh `/tmp/...` output paths in examples are local scratch
locations; copy new evidence to durable storage if you need to retain it.

The extension's [composition-subspace CSV](extensions/results/composition_subspaces.csv)
and [basis-unit CSV](extensions/results/basis_units.csv) are deliberately retained
as reference evidence. The [extension runner](../../../tests/papers/physletb_141078/run_extensions.py)
can regenerate these tables into a new output directory; historical snapshots
remain unchanged.
