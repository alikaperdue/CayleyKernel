# Coefficient types in the paper tests

The optional device replays use `tests/papers/type_replay.py`. Every original
real product still runs with its original inputs and expected result. The
same operand fixtures also exercise `float16`, `complex64` and `complex128`
through the actual `CayleyKernel.multiply()` API, with dtype assertions.
There is no CPU fallback for a missing CUDA runtime or missing typed entry
point. Missing prerequisites skip the GPU test; a missing entry point in an
otherwise usable library fails and requires rebuilding the kernel.

For these additional replays, each real operand is rescaled independently
by a power of two when its largest coordinate exceeds one. This bounds half
outputs even for the large intermediate powers in the radius companion.
References are computed from the quantized inputs in float64, then rounded
to half. The comparison permits one half relative rounding unit (rtol .001)
and one smallest half subnormal (atol 2^-24). Integer, dyadic and zero-product
fixtures remain useful exact controls; these are product checks, not claims
of exact real identities under repeated half rounding.

Complex tests include two cases per precision: scalar phases `(1+i)` and
`(1-2i)` applied to the two operands, and independent imaginary coordinates
made by cyclically shifting their real coordinates. The first preserves
zero-divisor witnesses under nonzero central scaling; the second checks
products that cannot be reduced to one common phase. An independent
four-call Cayley–Dickson recursion computes the expectation in complex128
from the actual input values. Basis conjugation negates non-scalar basis
coordinates and **does not conjugate the central complex coefficient**.
The tolerances are 2e-5 for complex64 and 1e-12 for complex128, on these
bounded fixtures. These checks do not extend positive real norm, ordered
field, real rank, topology or real division-algebra claims to complex scalars.

Split-signature supplements in the Zhilina, Ren–Zhao, Clifford/Hurwitz and
Wilmot companions use the new complex kernels more directly. For each +1
doubling parameter, send that generator to `i` times the standard generator.
A basis monomial receives the product of its generator phases. Every ordered
basis product and eight seeded dense pairs are compared with independently
parameterized real doubling, transported by this coordinate map. This checks
a complex embedding of the real split algebra; the native kernel still uses
the standard all-negative sequence and accepts no signature argument.

The finite-field companion uses half only for p=3,5,7 and dimensions 4,8,16:
nonnegative residues give integer outputs bounded by 16*6^2=576, so every
product, accumulation and final half output is exact. Modular reduction is
valid on this range. Complex coefficients are characteristic-zero scalars
and do not implement finite-field arithmetic. The existing int64 finite-field
checks remain the authoritative broader-domain checks.

The tensorial-network companion additionally compares a complete dense layer
with broadcast typed device products and a channel sum, using bounded integer
coefficients for exact half products and sums.

The holonomy companion additionally tests native, one-coordinate complex
phase multiplication, broadcasting and inverse phases. The quasilinear
companion compares complex device products with the four-real-product
bilinear extension of its independent real QL implementation. Its real
operation counts and real-only Python API retain their original meaning.

Run without writing paper reports:

```bash
python3 -m pytest tests/papers -q -p no:cacheprovider
```

The CPU tests in `tests/papers/test_type_replay.py` independently validate
fixture expectations, all signatures through dimension 16, broadcasting,
zero-divisor phase covariance and the half quantization protocol. They do
not establish device execution. Device replays require CuPy, CUDA and a
shared library rebuilt with the new entry points. Existing saved result
files describe their recorded runs and have not been regenerated for this
coverage; their hashes and pass counts do not identify the current sources.
