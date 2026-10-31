# Rust CPU benchmark snapshot

Measured on 2026-10-31 on an AMD Ryzen 9 7900X, Linux x86_64, CPython
3.14.3, NumPy 2.5.4, rustc 1.99.1, and Maturin 1.16.0. Release builds use
the checked-in Cargo lockfile. These are local workstation measurements, not
performance guarantees. Correctness is checked before timing; timings have no
pass/fail threshold.

## Dense real multiplication

The same pre-generated Python lists were passed to `cpu_multiply` with the
Python and Rust backends. Five repetitions, seed 2026; medians include input
conversion and NumPy output construction. Both paths return identical float64
coefficients, preserving row summation order. The algorithm remains O(n²).

| Dimension | Python µs/product | Rust µs/product | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 1.814 | 1.715 | 1.06× |
| 2 | 2.596 | 1.809 | 1.44× |
| 4 | 5.883 | 2.001 | 2.94× |
| 8 | 20.425 | 2.390 | 8.55× |
| 32 | 336.598 | 5.638 | 59.70× |
| 128 | 5401.115 | 36.564 | 147.72× |

The smallest inputs are dominated by wrapper costs; orders-of-magnitude gains
appear at dimensions 32 and 128 in this run. Four output tallies are manually
unrolled; this does not depend on Rayon or guarantee hardware SIMD.

## Fixed-width versus generic BigUint signs

Native-only timings compare public dispatch with the generic BigUint sign
formula on the same 128 sparse high-bit pairs. Each of five samples runs 1,000
iterations over those pairs; the table reports median nanoseconds per sign.
Inputs and results pass through `black_box`; no Python conversions are included.

| Bits | Dispatched ns | Generic BigUint ns |
| ---: | ---: | ---: |
| 32 | 3.49 | 20.56 |
| 64 | 3.54 | 20.48 |
| 65 | 9.27 | 39.12 |
| 128 | 9.37 | 40.47 |
| 129 | 10.69 | 42.64 |
| 256 | 12.02 | 42.57 |
| 257 | 44.31 | 42.99 |
| 1024 | 53.18 | 51.77 |
| 4096 | 108.92 | 107.55 |

The two-limb path covers 65–128 bits, the four-limb path 129–256 bits.
They avoid BigUint XOR/OR/shift allocations and improve these measurements by
about 3.5–4.3×. At 257 bits dispatch returns to the generic implementation;
the small difference there represents dispatch/timing overhead. Arbitrarily
large indices remain exact and scale with their bit length.

## End-to-end integer crossover

Random-index Python API timings include normalization, conversion to BigUint,
and conversion of basis labels back to Python. With 128 pairs and five samples,
Rust `twist` was 2.58× faster at 32 bits, 2.18× at 128 bits, 1.94× at 256 bits,
and 1.53× at 4096 bits. Corresponding `basis` speedups were 2.60×, 2.06×,
1.86×, and 1.12×.

A separate larger-width run used 16 pairs and three samples:

| Bits | `twist` Python/Rust µs | `basis` Python/Rust µs | `twist_many` Python/Rust µs | `basis_many` Python/Rust µs |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 1.939 / 1.227 | 2.856 / 2.601 | 1.892 / 1.237 | 2.841 / 2.628 |
| 16384 | 3.319 / 2.315 | 4.776 / 4.896 | 3.231 / 2.318 | 4.747 / 4.780 |
| 65536 | 10.955 / 8.523 | 15.727 / 18.472 | 10.874 / 8.381 | 15.672 / 21.123 |
| 262144 | 42.688 / 33.195 | 66.425 / 77.374 | 42.480 / 46.171 | 66.300 / 94.271 |
| 1048576 | 170.841 / 131.952 | 482.309 / 333.165 | 169.921 / 194.735 | 480.515 / 407.087 |

`basis` was near parity at 16,384 bits and slower in Rust at 65,536 and
262,144 bits, then faster again at 1,048,576 bits. Batched `twist` crossed
below parity at 262,144 bits while individual `twist` stayed faster through
1,048,576 bits. There is no single monotonic crossover: Python integer
algorithms, allocations, batch size, index shape, and conversion costs matter.
Select the backend using measurements from the intended workload.

## Reproduce

From the repository root with the extension installed in the active Python
environment:

```sh
python scripts/benchmark_cpu_backends.py \
  --bits 32 64 65 128 129 256 257 1024 4096 \
  --pairs 128 --repeats 5 --dims 1 2 4 8 32 128
python scripts/benchmark_cpu_backends.py \
  --bits 8192 16384 65536 262144 1048576 --pairs 16 --repeats 3 --dims 8
cargo test --manifest-path rust/Cargo.toml --locked --release \
  benchmark_index_paths -- --ignored --nocapture
```

The native benchmark deliberately uses sparse indices; the Python benchmark
uses random indices. Their speedup ratios answer different questions and should
not be compared as equivalent workloads.
