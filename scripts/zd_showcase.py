# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Find and confirm huge sparse zero divisors using only the new formulas.

Run from a clone with ``make zd-showcase``. No GPU, pytest, dense vectors,
or recursive doubling reference is used. All indices are zero-based except
at the adapter to basis's signed, one-based encoding.
"""

import argparse
import json
from pathlib import Path
import random
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from CayleyKernel import basis, twist

def safe_getrandbits(rng, total_bits):
    """Bypasses Python's 32-bit C-boundary by chunking huge bit requests."""
    # 1 billion bits is safely under the 2,147,483,647 limit
    CHUNK_SIZE = 1_000_000_000

    if total_bits <= CHUNK_SIZE:
        return rng.getrandbits(total_bits)

    result = 0
    remaining = total_bits
    while remaining > 0:
        bits_to_get = min(remaining, CHUNK_SIZE)
        # Shift the accumulator left and layer in the new random bits
        result = (result << bits_to_get) | rng.getrandbits(bits_to_get)
        remaining -= bits_to_get
    return result


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _levels(value):
    try:
        levels = [int(part) for part in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("use comma-separated levels >= 5")
    if not levels or any(level < 5 for level in levels):
        # All four indices occupy the upper half; at level 4 the resulting
        # product reduces to an octonion product, which cannot vanish.
        raise argparse.ArgumentTypeError("use comma-separated levels >= 5")
    return levels


def _preview(index):
    bits = index.bit_length()
    if bits <= 64:
        return hex(index)
    return f"0x{index >> (bits - 32):08x}...{index & 0xffffffff:08x}"


def _terms(a, b, c, d, s, t):
    return (basis(a + 1, c + 1),
            basis(a + 1, t * (d + 1)),
            basis(s * (b + 1), c + 1),
            basis(s * (b + 1), t * (d + 1)))


def _accumulate(terms):
    coefficients = {}
    for encoded in terms:
        index = abs(encoded) - 1
        coefficients[index] = coefficients.get(index, 0) + (1 if encoded > 0 else -1)
    return {index: coefficient for index, coefficient in coefficients.items() if coefficient}


def confirm(a, b, c, d, s, t):
    """Re-evaluate four signed products and a deliberately broken witness."""
    if len({a, b, c, d}) != 4 or a ^ b != c ^ d:
        raise AssertionError("invalid witness support")
    terms = _terms(a, b, c, d, s, t)
    if terms[0] != -terms[3] or terms[1] != -terms[2] or _accumulate(terms):
        raise AssertionError("four basis products did not cancel exactly")
    if not _accumulate(_terms(a, b, c, d, s, -t)):
        raise AssertionError("sign-flipped control unexpectedly vanished")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=_levels, default=_levels("4096,65536,1000000"),
                        help="doubling levels / index bit lengths (default: 4096,65536,1000000)")
    parser.add_argument("--witnesses", type=_positive, default=10,
                        help="confirmed random zero divisors per level (default: 10)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-attempts", type=_positive, default=10000,
                        help="candidate limit per level (default: 10000)")
    parser.add_argument("--output", type=Path,
                        help="optional JSON file containing full hexadecimal indices")
    args = parser.parse_args(argv)
    records = []
    print("CayleyKernel: enormous zero divisors, exact cancellation")
    print(f"New formulas only | CPU | seed={args.seed} | no dense arrays")
    print("x = e_a + s*e_b; y = e_c + t*e_d; x != 0, y != 0, x*y = 0")
    print("Indices below are zero-based hexadecimal previews; reproduce with the seed.")
    total_start = perf_counter()
    for level in args.levels:
        rng = random.Random(args.seed + level)
        top = 1 << (level - 1)
        found = 0
        verification_seconds = 0.0
        start = perf_counter()
        print(f"\nLEVEL {level:,} | dimension = 2^{level:,} | all four indices have {level:,} bits")
        for attempt in range(1, args.max_attempts + 1):
            #a, b, c = (rng.getrandbits(level) | top for _ in range(3))
            a, b, c = (safe_getrandbits(rng,level) | top for _ in range(3))
            d = a ^ b ^ c
            if len({a, b, c, d}) != 4:
                continue
            # XOR gives the two matching output indices; the twist signs
            # determine whether their coefficients can both cancel.
            ac, ad, bc, bd = (twist(p, q) for p, q in
                              ((a, c), (a, d), (b, c), (b, d)))
            s = rng.choice((-1, 1))
            t = -s * ac * bd
            if t * ad + s * bc:
                continue
            check_start = perf_counter()
            confirm(a, b, c, d, s, t)
            check_seconds = perf_counter() - check_start
            verification_seconds += check_seconds
            found += 1
            print(f"  CONFIRMED {found:>3}/{args.witnesses}: xy = 0 exactly; "
                  f"sign-flipped control != 0; check={check_seconds * 1000:.3f} ms")
            if found == 1:
                print(f"    a={_preview(a)}  b={_preview(b)}")
                print(f"    c={_preview(c)}  d={_preview(d)}  s={s:+d}  t={t:+d}")
                print("    Four nonzero basis products -> two exact cancellations -> zero.")
            if args.output:
                records.append({"level": level, "attempt": attempt,
                                "indices_hex": [hex(index) for index in (a, b, c, d)],
                                "s": s, "t": t})
            if found == args.witnesses:
                break
        else:
            raise RuntimeError(f"level {level}: found {found}/{args.witnesses} witnesses "
                               f"within {args.max_attempts} attempts")
        elapsed = perf_counter() - start
        print(f"  {found} confirmed from {attempt} candidates in {elapsed:.3f} s; "
              f"mean confirmation + negative control: {verification_seconds / found * 1000:.3f} ms")
    if args.output:
        args.output.write_text(json.dumps({"seed": args.seed, "index_base": 0,
                                          "verification": "new formulas only",
                                          "witnesses": records}, indent=2) + "\n")
        print(f"\nFull witnesses saved to {args.output}")
    print(f"\nSUCCESS: {len(args.levels) * args.witnesses} enormous zero-divisor pairs "
          f"confirmed in {perf_counter() - total_start:.3f} s.")
    print("Formula consistency checks; no independent doubling-reference comparison.")


if __name__ == "__main__":
    main()
