# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
"""Exact, CPU-only checks; explicitly enable with ``make test-sparse-zd``.

All support indices here are zero-based. Only the adapter to basis
uses its signed, one-based encoding. No dense 2**n arrays are allocated.
"""

from itertools import combinations, product
import platform
import random
from statistics import median
import sys
from time import perf_counter

import pytest

from CayleyKernel import basis, twist

pytestmark = pytest.mark.sparse_zd

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



def _doubling_sign(a, b, level):
    """Independent reference from (ac-conj(d)b, da+b conj(c)).

    Process the most significant doubling bit first. The four branches
    correspond to lower/lower, lower/upper, upper/lower, upper/upper.
    This is the recursive definition evaluated iteratively, so levels above
    Python's recursion limit are supported. It does not call twist or use
    its trailing-zero/popcount formula.
    """
    sign = 1
    for bit in range(level - 1, -1, -1):
        half = 1 << bit
        upper_a, upper_b = bool(a & half), bool(b & half)
        a, b = a & (half - 1), b & (half - 1)
        if upper_a:
            if b:  # conjugation of the second operand's lower component
                sign = -sign
            if upper_b:  # minus conj(d)*b
                sign = -sign
        if upper_b:  # da or conj(d)b reverses the remaining factors
            a, b = b, a
    return sign


def _kernel_sparse(x, y):
    out = {}
    for a, sa in x:
        for b, sb in y:
            encoded = basis(sa * (a + 1), sb * (b + 1))
            index = abs(encoded) - 1
            out[index] = out.get(index, 0) + (1 if encoded > 0 else -1)
    return {index: value for index, value in out.items() if value}


def _reference_sparse(x, y, signs):
    out = {}
    for i, (a, sa) in enumerate(x):
        for j, (b, sb) in enumerate(y):
            index = a ^ b
            out[index] = out.get(index, 0) + sa * sb * signs[2 * i + j]
    return {index: value for index, value in out.items() if value}


def _check_support(a, b, c, d, level, context):
    signs = [_doubling_sign(p, q, level)
             for p, q in ((a, c), (a, d), (b, c), (b, d))]
    zeros = 0
    for s, t in product((-1, 1), repeat=2):
        x, y = ((a, 1), (b, s)), ((c, 1), (d, t))
        expected = _reference_sparse(x, y, signs)
        actual = _kernel_sparse(x, y)
        assert actual == expected, (
            f"{context}; support={(a, b, c, d)}; signs={(s, t)}; "
            f"expected={expected}; actual={actual}"
        )
        cancels = (a ^ b == c ^ d and
                   signs[0] + s * t * signs[3] == 0 and
                   t * signs[1] + s * signs[2] == 0)
        assert cancels == (not expected), context
        zeros += not expected
    return zeros


def test_exhaustive_small_basiss():
    # Checks the reference/sign convention and all signed input combinations.
    for a, b in product(range(64), repeat=2):
        sign = _doubling_sign(a, b, 6)
        for sa, sb in product((-1, 1), repeat=2):
            assert basis(sa * (a + 1), sb * (b + 1)) == (
                sa * sb * sign * ((a ^ b) + 1)
            ), (a, b, sa, sb)


def test_exhaustive_sedenion_two_term_products():
    # All unordered two-element supports in each factor, all relative signs.
    supports = list(combinations(range(16), 2))
    zeros = 0
    for (a, b), (c, d) in product(supports, repeat=2):
        zeros += _check_support(a, b, c, d, 4, "exhaustive level=4")
    assert zeros > 0
    print(f"\nlevel=4 exhaustive: {4 * len(supports)**2} products; {zeros} zero products")


def test_embedded_known_witness(zd_level):
    # An embedding regression, deliberately separate from high-bit sampling.
    x, y = ((3, 1), (10, 1)), ((6, 1), (15, -1))
    signs = [_doubling_sign(a, b, zd_level) for a, _ in x for b, _ in y]
    assert _reference_sparse(x, y, signs) == {}
    assert _kernel_sparse(x, y) == {}


def _supports(level, samples, seed):
    rng = random.Random(seed + level)
    top = 1 << (level - 1)
    mask = (1 << level) - 1
    # Exercise isolated bits, dense runs, alternating bits, then uniform draws.
    alternating = sum(1 << bit for bit in range(0, level, 2))
    patterns = [(top, 1, 2), (mask, 1, top),
                (top | 1, top, 2), (alternating | top, 1, mask)]
    for sample in range(samples):
        if sample < len(patterns):
            a, b, c = patterns[sample]
        else:
            #a = rng.getrandbits(level) | top
            #b = rng.getrandbits(level)
            #while b == a:
                #b = rng.getrandbits(level)
            #c = rng.getrandbits(level)
            a = safe_getrandbits(rng,level) | top
            b = safe_getrandbits(rng,level)
            while b == a:
                b = safe_getrandbits(rng,level)
            c = safe_getrandbits(rng,level)
        d = a ^ b ^ c
        yield a, b, c, d


def test_high_bit_sparse_candidates(zd_level, request):
    samples = request.config.getoption("--zd-samples")
    seed = request.config.getoption("--zd-seed")
    zeros = 0
    for sample, (a, b, c, d) in enumerate(_supports(zd_level, samples, seed)):
        context = f"level={zd_level}; seed={seed}; sample={sample}"
        assert a != b and c != d and max(a, b, c, d).bit_length() == zd_level
        zeros += _check_support(a, b, c, d, zd_level, context)
        # Break XOR compatibility while retaining two nonzero terms in y.
        flip = 1 if a ^ b != 1 else 2
        assert _check_support(a, b, c, d ^ flip, zd_level, context + "; negative") == 0
    print(f"\nlevel={zd_level}; seed={seed}; candidates={samples}; "
          f"XOR-compatible products={4 * samples}; zero products={zeros}; "
          f"guaranteed nonzero controls={4 * samples}")


def _doubling_basis(r, c, level):
    """Match the public API's signed, one-based input/output convention."""
    a, b = abs(r) - 1, abs(c) - 1
    sign = -1 if (r > 0) != (c > 0) else 1
    return sign * _doubling_sign(a, b, level) * ((a ^ b) + 1)


def _doubling_sparse(x, y, level):
    # Recompute every sign, just as _kernel_sparse does; no cached signs.
    signs = [_doubling_sign(a, b, level) for a, _ in x for b, _ in y]
    return _reference_sparse(x, y, signs)


def _batch_seconds(function, cases, loops):
    start = perf_counter()
    for _ in range(loops):
        for args in cases:
            function(*args)
    return perf_counter() - start


def _compare_timings(old, new, cases, repeats):
    # Equality checks also warm both paths, outside all timed sections.
    for index, args in enumerate(cases):
        assert old(*args) == new(*args), f"timing workload mismatch at case={index}"
    functions = (old, new)
    loops = []
    for function in functions:
        count = 1
        while _batch_seconds(function, cases, count) < 0.02 and count < 65536:
            count *= 2
        loops.append(count)
    timings = [[], []]
    for repeat in range(repeats):
        # Alternate execution order to reduce systematic ordering effects.
        for method in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            elapsed = _batch_seconds(functions[method], cases, loops[method])
            timings[method].append(elapsed / (loops[method] * len(cases)))
    return tuple(median(values) for values in timings)


def test_method_timings(zd_level, request):
    """Report speed, without machine-dependent pass/fail thresholds.

    Baseline: the Python doubling reference above, not a claimed reproduction
    or benchmark of Bales's implementation. All methods execute on the CPU.
    """
    samples = request.config.getoption("--zd-samples")
    seed = request.config.getoption("--zd-seed")
    repeats = request.config.getoption("--zd-repeats")
    supports = list(_supports(zd_level, samples, seed))
    pairs = [(p, q) for a, b, c, d in supports
             for p, q in ((a, c), (a, d), (b, c), (b, d))]
    encoded = [((-1 if index & 1 else 1) * (a + 1),
                (-1 if index & 2 else 1) * (b + 1))
               for index, (a, b) in enumerate(pairs)]
    sparse = [(((a, 1), (b, s)), ((c, 1), (d, t)))
              for a, b, c, d in supports for s, t in product((-1, 1), repeat=2)]
    workloads = [
        ("sign", lambda a, b: _doubling_sign(a, b, zd_level), twist, pairs),
        ("basis product", lambda r, c: _doubling_basis(r, c, zd_level),
         basis, encoded),
        ("two-term product", lambda x, y: _doubling_sparse(x, y, zd_level),
         _kernel_sparse, sparse),
    ]
    print(f"\nCPU timings: Python {sys.version.split()[0]}, {platform.machine()}, "
          f"{platform.system()}; level={zd_level}; seed={seed}; "
          f"candidates={samples}; median of {repeats} rounds")
    print("  workload             cases   doubling us/op   new us/op   speedup")
    for name, old, new, cases in workloads:
        old_seconds, new_seconds = _compare_timings(old, new, cases, repeats)
        print(f"  {name:<20} {len(cases):>5} {old_seconds * 1e6:>16.3f} "
              f"{new_seconds * 1e6:>11.3f} {old_seconds / new_seconds:>8.2f}x")
