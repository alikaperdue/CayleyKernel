# SPDX-License-Identifier: Apache-2.0
"""Run with the pinned checkout on PYTHONPATH as shown in the appendix."""

import random

from CayleyKernel import cpu_backend, twist
from tests.support.oracles import doubling_sign_at_level


def check_pairs(pairs):
    count = 0
    for r, c in pairs:
        width = max(r.bit_length(), c.bit_length())
        expected = doubling_sign_at_level(r, c, width)
        assert doubling_sign_at_level(r, c, width + 7) == expected, (r, c)
        assert twist(r, c) == expected, (r, c, expected)
        count += 1
    return count


def dispatch_pairs():
    rng = random.Random(2026)
    for width in (63, 64, 65, 127, 128, 129, 255, 256, 257):
        top = 1 << (width - 1)
        yield 0, top
        yield top, 0
        yield top, top
        for _ in range(32):
            r = top | rng.getrandbits(width - 1)
            c = rng.getrandbits(width)
            yield r, c
            yield c, r


def limb_pairs():
    for boundary in (64, 128, 192, 256):
        top = 1 << (boundary + 3)
        for bit in range(boundary - 1, boundary + 2):
            pivot = 1 << bit
            # First set bit at the pivot, with either value of the other bit.
            for c in (top | 1, top | pivot | 1):
                yield top | pivot, c
                yield c, top | pivot
            # Equal first set bits and first difference at the pivot.
            yield top | 1, top | pivot | 1
            yield top | pivot | 1, top | 1


if __name__ == "__main__":
    assert twist(0, 0) == doubling_sign_at_level(0, 0, 0) == 1
    dispatch = check_pairs(dispatch_pairs())
    limbs = check_pairs(limb_pairs())
    print(f"{cpu_backend()}: {dispatch} dispatch-width and {limbs} limb checks passed")
