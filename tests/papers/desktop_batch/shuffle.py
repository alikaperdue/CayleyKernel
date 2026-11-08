# SPDX-License-Identifier: Apache-2.0
"""Shared Bales shuffle rules and the adapter to the block-basis kernel."""

from CayleyKernel import twist

from .common import basis_product


def omega(p, q, kind):
    """Table 1 doubling, LSB shuffle basis, independent of twist trees."""
    s = 1
    while p or q:
        u, v = p & 1, q & 1
        a, b = p >> 1, q >> 1
        k = kind % 4
        if not u and not v:
            swap = k < 2
        elif not u and v:
            if kind < 4:
                s *= -1 if a else 1
                swap = k in (0, 2)
            else:
                swap = k in (1, 3)
        elif u and not v:
            if kind < 4:
                swap = k in (1, 3)
            else:
                s *= -1 if b else 1
                swap = k in (0, 2)
        else:
            s *= (1 if a else -1) if kind < 4 else (1 if b else -1)
            swap = k in (1, 3)
        p, q = (b, a) if swap else (a, b)
    return s


def kernel_shuffle(p, q):
    n = max(1, p.bit_length(), q.bit_length())
    rev = lambda x: int(format(x, "0%db" % n)[::-1], 2)
    a, b = rev(p), rev(q)
    label = basis_product(a + 1, b + 1)
    assert abs(label) - 1 == rev(p ^ q)
    assert (1 if label > 0 else -1) == twist(a, b)
    return twist(a, b)


TREE = {
    "C": ("C", "T", "L", "-D"),
    "T": ("T", "T", "1", "-1"),
    "L": ("L", "-1", "L", "1"),
    "-D": ("-D", "1", "-1", "-D"),
    "1": ("1",) * 4,
    "-1": ("-1",) * 4,
}


def tree2(p, q):
    node = "C"
    for k in reversed(range(max(p.bit_length(), q.bit_length()))):
        node = TREE[node][2 * ((p >> k) & 1) + ((q >> k) & 1)]
    return -1 if node in ("-1", "-D") else 1
