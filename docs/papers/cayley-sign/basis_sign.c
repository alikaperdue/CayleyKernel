// SPDX-License-Identifier: Apache-2.0
#include <limits.h>
#include <stdint.h>
#include <stdio.h>

_Static_assert(UINT_MAX == UINT32_MAX, "requires 32-bit unsigned int");

static uint32_t lowbit(uint32_t x) {
    return x & (0u - x);
}

static int basis_sign(uint32_t r, uint32_t c) {
    if (r == 0 || c == 0)
        return 1;
    if (r == c)
        return -1;

    const uint32_t d = lowbit(r ^ c);
    const uint32_t R = lowbit(r), C = lowbit(c);
    const uint32_t u = r | c;
    uint32_t mask;
    if (C < R)
        mask = (~c & R) ^ (u & (0u - R));
    else if (d < C)
        mask = (r & C) ^ (u & (0u - C));
    else
        mask = (c & d) ^ (u & (0u - d));

    const int p = __builtin_popcount(mask) & 1;
    return 2 * p - 1;
}

static int doubling_sign(uint32_t r, uint32_t c) {
    int sign = 1;
    for (int i = 31; i >= 0; --i) {
        const uint32_t half = 1u << i;
        const int top_r = (r & half) != 0;
        const int top_c = (c & half) != 0;
        r &= half - 1;
        c &= half - 1;
        if (top_r && c != 0)
            sign = -sign;
        if (top_r && top_c)
            sign = -sign;
        if (top_c) {
            const uint32_t tmp = r;
            r = c;
            c = tmp;
        }
    }
    return sign;
}

static int agrees(uint32_t r, uint32_t c) {
    if (basis_sign(r, c) == doubling_sign(r, c))
        return 1;
    fprintf(stderr, "sign mismatch at (%lu, %lu)\n",
            (unsigned long)r, (unsigned long)c);
    return 0;
}

static uint32_t next_word(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}

int main(void) {
    for (uint32_t r = 0; r < 4096; ++r)
        for (uint32_t c = 0; c < 4096; ++c)
            if (!agrees(r, c))
                return 1;

    uint32_t state = 2026;
    for (int i = 0; i < 10000; ++i) {
        const uint32_t r = next_word(&state);
        const uint32_t c = next_word(&state);
        if (!agrees(r, c) || !agrees(c, r) || !agrees(r, r) ||
            !agrees(r, 0) || !agrees(0, c))
            return 1;
    }
    puts("16777216 exhaustive and 50000 full-width checks passed.");
    return 0;
}
