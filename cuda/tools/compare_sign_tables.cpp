// SPDX-License-Identifier: Apache-2.0
// Host-only sign check; no CUDA toolkit or GPU is required.
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <utility>

// Unsigned host translation of cayley_sign in cuda/cuda_kernel.cu.
// Keep this expression synchronized with the device implementation.
static int closed_sign(std::uint32_t r, std::uint32_t c) {
    const auto o = r | c;
    const auto x = r ^ c;
    const auto d = x & -x;
    const auto h = r & -r;
    const auto v = c & -c;
    const bool a = r && c;
    const auto p = (!x || !a) ? std::uint32_t(!a) :
                   v < h ? ((~c & h) ^ (o & -h)) :
                   d < v ? ((r & v) ^ (o & -v)) :
                           ((c & d) ^ (o & -d));
    return (__builtin_popcount(p) & 1) ? 1 : -1;
}

// Independent Cayley-Dickson doubling recurrence, from highest bit down.
static int doubling_sign(std::uint32_t r, std::uint32_t c) {
    int sign = 1;
    for (int bit = 31; bit >= 0; --bit) {
        const auto half = std::uint32_t(1) << bit;
        const bool upper_r = r & half, upper_c = c & half;
        r &= half - 1;
        c &= half - 1;
        if (upper_r) {
            if (c) sign = -sign;
            if (upper_c) sign = -sign;
        }
        if (upper_c) std::swap(r, c);
    }
    return sign;
}

static bool check(std::uint32_t r, std::uint32_t c) {
    if (closed_sign(r, c) == doubling_sign(r, c)) return true;
    std::cerr << "sign mismatch at (" << r << ", " << c << ")\n";
    return false;
}

int main(int argc, char** argv) {
    // Argument is a maximum algebra dimension, not a doubling level.
    char* end = nullptr;
    const auto dim = argc == 2 ? std::strtoul(argv[1], &end, 10) : 0;
    if (argc != 2 || end == argv[1] || *end || dim == 0 ||
        dim > 4096 || (dim & (dim - 1))) {
        std::cerr << "usage: compare_sign_tables DIM (power of two, 1..4096)\n";
        return 2;
    }
    for (std::uint32_t r = 0; r < dim; ++r)
        for (std::uint32_t c = 0; c < dim; ++c)
            if (!check(r, c)) return 1;

    // Cover the signed 32-bit GPU index range beyond the small dense table.
    std::mt19937 rng(2026);
    for (int n = 0; n < 10000; ++n) {
        const auto r = rng() & 0x7fffffffu, c = rng() & 0x7fffffffu;
        if (!check(r, c) || !check(r, r) || !check(r, 0) || !check(0, c))
            return 1;
    }
    std::cout << "Sign table agrees with doubling through dimension " << dim
              << "; 40000 large-index checks passed.\n";
}
