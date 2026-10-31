// cuda_kernel.cu
//
// Cayley-Dickson doubling product kernel.
//
// The algebra dimension ("dim") is the length of the *last* axis of the input
// and must be a power of two.  Any number of leading axes are treated as a
// batch, i.e. an input of shape (..., dim) is interpreted as
// prod(...) independent Cayley-Dickson vectors laid out contiguously.
//
// Two entry points are exported for every supported type:
//
//   mult_<type>   (a, b, out, dim)            -> single vector (legacy API)
//   mult_<type>_nd(a, b, out, dim, batch)     -> batched / multi-dimensional
//
// The legacy entry point is simply the batched one with batch == 1, so old
// callers keep working unchanged.
//
// This code implement O(1) sign determination first seen by this author here:
//
// An O(1) Bitwise Evaluator for Cayley–Dickson Sign Structure: Ordinary, Split, Dual, and Tensor Constructions -
// https://figshare.com/articles/preprint/A_Proven_Sign_Law_for_Cayley-Dickson_Algebras_Ordinary_and_Split_Constructions/33705022

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Bound the number of resident work tiles; all batch sizes use grid strides.
constexpr int max_work_blocks = 65535;

// Bool accumulates in int; FP16/BF16 use float to avoid repeated low-precision
// rounding. Other types accumulate in their own type.
template <typename T> struct accum_traits { typedef     T type; };
template <> struct accum_traits<bool>     { typedef   int type; };
// FP16/BF16 storage, FP32 products and accumulation; round once on output.
template <> struct accum_traits<__half>   { typedef float type; };
template <> struct accum_traits<__nv_bfloat16> { typedef float type; };
static_assert(sizeof(__nv_bfloat16) == 2, "BF16 ABI requires 2-byte storage");

// Keep scalar arithmetic unchanged; complex storage types use the vendor's
// explicit arithmetic functions instead of C++ scalar operators.
template <typename T> struct arithmetic_traits {
    __device__ static __forceinline__ T zero()        { return T(0);  }
    __device__ static __forceinline__ T mul(T a, T b) { return a * b; }
    __device__ static __forceinline__ void add(T& acc, T term) { acc += term; }
    __device__ static __forceinline__ void sub(T& acc, T term) { acc -= term; }
};

template <> struct arithmetic_traits<cuComplex> {
    __device__ static __forceinline__ cuComplex zero() {
        return make_cuComplex(0, 0);
    }
    __device__ static __forceinline__ cuComplex mul(cuComplex a, cuComplex b) {
        return cuCmulf(a, b);
    }
    __device__ static __forceinline__ void add(cuComplex& acc, cuComplex term) {
        acc = cuCaddf(acc, term);
    }
    __device__ static __forceinline__ void sub(cuComplex& acc, cuComplex term) {
        acc = cuCsubf(acc, term);
    }
};

template <> struct arithmetic_traits<cuDoubleComplex> {
    __device__ static __forceinline__ cuDoubleComplex zero() {
        return make_cuDoubleComplex(0, 0);
    }
    __device__ static __forceinline__ cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) {
        return cuCmul(a, b);
    }
    __device__ static __forceinline__ void add(cuDoubleComplex& acc, cuDoubleComplex term) {
        acc = cuCadd(acc, term);
    }
    __device__ static __forceinline__ void sub(cuDoubleComplex& acc, cuDoubleComplex term) {
        acc = cuCsub(acc, term);
    }
};


// Sign of the Cayley-Dickson basis product  e_row * e_col = sign * e_(row^col).
//
// Branch-free closed form; replaces the previous iterative loop that reduced
// (row, col) modulo dim/2, dim/4, ...  It depends only on row and col bits
//
// Device-code notes:
//   * __ffs()  is CUDA's native "find first set" (1-based, 0 for zero) and is
//     exactly the __builtin_ctz()-based ffs32() helper used in the host-side
//     equivalence test (compare_sign_tables.cpp).
//   * __popc() is CUDA's native population count, i.e. __builtin_popcount().
//   * std::max is avoided; the three-way max is spelled out so this compiles
//     cleanly as device code.
//
// Returns +1 or -1.
__device__ __forceinline__
int cayley_sign(int r, int c) {

    const int o = r | c;
    const int x = r ^ c;
    const int d = x & -x;
    const int h = r & -r;
    const int v = c & -c;
    const int a = r && c;

    return __popc(
        !x | !a ? !a :
         v <  h ? ~c & h ^ o & -h :
         d <  v ?  r & v ^ o & -v :
                   c & d ^ o & -d
    ) & 1;
}

// Flatten batch and component axes so a block can process multiple vectors.
// Block-size specializations give the compiler the actual launch bound rather
// than imposing a 1024-thread register budget on every launch.
template <typename T, int Threads>
__global__ __launch_bounds__(Threads, 1)
void mult_kernel(const T* __restrict__ a, const T* __restrict__ b,
                 T* __restrict__ out, int dim, long long total) {
    typedef typename accum_traits<T>::type Acc;
    typedef arithmetic_traits<Acc> Ops;
    const long long stride = (long long)gridDim.x * Threads;
    for (long long pos = (long long)blockIdx.x * Threads + threadIdx.x;
         pos < total; pos += stride) {
        const int index = pos % dim;
        const long long offset = pos - index;
        Acc acc = Ops::zero();
        for (int row = 0; row < dim; ++row) {
            const int col = index ^ row;
            const Acc term = Ops::mul(Acc(a[offset + row]), Acc(b[offset + col]));
            cayley_sign(row, col) ? Ops::add(acc, term) : Ops::sub(acc, term);
        }
        out[pos] = static_cast<T>(acc);
    }
}

// For dim <= 16, each tile contains Threads/dim complete vectors. Load each
// input coefficient once per tile and reuse it for all outputs in that vector.
// Threads and dim are powers of two, so no vector crosses a tile boundary.
template <typename T, int Threads>
__global__ __launch_bounds__(Threads, 1)
void mult_small_kernel(const T* __restrict__ a, const T* __restrict__ b,
                       T* __restrict__ out, int dim, long long total) {
    __shared__ T a_tile[Threads];
    __shared__ T b_tile[Threads];
    typedef typename accum_traits<T>::type Acc;
    typedef arithmetic_traits<Acc> Ops;
    const int lane = threadIdx.x;
    const int index = lane % dim;
    const int base = lane - index;
    const long long stride = (long long)gridDim.x * Threads;
    // Every thread participates in both barriers, including partial tiles.
    for (long long tile = (long long)blockIdx.x * Threads;
         tile < total; tile += stride) {
        const long long pos = tile + lane;
        if (pos < total) {
            a_tile[lane] = a[pos];
            b_tile[lane] = b[pos];
        }
        __syncthreads();
        if (pos < total) {
            Acc acc = Ops::zero();
            for (int row = 0; row < dim; ++row) {
                const int col = index ^ row;
                const Acc term = Ops::mul(Acc(a_tile[base + row]),
                                          Acc(b_tile[base + col]));
                cayley_sign(row, col) ? Ops::add(acc, term) : Ops::sub(acc, term);
            }
            out[pos] = static_cast<T>(acc);
        }
        __syncthreads();
    }
}

// Read on each call so tuning does not require restarting Python. Python
// rejects invalid values; direct legacy C callers fall back to the heuristic.
static int launch_threads(int dim, long long total) {
    const char* value = std::getenv("CAYLEY_KERNEL_BLOCK_SIZE");
    if (value) {
        const char* names[] = {"32", "64", "128", "256", "512", "1024"};
        for (int slot = 0; slot < 6; ++slot)
            if (std::strcmp(value, names[slot]) == 0) return 32 << slot;
    }
    int threads = dim <= 16 ? 128 : 256;
    while (threads > 32 && total <= threads / 2) threads /= 2;
    return threads;
}

template <typename T, int Threads>
static void launch_sized(const T* a, const T* b, T* o, int dim, long long total) {
    const long long needed = (total - 1) / Threads + 1;
    const unsigned int blocks = (unsigned int)(needed < max_work_blocks
                                               ? needed : max_work_blocks);
    if (dim <= 16)
        mult_small_kernel<T, Threads><<<blocks, Threads>>>(a, b, o, dim, total);
    else
        mult_kernel<T, Threads><<<blocks, Threads>>>(a, b, o, dim, total);
}

template <typename T>
static void launch_mult(const T* a, const T* b, T* o, int dim, long long batch) {
    if (dim <= 0 || batch <= 0) return;
    const long long total = batch * (long long)dim;
    switch (launch_threads(dim, total)) {
#define LAUNCH_CASE(N) case N: launch_sized<T, N>(a, b, o, dim, total); break;
        LAUNCH_CASE(32)
        LAUNCH_CASE(64)
        LAUNCH_CASE(128)
        LAUNCH_CASE(256)
        LAUNCH_CASE(512)
        LAUNCH_CASE(1024)
#undef LAUNCH_CASE
    }
    cudaDeviceSynchronize();
}

// Emits both the legacy (single vector) and the batched (N-D) entry point.
#define DEFINE_MULT_ENTRIES(SUFFIX, TYPE)                                        \
    void mult_##SUFFIX(const TYPE* a, const TYPE* b, TYPE* o, int dim) {         \
        launch_mult<TYPE>(a, b, o, dim, 1LL);                                    \
    }                                                                            \
    void mult_##SUFFIX##_nd(const TYPE* a, const TYPE* b, TYPE* o,               \
                            int dim, long long batch) {                          \
        launch_mult<TYPE>(a, b, o, dim, batch);                                  \
    }

extern "C" {

    DEFINE_MULT_ENTRIES(float,  float)
    DEFINE_MULT_ENTRIES(double, double)
    DEFINE_MULT_ENTRIES(uint64, uint64_t)
    DEFINE_MULT_ENTRIES(int64,  int64_t)
    DEFINE_MULT_ENTRIES(uint32, uint32_t)
    DEFINE_MULT_ENTRIES(int32,  int32_t)
    DEFINE_MULT_ENTRIES(int16,  int16_t)
    DEFINE_MULT_ENTRIES(uint16, uint16_t)
    DEFINE_MULT_ENTRIES(int8,   int8_t)
    DEFINE_MULT_ENTRIES(uint8,  uint8_t)
    DEFINE_MULT_ENTRIES(bool,   bool)
    DEFINE_MULT_ENTRIES(half, __half)
    DEFINE_MULT_ENTRIES(bfloat16, __nv_bfloat16)
    DEFINE_MULT_ENTRIES(complex64,  cuComplex)
    DEFINE_MULT_ENTRIES(complex128, cuDoubleComplex)
}
