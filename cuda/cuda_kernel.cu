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
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

//const int threads_per_block = 256;
const int threads_per_block = 1024;

// CUDA allows at most 65535 blocks in the .y grid dimension, so the batch axis
// is walked with a grid-stride loop.
const int max_grid_y = 65535;

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
    const int v = r & -r;
    const int h = c & -c;
    const int a = r && c;

    return __popc(
        !x | !a ? !a :
         h <  v ? ~c & v ^ o & -v :
         d <  h ?  r & h ^ o & -h :
                   c & d ^ o & -d
    ) & 1;
}

// 1. One template to rule them all
//    x = blockIdx.x/threadIdx.x -> component (column) index inside one vector
//    y = blockIdx.y             -> batch index (grid-stride)
template <typename T>
__global__ void mult_kernel(const T* __restrict__ a,
                            const T* __restrict__ b,
                                  T* __restrict__ out,
                            int dim,
                            long long batch) {
    typedef typename accum_traits<T>::type Acc;
    typedef arithmetic_traits<Acc> Ops;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= dim) return;

    for (long long item = blockIdx.y; item < batch; item += gridDim.y) {
        const long long offset = item * (long long)dim;
        const T* a_vec = a + offset;
        const T* b_vec = b + offset;
        T* out_vec   = out + offset;

        Acc acc = Ops::zero();

        for (int row = 0; row < dim; row++) {
	    int col = index ^ row;
            const Acc term = Ops::mul(Acc(a_vec[row]), Acc(b_vec[col]));
            cayley_sign(row, col) ?
                Ops::add(acc, term) :
                Ops::sub(acc, term) ;
        }

        out_vec[index] = static_cast<T>(acc);
    }
}

// Shared launch helper: sizes the grid over both the component axis and the
// batch axis.
template <typename T>
static void launch_mult(const T* a, const T* b, T* o, int dim, long long batch) {
    if (dim <= 0 || batch <= 0) return;

    // Do not spawn 1024 threads for a 4-component vector.
    int threads = threads_per_block;
    if (dim < threads_per_block) {
        threads = ((dim + 31) / 32) * 32;
        if (threads < 32) threads = 32;
    }

    unsigned int grid_x = (unsigned int)((dim + threads - 1) / threads);
    unsigned int grid_y = (unsigned int)(batch < (long long)max_grid_y ? batch : (long long)max_grid_y);

    dim3 grid(grid_x, grid_y, 1);
    dim3 block((unsigned int)threads, 1, 1);

    mult_kernel<T><<<grid, block>>>(a, b, o, dim, batch);
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
