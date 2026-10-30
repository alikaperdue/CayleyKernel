// amd_kernel.cu  (HIP / ROCm build of kernel.cu)
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

#include <hip/hip_runtime.h>
#include <cstdint>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>

//const int threads_per_block = 256;
const int threads_per_block = 1024;

// Cap on the .y grid dimension; the batch axis uses a grid-stride loop.
const int max_grid_y = 65535;

template <typename T> struct accum_traits         { typedef     T type; };
template <>           struct accum_traits<bool>   { typedef   int type; };
// FP16 storage, FP32 products and accumulation; round once on output.
template <>           struct accum_traits<__half> { typedef float type; };

// Keep scalar arithmetic unchanged; complex storage types use the vendor's
// explicit arithmetic functions instead of C++ scalar operators.
template <typename T> struct arithmetic_traits {
    __device__ static __forceinline__ T zero() { return T(0); }
    __device__ static __forceinline__ T mul(T a, T b) { return a * b; }
    __device__ static __forceinline__ void add(T& acc, T term) { acc += term; }
    __device__ static __forceinline__ void sub(T& acc, T term) { acc -= term; }
};

template <> struct arithmetic_traits<hipFloatComplex> {
    __device__ static __forceinline__ hipFloatComplex zero() {
        return make_hipFloatComplex(0, 0);
    }
    __device__ static __forceinline__ hipFloatComplex mul(hipFloatComplex a, hipFloatComplex b) {
        return hipCmulf(a, b);
    }
    __device__ static __forceinline__ void add(hipFloatComplex& acc, hipFloatComplex term) {
        acc = hipCaddf(acc, term);
    }
    __device__ static __forceinline__ void sub(hipFloatComplex& acc, hipFloatComplex term) {
        acc = hipCsubf(acc, term);
    }
};

template <> struct arithmetic_traits<hipDoubleComplex> {
    __device__ static __forceinline__ hipDoubleComplex zero() {
        return make_hipDoubleComplex(0, 0);
    }
    __device__ static __forceinline__ hipDoubleComplex mul(hipDoubleComplex a, hipDoubleComplex b) {
        return hipCmul(a, b);
    }
    __device__ static __forceinline__ void add(hipDoubleComplex& acc, hipDoubleComplex term) {
        acc = hipCadd(acc, term);
    }
    __device__ static __forceinline__ void sub(hipDoubleComplex& acc, hipDoubleComplex term) {
        acc = hipCsub(acc, term);
    }
};


// Sign of the Cayley-Dickson basis product of e_row * e_col = sign * e_(row^col)
// Implements the recent O(1) sign formula discovery which is credited in the README
// Return the sign portion as 1 or 0 for +1 or -1
__device__ __forceinline__
int cayley_sign(int r, int c) {

    const int o = r | c;
    const int x = r ^ c;
    const int d = x & -x;
    const int v = r & -r;
    const int h = c & -c;
    const int a = r && c;

    return __popc( static_cast<unsigned int>(
        !x | !a ? !a :
         h <  v ? ~c & v ^ o & -v :
         d <  h ?  r & h ^ o & -h :
                   c & d ^ o & -d
    ) & 1;
}

// mult_kernel() - performs the Cayley Dickson multiplication using the GPU matrix acelleration
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
        const T* a_vec =   a + offset;
        const T* b_vec =   b + offset;
        T* out_vec     = out + offset;

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

template <typename T>
static void launch_mult(const T* a, const T* b, T* o, int dim, long long batch) {
    if (dim <= 0 || batch <= 0) return;

    int threads = threads_per_block;
    if (dim < threads_per_block) {
        threads = ((dim + 63) / 64) * 64;   // wavefront is 64 on AMD
        if (threads < 64) threads = 64;
    }

    unsigned int grid_x = (unsigned int)((dim + threads - 1) / threads);
    unsigned int grid_y = (unsigned int)(batch < (long long)max_grid_y ? batch : (long long)max_grid_y);

    dim3 grid(grid_x, grid_y, 1);
    dim3 block((unsigned int)threads, 1, 1);

    mult_kernel<T><<<grid, block>>>(a, b, o, dim, batch);
    hipDeviceSynchronize();
}

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
    DEFINE_MULT_ENTRIES(complex64,  hipFloatComplex)
    DEFINE_MULT_ENTRIES(complex128, hipDoubleComplex)
}
