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
#include <cstdio>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>

// const int threads_per_block = 256;
const int threads_per_block = 1024;

// Cap on the .y grid dimension; the batch axis uses a grid-stride loop.
const int max_grid_y = 65535;

template <typename T>
struct accum_traits {
  typedef T type;
};
template <>
struct accum_traits<bool> {
  typedef int type;
};
// FP16 storage, FP32 products and accumulation; round once on output.
template <>
struct accum_traits<__half> {
  typedef float type;
};

// Keep scalar arithmetic unchanged; complex storage types use the vendor's
// explicit arithmetic functions instead of C++ scalar operators.
template <typename T>
struct arithmetic_traits {
  __device__ static __forceinline__ T zero() { return T(0); }
  __device__ static __forceinline__ T mul(T a, T b) { return a * b; }
  __device__ static __forceinline__ void add(T& acc, T term) { acc += term; }
  __device__ static __forceinline__ void sub(T& acc, T term) { acc -= term; }
};

template <>
struct arithmetic_traits<hipFloatComplex> {
  __device__ static __forceinline__ hipFloatComplex zero() { return make_hipFloatComplex(0, 0); }
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

template <>
struct arithmetic_traits<hipDoubleComplex> {
  __device__ static __forceinline__ hipDoubleComplex zero() { return make_hipDoubleComplex(0, 0); }
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
__device__ __forceinline__ int cayley_sign(int r, int c) {
  const int o = r | c;
  const int x = r ^ c;
  const int d = x & -x;
  const int h = r & -r;
  const int v = c & -c;
  const int a = r && c;

  return __popc(!x | !a ? !a
                : v < h ? ~c & h ^ o & -h
                : d < v ? r & v ^ o & -v
                        : c & d ^ o & -d) &
         1;
}

// mult_kernel() - performs the Cayley Dickson multiplication using the GPU matrix acelleration
template <typename T>

__global__ void mult_kernel(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ out,
                            int dim, long long batch) {
  typedef typename accum_traits<T>::type Acc;
  typedef arithmetic_traits<Acc> Ops;

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= dim)
    return;

  for (long long item = blockIdx.y; item < batch; item += gridDim.y) {
    const long long offset = item * (long long)dim;
    const T* a_vec = a + offset;
    const T* b_vec = b + offset;
    T* out_vec = out + offset;

    Acc acc = Ops::zero();

    for (int row = 0; row < dim; row++) {
      int col = index ^ row;
      const Acc term = Ops::mul(Acc(a_vec[row]), Acc(b_vec[col]));
      cayley_sign(row, col) ? Ops::add(acc, term) : Ops::sub(acc, term);
    }

    out_vec[index] = static_cast<T>(acc);
  }
}

template <typename T>
static void launch_mult(const T* a, const T* b, T* o, int dim, long long batch) {
  if (dim <= 0 || batch <= 0)
    return;

  int threads = threads_per_block;
  if (dim < threads_per_block) {
    threads = ((dim + 63) / 64) * 64;  // wavefront is 64 on AMD
    if (threads < 64)
      threads = 64;
  }

  unsigned int grid_x = (unsigned int)((dim + threads - 1) / threads);
  unsigned int grid_y =
      (unsigned int)(batch < (long long)max_grid_y ? batch : (long long)max_grid_y);

  dim3 grid(grid_x, grid_y, 1);
  dim3 block((unsigned int)threads, 1, 1);

  mult_kernel<T><<<grid, block>>>(a, b, o, dim, batch);
  hipDeviceSynchronize();
}

#define DEFINE_MULT_ENTRIES(SUFFIX, TYPE)                                                    \
  void mult_##SUFFIX(const TYPE* a, const TYPE* b, TYPE* o, int dim) {                       \
    launch_mult<TYPE>(a, b, o, dim, 1LL);                                                    \
  }                                                                                          \
  void mult_##SUFFIX##_nd(const TYPE* a, const TYPE* b, TYPE* o, int dim, long long batch) { \
    launch_mult<TYPE>(a, b, o, dim, batch);                                                  \
  }

// =============================================================================
// Fused multiply-add  --  out[i] = sign * sum_row +/- sign_term(c[i])
//
//   sign in {-1, +1};  +1 means multiply_add(a, b, c) = a*b + c,  -1 means
//   multiply_subtract(a, b, c) = a*b - c.  The +/- c[i] term is folded into
//   the same per-thread accumulator as the dim basis-sign terms so the whole
//   expression executes in a single kernel launch (no separate add kernel).
// =============================================================================
template <typename T>
__global__ void fma_mult_kernel(const T* __restrict__ a, const T* __restrict__ b,
                                const T* __restrict__ c, T* __restrict__ out, int dim,
                                long long batch, int sign) {
  typedef typename accum_traits<T>::type Acc;
  typedef arithmetic_traits<Acc> Ops;

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= dim)
    return;

  for (long long item = blockIdx.y; item < batch; item += gridDim.y) {
    const long long offset = item * (long long)dim;
    const T* a_vec = a + offset;
    const T* b_vec = b + offset;
    const T* c_vec = c + offset;
    T* out_vec = out + offset;

    Acc acc = Ops::zero();

    for (int row = 0; row < dim; row++) {
      int col = index ^ row;
      const Acc term = Ops::mul(Acc(a_vec[row]), Acc(b_vec[col]));
      cayley_sign(row, col) ? Ops::add(acc, term) : Ops::sub(acc, term);
    }
    sign > 0 ? Ops::add(acc, Acc(c_vec[index])) : Ops::sub(acc, Acc(c_vec[index]));

    out_vec[index] = static_cast<T>(acc);
  }
}

template <typename T>
static void launch_fma_mult(const T* a, const T* b, const T* c, T* o, int dim, long long batch,
                            int sign) {
  if (dim <= 0 || batch <= 0)
    return;
  if (sign != -1 && sign != 1)
    sign = 1;

  int threads = threads_per_block;
  if (dim < threads_per_block) {
    threads = ((dim + 63) / 64) * 64;  // wavefront is 64 on AMD
    if (threads < 64)
      threads = 64;
  }

  unsigned int grid_x = (unsigned int)((dim + threads - 1) / threads);
  unsigned int grid_y =
      (unsigned int)(batch < (long long)max_grid_y ? batch : (long long)max_grid_y);

  dim3 grid(grid_x, grid_y, 1);
  dim3 block((unsigned int)threads, 1, 1);

  fma_mult_kernel<T><<<grid, block>>>(a, b, c, o, dim, batch, sign);
  hipDeviceSynchronize();
}

// Emits the single fused entry point; the Python wrapper dispatches the same
// symbol with sign = +1 (add) or sign = -1 (subtract).
#define DEFINE_FMA_ENTRIES(SUFFIX, TYPE)                                                \
  void fma_mult_##SUFFIX(const TYPE* a, const TYPE* b, const TYPE* c, TYPE* o, int dim, \
                         long long batch, int sign) {                                   \
    launch_fma_mult<TYPE>(a, b, c, o, dim, batch, sign);                                \
  }

// Sparse CSR product (ABI-5 symbol family).
//
// The sparse path is NVIDIA-only (Phase C).  The HIP build must still
// export the same symbols so that a shared .so built from either source
// resolves in the Python wrapper.  Each entry point is an intentional
// no-op that aborts the process with a clear message; a HIP build of the
// sparse kernel is future work and is explicitly out of scope for this
// phase (see docs/sparse-csr-design.md §7 "Non-goals").

#define SPARSE_STUB(SUFFIX)                                                        \
  void sparse_mult_##SUFFIX(void* xs, void* xsg, void* xv, void* ys, void* ysg,   \
                            void* yv, void* o, int K_a, int K_b, int dim,         \
                            long long batch) {                                    \
    (void)xs; (void)xsg; (void)xv; (void)ys; (void)ysg; (void)yv; (void)o;        \
    (void)K_a; (void)K_b; (void)dim; (void)batch;                                 \
    fprintf(stderr,                                                              \
            "CayleyKernel sparse_mult_%s: sparse path is not implemented on HIP " \
            "(NVIDIA-only, Phase C). See docs/sparse-csr-design.md \u00a77.\n", #SUFFIX); \
    __builtin_unreachable();                                                     \
  }

extern "C" {
DEFINE_MULT_ENTRIES(float, float)
DEFINE_MULT_ENTRIES(double, double)
DEFINE_MULT_ENTRIES(uint64, uint64_t)
DEFINE_MULT_ENTRIES(int64, int64_t)
DEFINE_MULT_ENTRIES(uint32, uint32_t)
DEFINE_MULT_ENTRIES(int32, int32_t)
DEFINE_MULT_ENTRIES(int16, int16_t)
DEFINE_MULT_ENTRIES(uint16, uint16_t)
DEFINE_MULT_ENTRIES(int8, int8_t)
DEFINE_MULT_ENTRIES(uint8, uint8_t)
DEFINE_MULT_ENTRIES(bool, bool)
DEFINE_MULT_ENTRIES(half, __half)
DEFINE_MULT_ENTRIES(complex64, hipFloatComplex)
DEFINE_MULT_ENTRIES(complex128, hipDoubleComplex)

// Fused a*b + c / a*b - c kernels (ABI-4 symbol family).  The HIP port does
// not export BF16 (no __nv_bfloat16 on ROCm), matching the existing
// mult_<type> exports above.
DEFINE_FMA_ENTRIES(float, float)
DEFINE_FMA_ENTRIES(double, double)
DEFINE_FMA_ENTRIES(uint64, uint64_t)
DEFINE_FMA_ENTRIES(int64, int64_t)
DEFINE_FMA_ENTRIES(uint32, uint32_t)
DEFINE_FMA_ENTRIES(int32, int32_t)
DEFINE_FMA_ENTRIES(int16, int16_t)
DEFINE_FMA_ENTRIES(uint16, uint16_t)
DEFINE_FMA_ENTRIES(int8, int8_t)
DEFINE_FMA_ENTRIES(uint8, uint8_t)
DEFINE_FMA_ENTRIES(bool, bool)
DEFINE_FMA_ENTRIES(half, __half)
DEFINE_FMA_ENTRIES(complex64, hipFloatComplex)
DEFINE_FMA_ENTRIES(complex128, hipDoubleComplex)

// ABI / feature version reported to the Python wrapper.
//   1 -- legacy build (mult only)
//   2 -- batched _nd entry points + diagnostics
//   3 -- + conjugate_<type> and norm_sq_<type> entry points
//   4 -- + fma_mult_<type> entry points (fused a*b + c / a*b - c)
//   5 -- + sparse_mult_<type> entry points (CSR-style sparse product,
//        NVIDIA only; HIP stubs are __builtin_trap() no-ops)
int cayley_abi_version(void) {
  return 5;
}

// Sparse CSR product stubs (ABI-5).  These must exist so that the Python
// wrapper's ctypes lookup does not fail on a HIP build, but the sparse
// kernel itself is NVIDIA-only (see docs/sparse-csr-design.md §7).
SPARSE_STUB(float)
SPARSE_STUB(double)
SPARSE_STUB(half)
SPARSE_STUB(bfloat16)
SPARSE_STUB(complex64)
SPARSE_STUB(complex128)
}
