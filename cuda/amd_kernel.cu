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
#include <cstring>
#include <cstdlib>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>

// const int threads_per_block = 256;
const int threads_per_block = 1024;

// Check a HIP API call; abort with a diagnostic on failure.
#define HIP_CHECK(call)                                                                       \
  do {                                                                                        \
    hipError_t _e = (call);                                                                   \
    if (_e != hipSuccess) {                                                                   \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
      abort();                                                                                \
    }                                                                                         \
  } while (0)

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

// Negation helper for the conjugate kernel.  The HIP vendor types
// (hipFloatComplex, hipDoubleComplex, __half) do not define a C++ unary
// minus, so this trait routes them through the vendor's explicit negation.
template <typename T>
struct negation {
  __device__ static __forceinline__ T apply(T a) { return -a; }
};
template <>
struct negation<hipFloatComplex> {
  __device__ static __forceinline__ hipFloatComplex apply(hipFloatComplex a) {
    return make_hipFloatComplex(-a.x, -a.y);
  }
};
template <>
struct negation<hipDoubleComplex> {
  __device__ static __forceinline__ hipDoubleComplex apply(hipDoubleComplex a) {
    return make_hipDoubleComplex(-a.x, -a.y);
  }
};
template <>
struct negation<__half> {
  __device__ static __forceinline__ __half apply(__half a) {
    return __hmul(a, __float2half(-1.0f));
  }
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

  return __popc((!x | !a) ? !a
                : (v < h) ? ~c & h ^ o & -h
                : (d < v) ? r & v ^ o & -v
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
  HIP_CHECK(hipDeviceSynchronize());
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
  HIP_CHECK(hipDeviceSynchronize());
}

// Emits the single fused entry point; the Python wrapper dispatches the same
// symbol with sign = +1 (add) or sign = -1 (subtract).
#define DEFINE_FMA_ENTRIES(SUFFIX, TYPE)                                                \
  void fma_mult_##SUFFIX(const TYPE* a, const TYPE* b, const TYPE* c, TYPE* o, int dim, \
                         long long batch, int sign) {                                   \
    launch_fma_mult<TYPE>(a, b, c, o, dim, batch, sign);                                \
  }

// =============================================================================
// Conjugate  --  out[i] = -a[i] for i > 0,  out[0] = a[0]
// =============================================================================
template <typename T>
__global__ void conjugate_kernel(const T* __restrict__ a, T* __restrict__ out, int dim,
                                 long long batch) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= dim)
    return;

  for (long long item = blockIdx.y; item < batch; item += gridDim.y) {
    const long long pos = item * (long long)dim + index;
    if (index == 0)
      out[pos] = a[pos];
    else
      out[pos] = negation<T>::apply(a[pos]);
  }
}

template <typename T>
static void launch_conjugate(const T* a, T* o, int dim, long long batch) {
  if (dim <= 0 || batch <= 0)
    return;

  int threads = threads_per_block;
  if (dim < threads_per_block) {
    threads = ((dim + 63) / 64) * 64;
    if (threads < 64)
      threads = 64;
  }

  unsigned int grid_x = (unsigned int)((dim + threads - 1) / threads);
  unsigned int grid_y =
      (unsigned int)(batch < (long long)max_grid_y ? batch : (long long)max_grid_y);

  dim3 grid(grid_x, grid_y, 1);
  dim3 block((unsigned int)threads, 1, 1);

  conjugate_kernel<T><<<grid, block>>>(a, o, dim, batch);
  HIP_CHECK(hipDeviceSynchronize());
}

#define DEFINE_CONJ_ENTRIES(SUFFIX, TYPE)                                     \
  void conjugate_##SUFFIX(const TYPE* a, TYPE* o, int dim, long long batch) { \
    launch_conjugate<TYPE>(a, o, dim, batch);                                 \
  }

// =============================================================================
// norm_sq  --  out[b] = sum_i a[b,i] * a[b,i]
// =============================================================================
template <typename T>
__global__ void norm_sq_kernel(const T* __restrict__ a,
                               typename accum_traits<T>::type* __restrict__ out, int dim,
                               long long batch) {
  typedef typename accum_traits<T>::type Acc;
  typedef arithmetic_traits<Acc> Ops;

  int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= batch)
    return;

  for (long long b = item; b < batch; b += (long long)gridDim.x * blockDim.x) {
    const T* row = a + b * (long long)dim;
    Acc acc = Ops::zero();
    for (int i = 0; i < dim; ++i)
      Ops::add(acc, Ops::mul(Acc(row[i]), Acc(row[i])));
    out[b] = acc;
  }
}

template <typename T>
static void launch_norm_sq(const T* a, typename accum_traits<T>::type* o, int dim,
                           long long batch) {
  if (dim <= 0 || batch <= 0)
    return;

  int threads = threads_per_block;
  if (batch < (long long)threads_per_block) {
    threads = ((batch + 63) / 64) * 64;
    if (threads < 64)
      threads = 64;
  }

  unsigned int grid_x = (unsigned int)((batch + threads - 1) / threads);

  dim3 grid(grid_x, 1, 1);
  dim3 block((unsigned int)threads, 1, 1);

  norm_sq_kernel<T><<<grid, block>>>(a, o, dim, batch);
  HIP_CHECK(hipDeviceSynchronize());
}

#define DEFINE_NORMSQ_ENTRIES(SUFFIX, TYPE)                                           \
  void norm_sq_##SUFFIX(const TYPE* a, typename accum_traits<TYPE>::type* o, int dim, \
                        long long batch) {                                            \
    launch_norm_sq<TYPE>(a, o, dim, batch);                                           \
  }

// =============================================================================
// Sparse CSR product  --  ABI-5 symbol family (HIP port)
//
//   sparse_mult_<type>(x_support, x_signs, x_values,
//                      y_support, y_signs, y_values,
//                      out, K_a, K_b, dim, batch)
//
//   CSR layout (row-major contiguous across the batch axis, matching the
//   dense _nd convention):
//     x_support : int64, shape (batch, K_a)  -- padding sentinel: -1
//     x_signs   : int8,  shape (batch, K_a)  -- padding sentinel:  0
//     x_values  : T,     shape (batch, K_a)  -- padding sentinel:  0
//     (same for y)
//     out       : T,     shape (batch, dim)
//
//   Algorithm (docs/sparse-csr-design.md §3):
//     One 32-lane warp per (batch_index, output_index) work item.  The 32 lanes
//     enumerate the K_a*K_b candidate pairs round-robin (pair_id % 32 == lane),
//     each lane evaluates cayley_sign and the signed product term for the
//     pairs it owns, and the warp reduces the partial sums with __shfl_down.
//     Lane 0 writes out[b*dim + i].
//
//   Complexity per work item: O(K_a*K_b / 32 + log 32) — versus the dense
//   kernel's O(dim) per output component.
// =============================================================================

// Warp-shuffle butterfly reduction (32-lane subgroups).  The default template
// works for any scalar accumulator (float, double, int, …).  Complex types need
// component shuffles because __shfl_down operates on 32-bit words.
template <typename Acc>
struct sparse_shuffle_reduce {
  static __device__ __forceinline__ Acc apply(Acc acc) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      acc += __shfl_down(acc, offset, 32);
    return acc;
  }
};
template <>
struct sparse_shuffle_reduce<hipFloatComplex> {
  static __device__ __forceinline__ hipFloatComplex apply(hipFloatComplex acc) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc.x += __shfl_down(acc.x, offset, 32);
      acc.y += __shfl_down(acc.y, offset, 32);
    }
    return acc;
  }
};
template <>
struct sparse_shuffle_reduce<hipDoubleComplex> {
  static __device__ __forceinline__ hipDoubleComplex apply(hipDoubleComplex acc) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc.x += __shfl_down(acc.x, offset, 32);
      acc.y += __shfl_down(acc.y, offset, 32);
    }
    return acc;
  }
};

template <typename T>
__global__ void sparse_mult_kernel(const int64_t* __restrict__ x_support,
                                   const int8_t* __restrict__ x_signs,
                                   const T* __restrict__ x_values,
                                   const int64_t* __restrict__ y_support,
                                   const int8_t* __restrict__ y_signs,
                                   const T* __restrict__ y_values, T* __restrict__ out, int K_a,
                                   int K_b, int dim, long long batch) {
  typedef typename accum_traits<T>::type Acc;
  typedef arithmetic_traits<Acc> Ops;

  const int lane = threadIdx.x % 32;
  const int warp_in_block = threadIdx.x / 32;
  const int warps_per_block = blockDim.x / 32;
  const long long global_warp = (long long)blockIdx.x * warps_per_block + warp_in_block;
  const long long total_warps = (long long)batch * (long long)dim;
  const long long warp_stride = (long long)gridDim.x * warps_per_block;
  const int total_pairs = K_a * K_b;

  for (long long w = global_warp; w < total_warps; w += warp_stride) {
    const long long b = w / (long long)dim;
    const int i = (int)(w - b * (long long)dim);

    const int64_t* xs = x_support + b * (long long)K_a;
    const int8_t* xsg = x_signs + b * (long long)K_a;
    const T* xv = x_values + b * (long long)K_a;
    const int64_t* ys = y_support + b * (long long)K_b;
    const int8_t* ysg = y_signs + b * (long long)K_b;
    const T* yv = y_values + b * (long long)K_b;

    Acc acc = Ops::zero();
    for (int pair_id = lane; pair_id < total_pairs; pair_id += 32) {
      const int ia = pair_id / K_b;
      const int ib = pair_id - ia * K_b;
      const int a = (int)xs[ia];
      const int c = (int)ys[ib];
      if (a < 0 || c < 0)
        continue;  // padding sentinel
      if ((a ^ c) != i)
        continue;  // not this output index
      const int s_ab = (cayley_sign(a, c) == 1) ? 1 : -1;
      const int s_total = (int)xsg[ia] * (int)ysg[ib] * s_ab;
      const Acc term = Ops::mul(Acc(xv[ia]), Acc(yv[ib]));
      s_total > 0 ? Ops::add(acc, term) : Ops::sub(acc, term);
    }

    acc = sparse_shuffle_reduce<Acc>::apply(acc);
    if (lane == 0)
      out[b * (long long)dim + i] = static_cast<T>(acc);
  }
}

template <typename T>
static void launch_sparse(const int64_t* xs, const int8_t* xsg, const T* xv, const int64_t* ys,
                          const int8_t* ysg, const T* yv, T* o, int K_a, int K_b, int dim,
                          long long batch) {
  if (dim <= 0 || batch <= 0 || K_a <= 0 || K_b <= 0)
    return;

  const long long total_warps = (long long)batch * (long long)dim;
  const int threads = threads_per_block;  // 1024 threads = 32 warps of 32
  const long long needed = (total_warps * 32 + threads - 1) / threads;
  unsigned int grid_x =
      (unsigned int)(needed < (long long)max_grid_y ? needed : (long long)max_grid_y);

  dim3 grid(grid_x, 1, 1);
  dim3 block((unsigned int)threads, 1, 1);

  sparse_mult_kernel<T><<<grid, block>>>(xs, xsg, xv, ys, ysg, yv, o, K_a, K_b, dim, batch);
  HIP_CHECK(hipDeviceSynchronize());
}

#define DEFINE_SPARSE_ENTRIES(SUFFIX, TYPE)                                                \
  void sparse_mult_##SUFFIX(const int64_t* xs, const int8_t* xsg, const TYPE* xv,          \
                            const int64_t* ys, const int8_t* ysg, const TYPE* yv, TYPE* o, \
                            int K_a, int K_b, int dim, long long batch) {                  \
    launch_sparse<TYPE>(xs, xsg, xv, ys, ysg, yv, o, K_a, K_b, dim, batch);                \
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

// Fused a*b + c / a*b - c kernels (ABI-4 symbol family).  The HIP build does
// not export BF16 (no hardware bf16 on RDNA3/CDNA), matching the existing
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

// conjugate and norm_sq (ABI-3 symbol family).  Emitted only for the dtypes
// the Python _algebra helpers actually dispatch after their own promotion
// rules (integer/unsigned -> float64, float16/bfloat16 -> float32, complex
// stays as-is).  BF16 is not exported (no hardware bf16 on RDNA3/CDNA).
DEFINE_CONJ_ENTRIES(float, float)
DEFINE_CONJ_ENTRIES(double, double)
DEFINE_CONJ_ENTRIES(complex64, hipFloatComplex)
DEFINE_CONJ_ENTRIES(complex128, hipDoubleComplex)

DEFINE_NORMSQ_ENTRIES(float, float)
DEFINE_NORMSQ_ENTRIES(double, double)
DEFINE_NORMSQ_ENTRIES(complex64, hipFloatComplex)
DEFINE_NORMSQ_ENTRIES(complex128, hipDoubleComplex)

// Sparse CSR product (ABI-5 symbol family).  Exported for the real-float and
// complex dtypes the design doc targets.  BF16 is not exported (no hardware
// bf16 on RDNA3/CDNA).  Integer and bool sparse products are a separate
// research question (see docs/sparse-csr-design.md §7).
DEFINE_SPARSE_ENTRIES(float, float)
DEFINE_SPARSE_ENTRIES(double, double)
DEFINE_SPARSE_ENTRIES(half, __half)
DEFINE_SPARSE_ENTRIES(complex64, hipFloatComplex)
DEFINE_SPARSE_ENTRIES(complex128, hipDoubleComplex)

// ABI / feature version reported to the Python wrapper.
//   1 -- legacy build (mult only)
//   2 -- batched _nd entry points + diagnostics
//   3 -- + conjugate_<type> and norm_sq_<type> entry points
//   4 -- + fma_mult_<type> entry points (fused a*b + c / a*b - c)
//   5 -- + sparse_mult_<type> entry points (CSR-style sparse product)
//
//   The HIP/ROCm build exports all families (mult, fma_mult, conjugate,
//   norm_sq, sparse_mult) for all dtypes EXCEPT bfloat16, which is a
//   CUDA-only capability (no hardware bf16 on RDNA3/CDNA targets).
//   The Python wrapper probes for bf16 symbols with hasattr and falls
//   back gracefully (see docs/building.md).
int cayley_abi_version(void) {
  return 5;
}
}
