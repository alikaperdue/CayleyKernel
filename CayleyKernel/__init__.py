import os
import ctypes
import cupy as cp
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KERNEL_FILE = os.path.join(BASE_DIR, "cayleykernel.so")
cuda_lib = ctypes.CDLL(KERNEL_FILE)

# --- 16-bit Float and Complex Types ---
cuda_lib.mult_float.argtypes      = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
cuda_lib.mult_double.argtypes     = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
#cuda_lib.mult_half.argtypes       = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
#cuda_lib.mult_complex64.argtypes  = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
#cuda_lib.mult_complex128.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

# --- Integer Types ---
cuda_lib.mult_int32.argtypes      = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
cuda_lib.mult_int64.argtypes      = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
cuda_lib.mult_uint32.argtypes     = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
cuda_lib.mult_uint64.argtypes     = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
cuda_lib.mult_uint16.argtypes     = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
cuda_lib.mult_int16.argtypes      = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
cuda_lib.mult_uint8.argtypes      = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
cuda_lib.mult_int8.argtypes       = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

# --- Logical Mask Type ---
cuda_lib.mult_bool.argtypes       = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

cuda_lib.mult_float.restype       = None
cuda_lib.mult_double.restype      = None
#cuda_lib.mult_half.restype        = None  # 16-bit Float (FP16 / __half)
#cuda_lib.mult_complex64.restype   = None  # float2 / complex float
#cuda_lib.mult_complex128.restype  = None  # double2 / complex double
cuda_lib.mult_uint64.restype      = None
cuda_lib.mult_int64.restype       = None
cuda_lib.mult_uint32.restype      = None
cuda_lib.mult_int32.restype       = None
cuda_lib.mult_uint16.restype      = None  # unsigned short
cuda_lib.mult_int16.restype       = None  # short
cuda_lib.mult_uint8.restype       = None  # unsigned char
cuda_lib.mult_int8.restype        = None  # signed char
cuda_lib.mult_bool.restype        = None  # bool / mask array

def is_power_of_two(n):
    """Verifies that the horizontal element length is exactly 2^n."""
    return n > 0 and (n & (n - 1)) == 0

def multiply(a, b):
    # 1. Handle incoming type tracking and standardize to CuPy
    return_type = 'cupy'

    if isinstance(a, list):
        a = cp.asarray(a, dtype=cp.float32)
        b = cp.asarray(b, dtype=cp.float32)
        return_type = 'list'
    elif isinstance(a, np.ndarray):
        a = cp.asarray(a)
        b = cp.asarray(b)
        return_type = 'numpy'

    # 2. Structural Shape Verification
    assert a.ndim == 1 or (a.ndim == 2 and a.shape[0] == 1), "Input 'a' must be a flat 1D or 1-row 2D vector"
    assert b.ndim == 1 or (b.ndim == 2 and b.shape[0] == 1), "Input 'b' must be a flat 1D or 1-row 2D vector"
    assert a.shape == b.shape, "Input vectors must have identical dimensions"

    dim = a.size
    if not is_power_of_two(dim):
        raise ValueError(f"Vector length (dim={dim}) must be a power of two (2^n).")

    if a.dtype != b.dtype:
        raise TypeError("Inputs 'a' and 'b' must have matching data types.")

    # Create a clean data-type dispatch mapping
    DTYPE_MAP = {
        'float32'    : cuda_lib.mult_float,
        'float64'    : cuda_lib.mult_double,
        'int32'      : cuda_lib.mult_int32,
        'int64'      : cuda_lib.mult_int64,
        'uint32'     : cuda_lib.mult_uint32,
        'uint64'     : cuda_lib.mult_uint64,
        'int16'      : cuda_lib.mult_int16,
        'uint16'     : cuda_lib.mult_uint16,
        'int8'       : cuda_lib.mult_int8,
        'uint8'      : cuda_lib.mult_uint8,
        'bool_'      : cuda_lib.mult_bool,
        #'float16'   : cuda_lib.mult_half,
        #'complex64' : cuda_lib.mult_complex64,
        #'complex128': cuda_lib.mult_complex128,
    }
    a_dtype = a.dtype.name
    if a_dtype in DTYPE_MAP:
        kernel_func = DTYPE_MAP[a_dtype]
    else:
        raise TypeError(f"Unsupported precision: {a.dtype}! Use standard float, double, complex, integer, or boolean variants.")

    # 4. Allocate matching flat output vector
    o = cp.zeros(dim, dtype=a.dtype)

    # 5. Execute CUDA Pass over the full 2^n layout
    kernel_func(a.data.ptr, b.data.ptr, o.data.ptr, ctypes.c_int(dim))

    # 6. Return Data Formatting matching input type
    if return_type == 'list':
        return cp.asnumpy(o).tolist()
    elif return_type == 'numpy':
        return cp.asnumpy(o)
    else:
        return o

# ==========================================
# QUICK TEST
# ==========================================
# get rid of this... test code don't go here...
def conj(x):
     xstar = -x
     xstar[0] *= -1
     return xstar

def recursion_mult(x, y):
    n = len(x)
    if n == 1:
        return x*y
    m = n // 2
    a, b = x[:m], x[m:]
    c, d = y[:m], y[m:]
    z = cp.zeros(n)
    z[:m] = recursion_mult(a, c) - recursion_mult(conj(d), b)
    z[m:] = recursion_mult(d, a) + recursion_mult(b, conj(c))
    return z

if __name__ == "__main__":
    # Improve these tests...
    # Test with CuPy arrays (float32)
    x = cp.array([1.2, 2.0, 3.0, 4.0], dtype=cp.float32)
    y = cp.array([5.0, 6.0, 7.0, 8.0], dtype=cp.float32)
    
    print("Testing with CuPy arrays...")
    result_gpu = multiply(x, y)
    print("Type:", type(result_gpu))
    print("Result:", result_gpu)
    print("d:", recursion_mult(x,y))

    ## Test with NumPy arrays (float64)
    x_np = np.array([1.0, 2.0], dtype=np.float64)
    y_np = np.array([3.0, 4.0], dtype=np.float64)
    
    print("\nTesting with NumPy arrays...")
    result_cpu = multiply(x_np, y_np)
    print("Type:", type(result_cpu))
    print("Result:", result_cpu)
    print("d:", recursion_mult(cp.asarray(x_np),cp.asarray(y_np)))

