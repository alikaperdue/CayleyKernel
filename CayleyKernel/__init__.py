import ctypes
import os
import cupy as cp
import numpy as np
#import Base
#import CayleyDickson as cd

KERNEL_FILE = './cayleykernel.so'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(BASE_DIR, KERNEL_FILE)
cuda_lib = ctypes.CDLL(lib_path)

cuda_lib.c_cayley_mult_float.argtypes  = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
cuda_lib.c_cayley_mult_double.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

cuda_lib.c_cayley_mult_float.restype   = None
cuda_lib.c_cayley_mult_double.restype  = None

def multiply(a, b):
    """
    Unified entry point for Cayley-Dickson parallel multiplication via .so library.
    Handles NumPy/CuPy inputs and 32/64-bit precision dynamically.
    """
    return_type = 'cupy'
    if not isinstance(a, cp.ndarray):
        a = cp.asarray(a)
        b = cp.asarray(b)
        return_type = 'numpy'
    if a.dtype != b.dtype:
        raise TypeError("Inputs 'a' and 'b' must have matching data types.")
    dim = a.size
    if a.dtype == cp.float32:
        kernel_func = cuda_lib.c_cayley_mult_float
    elif a.dtype == cp.float64:
        kernel_func = cuda_lib.c_cayley_mult_double
    else:
        raise TypeError("Unsupported precision! Use float32 or float64.")
    o = cp.zeros(dim, dtype=a.dtype)
    kernel_func(a.data.ptr, b.data.ptr, o.data.ptr, ctypes.c_int(dim))
    return cp.asnumpy(o) if return_type == 'numpy' else o

# ==========================================
# QUICK TEST
# ==========================================
if __name__ == "__main__":
    # Improve these tests...
    # Test with CuPy arrays (float32)
    x = cp.array([1.2, 2.0, 3.0, 4.0], dtype=cp.float32)
    y = cp.array([5.0, 6.0, 7.0, 8.0], dtype=cp.float32)
    
    print("Testing with CuPy arrays...")
    result_gpu = multiply(x, y)
    print("Type:", type(result_gpu))
    print("Result:", result_gpu)
    #print("d:", cd.mult(x,y))

    ## Test with NumPy arrays (float64)
    x_np = np.array([1.0, 2.0], dtype=np.float64)
    y_np = np.array([3.0, 4.0], dtype=np.float64)
    
    print("\nTesting with NumPy arrays...")
    result_cpu = multiply(x_np, y_np)
    print("Type:", type(result_cpu))
    print("Result:", result_cpu)
    #print("d:", cd.mult(cp.asarray(x_np),cp.asarray(y_np)))

