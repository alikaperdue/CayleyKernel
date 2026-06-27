## Cuda kernel python wrapper
 * determines Cayley Dickson doubling products 
 * supports 32 and 64 bit inputs size 2^n
 * output equals input size
 * see run_test.py for simple examples

pip install https://github.com/alikaperdue/CayleyKernel/releases/download/v0.1.0/cayleykernel-0.1.0-py3-none-manylinux_2_34_x86_64.whl

```python
from CayleyKernel import multiply
import cupy as cp
a = cp.array([1,2,3,4],dtype=cp.float32)
b = cp.array([1,2,3,4],dtype=cp.float32)
c = multiply(a,b)
```

2026-06-24:
 - added more types 
        cp.int16:      cuda_lib.mult_int16,
        cp.uint16:     cuda_lib.mult_uint16,
        cp.int8:       cuda_lib.mult_int8,
        cp.uint8:      cuda_lib.mult_uint8,
        cp.bool_:      cuda_lib.mult_bool,


