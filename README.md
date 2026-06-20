## Cuda kernel python wrapper
 * determines Cayley Dickson doubling products 
 * supports 32 and 64 bit inputs size 2^n
 * output equals input size
 * see run_test.py for simple examples

```python
from CayleyKernel import multiply
import cupy as cp
a = cp.array([1,2,3,4],dtype=cp.float32)
b = cp.array([1,2,3,4],dtype=cp.float32)
c = multiply(a,b)
```
