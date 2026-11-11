"""CPU only: (1 + 2e1)(3 + 4e1) = -5 + 10e1 in the octonions."""

import numpy as np

import CayleyKernel as ck

a = [1, 2, 0, 0, 0, 0, 0, 0]
b = [3, 4, 0, 0, 0, 0, 0, 0]
result = ck.cpu_multiply(a, b)
np.testing.assert_array_equal(result, [-5, 10, 0, 0, 0, 0, 0, 0])
assert result.dtype == np.float64
assert ck.twist(1, 1) == -1
assert ck.basis(2, 2) == -1
print("CPU octonion:", result.tolist())
