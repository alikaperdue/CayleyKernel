"""Requires NVIDIA GPU/driver, compatible CuPy, and built CUDA kernel.

See ../SKILL.md for installation. CPU CI checks syntax, not GPU execution.
"""

import cupy as cp

import CayleyKernel as ck

# Broadcast one octonion against 1024 independent octonions.
a = cp.zeros((1024, 8), dtype=cp.float32)
a[:, 0] = 1
a[:, 1] = 2
b = cp.array([3, 4, 0, 0, 0, 0, 0, 0], dtype=cp.float32)
result = ck.multiply(a, b)
expected = cp.broadcast_to(cp.array([-5, 10, 0, 0, 0, 0, 0, 0], dtype=cp.float32), a.shape)
cp.testing.assert_array_equal(result, expected)
assert result.shape == (1024, 8)
cp.cuda.get_current_stream().synchronize()
print("GPU batching passed:", result.shape)
