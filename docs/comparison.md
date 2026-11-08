# Positioning and related libraries

[Back to the README](../README.md)

CayleyKernel computes the **Cayley-Dickson doubling product** for any `2^n`-dimensional
hypercomplex algebra on the GPU. It is not a rotation library, not a quaternion-specific
package, and not a general-purpose linear-algebra framework. This page positions it
relative to the most common alternatives so that you can pick the right tool quickly.

## What CayleyKernel is and is not

| Aspect | CayleyKernel |
|---|---|
| Dimensions | Any `2^n`: 1, 2, 4, 8, 16, 32, …, `2^22`+ |
| Algebraic structure | Raw CD doubling — noncommutative (≥ 4), nonassociative (≥ 8) |
| Norm | Arbitrary; no unit-norm constraint. `norm_sq` is a helper, not an invariant |
| Batch | `(B, …, dim)` tensor in a single kernel launch (grid-stride) |
| Runtime | NVIDIA CUDA (primary), AMD HIP (experimental), CPU reference (NumPy / optional Rust) |
| Typical use | Batched hypercomplex arithmetic, algebra research, sign-law validation |

## Comparison with `scipy.spatial.transform.Rotation`

`scipy.spatial.transform.Rotation` represents **unit quaternions** in `SO(3)` (3-D
rotations). It is the most common "quaternion math" in scientific Python.

| | CayleyKernel | `scipy … Rotation` |
|---|---|---|
| Dimension | Any `2^n` | 4 (quaternions only) |
| Norm | Arbitrary | Unit-norm enforced |
| Group | None (raw bilinear CD product) | `SO(3)` / `SE(3)` |
| Batch | `(B, …, dim)` tensor, GPU | CPU, element-wise or vectorised NumPy |
| Nonassociative dims | Yes (≥ 8) | N/A (quaternions only) |
| Backend | CUDA / HIP GPU | CPU (NumPy) |
| Extras | `conjugate`, `norm_sq`, `inverse`, `exp`, `twist`, `basis` | `as_matrix`, `as_euler`, `as_rotvec`, composition, interpolation |

They are **not interchangeable**. A quaternion product via CayleyKernel at `dim = 4`
produces the same bilinear result as the raw quaternion multiplication that underlies
`Rotation`, for the same four real coefficients. CayleyKernel does **not** normalize,
does **not** constrain to the unit sphere, and does **not** expose `as_matrix()`,
`as_euler()`, or Slerp.

### Equivalence snippet (dim = 4, quaternion)

```python
import numpy as np

# ---- CayleyKernel (CPU path; swap to GPU multiply on a CUDA machine) ----
from CayleyKernel import cpu_multiply

a = np.array([1.0, 2.0, 3.0, 4.0])  # coefficient order (e0, e1, e2, e3) = (w, x, y, z)
b = np.array([5.0, 6.0, 7.0, 8.0])
ck = cpu_multiply(a, b)  # raw bilinear Hamilton product, dim 4


# ---- Reference: hand-rolled bilinear Hamilton product (no normalisation) ----
# Same (w, x, y, z) order; this is what the `quaternion` PyPI package returns.
def hamilton(p, q):
    return np.array(
        [
            p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
            p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
            p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
            p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
        ]
    )


ref = hamilton(a, b)
np.testing.assert_allclose(ck, ref)  # exact bilinear agreement, no normalisation

# ---- scipy: quaternion product via Rotation (SO(3), unit-normalised) ----
from scipy.spatial.transform import Rotation

# scipy uses (x, y, z, w) ordering; reorder from (w, x, y, z)
q_a = Rotation.from_quat([a[1], a[2], a[3], a[0]])
q_b = Rotation.from_quat([b[1], b[2], b[3], b[0]])
q_c = q_a * q_b  # group composition, then normalised to unit length
scipy = np.array([q_c.as_quat()[3], q_c.as_quat()[0], q_c.as_quat()[1], q_c.as_quat()[2]])
# scipy result == ck result / (|a| * |b|), i.e. equal direction, unit magnitude.
np.testing.assert_allclose(scipy, ck / (np.linalg.norm(a) * np.linalg.norm(b)))
```

> **Note:** `scipy`'s `Rotation` composes in the `SO(3)` group and **normalises** the
> result to a unit quaternion, so its coefficients equal CayleyKernel's bilinear product
> divided by `|a| * |b|` (same direction, unit magnitude). The `quaternion` PyPI package
> and the hand-rolled `hamilton()` above return the **raw** bilinear product and match
> CayleyKernel exactly for the same `(w, x, y, z)` coefficient order. CayleyKernel neither
> normalises nor exposes `as_matrix()`, `as_euler()`, or Slerp.

## Comparison with the `quaternion` PyPI package

| | CayleyKernel | `quaternion` (PyPI) |
|---|---|---|
| Dimension | Any `2^n` | 4 (quaternions only) |
| Container | NumPy / CuPy arrays, lists | Python `quaternion.quaternion` objects |
| Backend | GPU (CUDA) or CPU | CPU only (pure Python / NumPy) |
| Batch | `(B, …, dim)` tensor in one launch | Element-wise Python objects |
| Use case | High-throughput batch, algebra research | Interactive quaternion arithmetic |

The `quaternion` package is a clean, readable, pure-Python quaternion library. It is
not a GPU batch library and does not support dimensions beyond 4.

## Comparison with the in-repo recursive reference

`recursive_product` in `tests/support/oracles.py` evaluates the
Cayley–Dickson doubling recurrence independently of the production sign law.
It accepts NumPy or CuPy arrays with broadcastable batch axes. Tests select
the calculation dtype and apply any final storage cast separately.

Both the recursive oracle and GPU kernel evaluate `O(dim²)` coefficient
products. The oracle recursively combines four half-size products, while
the GPU kernel uses the constant-time sign law and computes the output
components in parallel.

## When to use what

| Need | Recommended |
|---|---|
| 3-D rotation in `SO(3)`, unit quaternions, Euler angles, Slerp | `scipy.spatial.transform.Rotation` |
| Interactive quaternion arithmetic, dim 4, pure Python | `quaternion` PyPI package |
| Batched `2^n`-dim products, arbitrary dimension, GPU throughput | **CayleyKernel** |
| Validating a new CD-product implementation | `tests/support/oracles.py` as oracle + CayleyKernel as target |
| Nonassociative algebra research (octonions, sedenions, 32-dim, …) | **CayleyKernel** |
| Complex-number arithmetic (dim 2) | `numpy` / `cupy` (no special library needed) |
