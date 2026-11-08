# SPDX-License-Identifier: Apache-2.0
"""Error budgets for approximate product checks.

Keep exact integer, boolean, basis, and constructed rounding checks as
explicit equality assertions. A floating comparison policy only sets the
error budget. Tests still choose the reference precision and any final cast.
"""

import numpy as np

_FLOAT64 = (1e-10, 1e-12)
_HALF = (2e-3, 5e-4)

_TOLERANCES = {
    # Random dense products and comparisons of batched and single GPU paths.
    "dense": {"float32": (1e-4, 1e-5), "float64": _FLOAT64},
    # Complex coefficients, including bilinearity and the complex GPU ABI.
    "complex": {"complex64": (2e-5, 2e-5), "complex128": (1e-12, 1e-12)},
    # Random half products compared after rounding the reference to half.
    "rounded_half": {"float16": _HALF},
    # A fused product can differ from separately rounded multiply and add.
    "fused": {
        "float16": _HALF,
        "float32": (1e-5, 1e-6),
        "float64": _FLOAT64,
        "complex64": (1e-4, 1e-5),
        "complex128": _FLOAT64,
    },
    # Framework adapters compared with NumPy products on the host.
    "interop": {"float32": (1e-5, 1e-5), "float64": _FLOAT64, "complex64": (1e-5, 1e-5)},
    # Rescaled paper fixtures, including subnormal half coefficients.
    "typed_replay": {
        "float16": (1e-3, float(np.nextafter(np.float16(0), np.float16(1)))),
        "complex64": (2e-5, 2e-5),
        "complex128": (1e-12, 1e-12),
    },
}


def assert_product_close(
    actual, expected, dtype, *, contract, xp=np, override=None, reason=None, err_msg=""
):
    """Compare under a named contract for the storage dtype.

    An override is an (rtol, atol) pair and requires a reason at the call site.
    Pass the array backend when checking values that are still on the device.
    """
    rtol, atol = _TOLERANCES[contract][np.dtype(dtype).name]
    if override is not None:
        if not reason:
            raise ValueError("a tolerance override requires a reason")
        rtol, atol = override
    xp.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=err_msg)
