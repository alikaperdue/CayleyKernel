"""Smoke checks for release artifacts installed in an isolated environment."""

import argparse
import ctypes
import importlib.util
import sys
from pathlib import Path

import numpy as np

import CayleyKernel as ck


def check_cpu(backend):
    assert ck.cpu_backend() == backend, ck.cpu_backend()
    assert ck.basis(2, 2) == -1
    assert ck.twist(1, 1) == -1
    np.testing.assert_array_equal(ck.cpu_multiply([1, 2], [3, 4]), [-5, 10])
    assert ck.library_path() is None
    assert not ck.loaded()
    assert ck.abi_version() == 0
    assert not (Path(ck.__file__).parent / "cuda_kernel.so").exists()


def check_rust(prefix):
    import cayley_kernel_rust as native

    assert Path(native.__file__).resolve().is_relative_to(prefix), native.__file__
    # A companion without this entry point would silently use Python for dense products.
    assert callable(native.multiply)
    check_cpu("rust")


def check_cuda():
    kernel = Path(ck.__file__).resolve().with_name("cuda_kernel.so")
    # Open directly first so missing shared dependencies produce a loader diagnostic.
    library = ctypes.CDLL(str(kernel))
    assert ck.loaded()
    assert Path(ck.library_path()).resolve() == kernel
    assert ck.abi_version() == 5, ck.abi_version()
    for symbol in ("mult_float_nd", "fma_mult_float", "sparse_mult_float"):
        getattr(library, symbol)


def check_gpu():
    import cupy as cp

    assert cp.cuda.runtime.getDeviceCount() > 0
    a = cp.asarray([[1, 2], [0, 1]], dtype=cp.float32)
    b = cp.asarray([[3, 4], [0, 1]], dtype=cp.float32)
    result = ck.multiply(a, b)
    assert isinstance(result, cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(result), [[-5, 10], [-1, 0]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("python", "rust", "cuda-load", "cuda-execute"))
    mode = parser.parse_args().mode
    prefix = Path(sys.prefix).resolve()
    assert Path(ck.__file__).resolve().is_relative_to(prefix), ck.__file__
    if mode == "python":
        assert importlib.util.find_spec("cayley_kernel_rust") is None
        check_cpu("python")
    elif mode == "rust":
        check_rust(prefix)
    else:
        check_cuda()
        if mode == "cuda-execute":
            check_gpu()
    print(f"{mode} smoke passed using {ck.__file__}")


if __name__ == "__main__":
    main()
