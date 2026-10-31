# Building and distributing wheels

[Back to the README](../README.md)

Run these commands from the repository root, with the CUDA toolkit available.

Build a redistributable wheel (compiles the kernel, bundles it, and runs
`auditwheel repair` when available):

```bash
python3 -m pip install build auditwheel
make wheel
```

The kernel is built directly into `CayleyKernel/cuda_kernel.so` and bundled in
the wheel in `dist/`. The wheel is tagged for the build platform (for example,
`py3-none-linux_x86_64`), because its native library uses `ctypes` rather than
the Python extension ABI. To select GPU architectures explicitly, run
`ARCHS="sm_80 sm_89" make wheel`; otherwise the Makefile's defaults apply.
Use `make wheel PYTHON=/path/to/python` to select the build interpreter.

When `auditwheel` is available, repaired wheels are written to `wheelhouse/`.
Repair can fail if required shared libraries or system tools are unavailable;
a platform tag alone does not guarantee manylinux portability or CUDA driver
and GPU compatibility. Validate release wheels on the environments you support.
Publish wheels as release assets or on a package index; keep generated binaries
out of Git.

To run the wheel-build regression tests, install the test's build tools in your
virtual environment and run:

```bash
python3 -m pip install build 'setuptools>=77.0.3' wheel
python3 tests/test_wheel_build.py -v
```

The metadata tests run offline without build isolation and skip with an
explanation when build tools are missing. Normal `make wheel` builds use
isolation and install their build dependencies automatically.

## Release wheels and CPU installations

`.github/workflows/wheels.yml` builds on packaging pull requests, manual
workflow runs, and published GitHub releases. Only the release event uploads
assets to that release, after every build and installation check succeeds.
It produces:

* A portable Python wheel and source distribution for the main package.
* Rust companion wheels for Python 3.10–3.14 on Linux x86_64, macOS arm64
  (`macos-latest`), and Windows x86_64. Linux wheels use Maturin's manylinux
  build environment. These accelerate CPU basis operations; dense multiplication
  still requires the CUDA kernel.
* Linux x86_64 CUDA wheels built with 11.8, 12.0, and 12.1 toolkits, with
  build tags `1cu118`, `1cu120`, and `1cu121`. These target SM 75, 80, 86,
  89, and 90. Select a source build with a newer toolkit for newer GPUs.

The CPU wheel installs without CuPy. For Rust acceleration, download the main
portable wheel and the Rust wheel matching your OS and Python version from
[GitHub Releases](https://github.com/alikaperdue/CayleyKernel/releases) into an
otherwise empty `wheelhouse` directory, then run:

```bash
python -m pip install --find-links=wheelhouse 'cayley-kernel[rust]==0.3.0'
```

The extra requires `cayley-kernel-rust==0.1.0`. Both packages must be available
on the chosen package index or supplied with `--find-links`; GitHub Release
assets are not automatically indexed by pip. The workflow publishes to GitHub
Releases, not PyPI. From a checkout, install `./rust` first and then `'.[rust]'`.
Rust is only needed when building the companion from source.

For GPU use, download exactly the CUDA main wheel you need and install its
explicit filename with the matching optional CuPy dependency, for example:

```bash
python -m pip install './cayley_kernel-0.3.0-1cu118-py3-none-manylinux_2_35_x86_64.whl[cuda11]'
# For a cu120/cu121 wheel, use [cuda12] instead.
```

Use the actual downloaded filename: the manylinux tag is determined by
`auditwheel`, and may differ from this example. CUDA wheel build tags distinguish
release assets; pip does not select CUDA variants based on the installed driver
or GPU. Do not put all CUDA variants into a shared `--find-links` directory.
CuPy is optional and must be selected explicitly for GPU use. The `cuda11` extra
uses CuPy 13 (CUDA 11 support was removed in CuPy 14); `cuda12` uses the CUDA 12
binary distribution. Install only one CuPy distribution in an environment.
See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

`setup.py` is intentionally a thin shim: `pyproject.toml` owns all metadata and
dependencies, while the shim supplies the native ctypes wheel/platform tags.
The release jobs check metadata and install built wheels outside the checkout.
CUDA jobs verify library loading and ABI without GPU hardware; numerical GPU
testing still requires a machine with a supported NVIDIA GPU and driver.
