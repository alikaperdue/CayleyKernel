# Building and distributing wheels

[Back to the README](../README.md)

## Build from a source checkout

For the portable Python wheel and source distribution, start from a clean
checkout without a compiled `CayleyKernel/cuda_kernel.so` or a previous
`build/` directory. Run from the repository root:

```bash
python -m pip install build
python -m build
```

The outputs are a `py3-none-any` wheel and a source archive in `dist/`. The
archive builds the portable package. Use the Git checkout for the CUDA and
Rust sources. An editable CPU install from a checkout uses
`python -m pip install -e .`. To build and install the optional companion
from source, use a Rust toolchain and run `python -m pip install ./rust` followed by
`python -m pip install -e '.[rust]'`.

To build a CUDA wheel, use a separate checkout or start with empty `dist/`
and `wheelhouse/` directories. With `nvcc` from a compatible CUDA toolkit on
`PATH`, run:

```bash
python3 -m pip install build auditwheel 'patchelf>=0.14.5'
make wheel
```

The kernel is built directly into `CayleyKernel/cuda_kernel.so` and bundled in
the wheel in `dist/`. The wheel is tagged for the build platform (for example,
`py3-none-linux_x86_64`), because its native library uses `ctypes` rather than
the Python extension ABI. To select GPU architectures explicitly, run
`ARCHS="sm_80 sm_89" make wheel`; otherwise the Makefile's defaults apply.
Use `make wheel PYTHON=/path/to/python` to select the build interpreter.

When `auditwheel` is available, repaired wheels are written to `wheelhouse/`.
Repair can fail if required shared libraries or system tools are unavailable.
A repaired wheel's manylinux tag specifies its Linux platform baseline.
CUDA driver and GPU compatibility require separate checks. Keep generated
binaries out of Git.

To run the wheel-build regression tests, install the test's build tools in your
virtual environment and run:

```bash
python3 -m pip install build 'setuptools>=77.0.3' wheel
python3 -m tests.integration.test_wheel_build -v
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
  (`macos-latest`), and Windows x86_64. The matrix uses standard CPython builds.
  Linux wheels use Maturin's manylinux
  build environment. These accelerate CPU basis operations and dense real
  multiplication through `cpu_multiply`.
* Linux x86_64 CUDA wheels built with 11.8, 12.0, and 12.1 toolkits, with
  build tags `1cu118`, `1cu120`, and `1cu121`. These target SM 75, 80, 86,
  89, and 90. For a GPU outside this set, build from source with a toolkit
  and `ARCHS` setting that support your device.

Download the portable wheel from
[GitHub Releases](https://github.com/alikaperdue/CayleyKernel/releases).
It installs NumPy, which is sufficient for `cpu_multiply` with the Python
backend. In the download directory, run:

```bash
python -m pip install ./cayley_kernel-0.3.0-py3-none-any.whl
```

For Rust acceleration, also download the companion wheel matching your OS,
architecture, and CPython version. For example, with exactly one downloaded
companion in the directory:

```bash
python -m pip install --only-binary=:all: \
  './cayley_kernel-0.3.0-py3-none-any.whl[rust]' ./cayley_kernel_rust-0.1.0-*.whl
```

The extra requires `cayley-kernel-rust==0.1.0`. Supplying both filenames
selects the downloaded artifacts. Pip can still resolve NumPy from the
configured index. With the default backend selection, `CayleyKernel.cpu_backend()`
returns `"rust"` when the extension imports successfully.

For GPU use, download the CUDA main wheel for your environment and install its
filename with the matching CuPy extra. Substitute the full downloaded filename:

```bash
python -m pip install --only-binary=:all: './<downloaded-cu118-filename>.whl[cuda11]'
# For a cu120/cu121 wheel, use [cuda12] instead.
```

The wheel's manylinux tag is set by `auditwheel`. Choose the file yourself
since pip ranks wheel tags without inspecting the driver or GPU. For
`cu118`, use `cuda11` with CPython 3.10–3.13 for the
available CuPy 13 binaries. The `cuda12` extra selects `cupy-cuda12x` for
`cu120` and `cu121`. Install one CuPy distribution and provide its compatible
CUDA runtime and driver. The prebuilt CayleyKernel wheel already contains the
compiled kernel.
See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

Both host and device inputs to `multiply` require a compatible GPU, driver,
CuPy build, and compiled kernel. For experimental AMD support, build with
`make kernel-amd` using `hipcc` and use a ROCm-compatible CuPy environment.
The HIP build writes the same `CayleyKernel/cuda_kernel.so` used by CUDA,
replacing any existing build at that path. Restart Python after switching
builds. HIP supports batched dense products and omits BF16 and CUDA block-size
tuning.

`pyproject.toml` declares package metadata and dependencies. `setup.py`
assigns platform tags to wheels containing the native ctypes library.

## Verify a release candidate

Run the **Release wheels** workflow manually and download its artifacts. The
portable and Rust jobs check metadata, install the wheels, and run the same
known dense CPU product outside the checkout. The Rust matrix also checks
automatic backend selection and the companion's dense entry point. The CUDA
build jobs repair and check wheel metadata in the toolkit containers. A
separate hosted Linux job installs every repaired wheel in a fresh environment,
loads the packaged library, and checks ABI 5 and representative symbols.
It then resolves each wheel's CuPy extra for Python 3.12.

To run a numerical check, set the repository variable `CI_GPU_LABEL` to the
existing Linux x86_64 NVIDIA runner label and choose `cu118`, `cu120`, or
`cu121` for the manual run's `gpu_variant` input. The runner needs a compatible
GPU, driver, and runtime for the selected CuPy extra. That job installs the
selected artifact with its extra and checks a batched FP32 product on the
device. Repeat for each variant on a compatible runner. The default
`gpu_variant=none` runs the host checks.

To reproduce a check locally, create a fresh virtual environment outside the
checkout, install the candidate as shown above, and run:

```bash
python -m pip check
python -I /path/to/CayleyKernel/tests/integration/wheel_smoke.py python
```

Use `rust` for the portable wheel with its companion, `cuda-load` for the
CUDA loader and ABI check, or `cuda-execute` with CuPy and the GPU stack
available. Run `python -m twine check --strict /path/to/artifacts/*` in your
build environment to check the candidate distribution metadata.

## Package-index distribution

The existing release workflow attaches files to GitHub Releases. On an index
that hosts these distributions, `python -m pip install cayley-kernel` installs
the main package (declared as `cayley_kernel`). Once both projects are available
on the index, `python -m pip install 'cayley-kernel[rust]'` also resolves
`cayley-kernel-rust`. For a future PyPI release, publish the portable
distributions and the Rust wheels there. Keep the CUDA variants on GitHub
Releases for explicit file selection because they share the main distribution
name and version.

Before a first PyPI upload, verify that the project maintainers control both
normalized names in PyPI. The main package metadata identifies Yotta Space
Corp as maintainer. Verify ownership in the PyPI account for each project
independently of the metadata and GitHub repository access. For an unclaimed
name, configure a pending Trusted Publisher. For an existing project, a
PyPI project owner must authorize the publisher.
See [creating a project through Trusted Publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [adding a publisher to an existing project](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

Register `alikaperdue/CayleyKernel`, the exact publishing workflow filename,
and the GitHub environment if used, for each project. A future publishing
job needs `id-token: write` to exchange GitHub's OIDC token and should upload
only the verified files for the corresponding project. TestPyPI has separate
project ownership and publisher configuration. The manual candidate workflow
and local smoke commands above verify the files without an upload.
See [publishing with a Trusted Publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
