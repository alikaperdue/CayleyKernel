#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
#
# scripts/build_wheel.sh -- build a redistributable CayleyKernel wheel.
#
# Flow:
#   1. Compile the CUDA kernel (nvcc) into cuda_kernel.so.
#   2. Copy the kernel into the Python package so setuptools bundles it.
#   3. Build a sdist+wheel with PEP 517 tooling.
#   4. Optionally repair the wheel for manylinux portability (auditwheel).
#
# Requirements: nvcc on PATH, python3 with `build`, and (optionally)
# `auditwheel`.  See README.md -> "Building from source".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# GPU architectures to target.  Override with: ARCHS="sm_80" ./scripts/build_wheel.sh
ARCHS="${ARCHS:-}"

echo "==> Compiling CUDA kernel..."
make kernel ARCHS="$ARCHS"

if [[ ! -f "$ROOT/cuda_kernel.so" ]]; then
    echo "ERROR: kernel build did not produce cuda_kernel.so" >&2
    exit 1
fi

echo "==> Staging kernel into package..."
cp "$ROOT/cuda_kernel.so" "$ROOT/CayleyKernel/cuda_kernel.so"

echo "==> Building wheel..."
python3 -m build --wheel

echo "==> Done. Wheel(s) in dist/:"
ls -1 dist/*.whl

# Optional manylinux repair (only if auditwheel is available).
if command -v auditwheel >/dev/null 2>&1; then
    echo "==> Running auditwheel repair..."
    mkdir -p wheelhouse
    auditwheel repair dist/*.whl -w wheelhouse/
    echo "==> Repaired wheel(s) in wheelhouse/:"
    ls -1 wheelhouse/*.whl
fi
