#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
#
# scripts/build_wheel.sh -- build a redistributable CayleyKernel wheel.
#
# Flow:
#   1. Compile the CUDA kernel directly into CayleyKernel/cuda_kernel.so.
#   2. Build a wheel with PEP 517 tooling.
#   3. Optionally repair the wheel for manylinux portability (auditwheel).
#
# Requirements: nvcc on PATH, python3 with `build`, and (optionally)
# `auditwheel`.  See README.md -> "Building from source".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# GPU architectures to target.  Override with: ARCHS="sm_80" ./scripts/build_wheel.sh
# Leave the Makefile's default architecture list intact unless overridden.
make_args=()
if [[ -n "${ARCHS:-}" ]]; then
    make_args+=("ARCHS=$ARCHS")
else
    unset ARCHS
fi
PYTHON="${PYTHON:-python3}"
KERNEL="$ROOT/CayleyKernel/cuda_kernel.so"

echo "==> Compiling CUDA kernel..."
make kernel "LIB_NAME=$KERNEL" "${make_args[@]}"

if [[ ! -f "$KERNEL" ]]; then
    echo "ERROR: kernel build did not produce $KERNEL" >&2
    exit 1
fi

echo "==> Building wheel..."
"$PYTHON" -m build --wheel

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
