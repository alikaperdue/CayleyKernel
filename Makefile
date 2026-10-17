# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
#
# CayleyKernel Makefile
#
# Targets:
#   make kernel          compile the CUDA kernel -> cuda_kernel.so
#   make kernel-amd      compile the HIP/ROCm kernel (AMD)
#   make sign-check      host-side equivalence test of the sign function
#   make test            run the default pytest suite
#   make test-all        default + slow + benchmark suites
#   make wheel           build a redistributable wheel (bundles the kernel)
#   make clean           remove build artifacts and compiled binaries

kernel:
NVCC       ?= nvcc
HIPCC      ?= hipcc
CXX        ?= g++

KERNEL_SRC ?= cuda/cuda_kernel.cu
AMD_SRC    ?= cuda/amd_kernel.cu
LIB_NAME   ?= CayleyKernel/cuda_kernel.so
SIGN_TOOL  ?= compare_sign_tables

# GPU architectures.  A sensible default covering recent NVIDIA GPUs.
# Override on the command line:  make kernel ARCHS="sm_80 sm_89"
ARCHS ?= sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_110 sm_120

NVCC_GENCODES := $(foreach a,$(ARCHS),-gencode arch=compute_$(subst sm_,,$(a)),code=$(a))

.PHONY: all kernel kernel-amd sign-check test test-all wheel clean

all: kernel

kernel:

	$(NVCC) -O3 -Xcompiler -fPIC -shared \
        $(NVCC_GENCODES) \
	    -o $(LIB_NAME) $(KERNEL_SRC)
	@echo "Built $(LIB_NAME)"

# Single-arch quick build (e.g. for local iteration):
#   make kernel ARCHS="sm_86"

kernel-amd:
	$(HIPCC) -O3 -fPIC --shared \
	    -o $(LIB_NAME) $(AMD_SRC)
	@echo "Built $(LIB_NAME) (HIP/ROCm)"

sign-check:
	$(CXX) -O2 -std=c++17 cuda/tools/compare_sign_tables.cpp -o $(SIGN_TOOL)
	./$(SIGN_TOOL) 16

test:
	python3 -m pytest -q tests/

test-all:
	python3 -m pytest -q tests/
	python3 -m pytest -q -m slow tests/
	python3 -m pytest -q -m benchmark tests/

wheel:
	bash scripts/build_wheel.sh

clean:
	rm -f $(LIB_NAME) $(SIGN_TOOL)
	rm -rf build/ dist/ wheelhouse/ *.egg-info CayleyKernel/*.so
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache/
