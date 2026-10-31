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
#   make test-all        default + slow suites (excludes opt-in sparse/timing checks)
#   make benchmark      opt-in timing comparison against BENCHMARK_BASELINE
#   make test-sparse-zd  opt-in CPU sparse zero-divisor validation
#   make zd-showcase     random million-bit zero-divisor confirmations
#   make rust-install    install the optional Rust CPU extension
#   make rust-check      format/lint checks and standalone Rust tests
#   make test-cpu        run CPU tests without requiring a GPU runtime
#   make wheel           build a redistributable wheel (bundles the kernel)
#   make clean           remove build artifacts and compiled binaries

kernel:
NVCC       ?= nvcc
HIPCC      ?= hipcc
CXX        ?= g++
CARGO      ?= cargo
PYTHON     ?= python3

# Compilation flags for native optimizations (Targeting Zen 4 / Ryzen 9)
CC ?= gcc
CFLAGS ?= -O3 -march=native -Wall

KERNEL_SRC ?= cuda/cuda_kernel.cu
AMD_SRC    ?= cuda/amd_kernel.cu
LIB_NAME   ?= CayleyKernel/cuda_kernel.so
SIGN_TOOL  ?= compare_sign_tables

# GPU architectures.  A sensible default covering recent NVIDIA GPUs.
# Override on the command line:  make kernel ARCHS="sm_80 sm_89"
ARCHS ?= sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_110 sm_120

NVCC_GENCODES := $(foreach a,$(ARCHS),-gencode arch=compute_$(subst sm_,,$(a)),code=$(a))

.PHONY: all kernel kernel-amd sign-check test test-all test-sparse-zd zd-showcase wheel clean
.PHONY: rust-install rust-check test-cpu benchmark benchmark-record

# The optional CPU extension is packaged separately; ordinary Python/GPU
# installations never need a Rust toolchain.
rust-install:
	$(PYTHON) -m pip install ./rust

rust-check:
	$(CARGO) fmt --manifest-path rust/Cargo.toml --check
	$(CARGO) clippy --manifest-path rust/Cargo.toml --locked --all-targets --features python -- -D warnings
	$(CARGO) test --manifest-path rust/Cargo.toml --locked

test-cpu:
	$(PYTHON) -m pytest -q -m cpu tests/

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
	$(PYTHON) -m pytest -q -m "not slow" tests/

ZD_LEVELS  ?= 4,8,32,128,1024,4096
ZD_SAMPLES ?= 128
ZD_SEED    ?= 2026
ZD_REPEATS ?= 3
ZD_HUGE_LEVELS ?= 4096,65536,1000000
#ZD_HUGE_LEVELS ?= 4096,65536,1000000,2147483647 #(2**31-1)
ZD_WITNESSES ?= 10

zd-showcase:
	python3 scripts/zd_showcase.py --levels=$(ZD_HUGE_LEVELS) \
	    --witnesses=$(ZD_WITNESSES) --seed=$(ZD_SEED)

test-sparse-zd:
	python3 -m pytest -q -s tests/test_sparse_zd.py --run-sparse-zd \
	    --zd-levels=$(ZD_LEVELS) --zd-samples=$(ZD_SAMPLES) --zd-seed=$(ZD_SEED) \
	    --zd-repeats=$(ZD_REPEATS)

test-all:
	$(PYTHON) -m pytest -q -m "not slow" tests/
	$(PYTHON) -m pytest -q -m slow tests/

BENCHMARK_BASELINE ?= benchmark-results/gpu.json
BENCHMARK_MAX_REGRESSION ?= 0.20

benchmark:
	$(PYTHON) -m pytest -q -s tests/test_benchmark.py --run-benchmarks \
	    --benchmark-baseline "$(BENCHMARK_BASELINE)" \
	    --benchmark-max-regression "$(BENCHMARK_MAX_REGRESSION)"

benchmark-record:
	$(PYTHON) -m pytest -q -s tests/test_benchmark.py --run-benchmarks \
	    --benchmark-record "$(BENCHMARK_BASELINE)"

wheel:
	PYTHON="$(PYTHON)" bash scripts/build_wheel.sh

clean:
	rm -f $(LIB_NAME) $(SIGN_TOOL)
	rm -rf build/ dist/ wheelhouse/ *.egg-info CayleyKernel/*.so
	rm -rf rust/target/ rust/build/ rust/dist/ rust/wheelhouse/ rust/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache/
