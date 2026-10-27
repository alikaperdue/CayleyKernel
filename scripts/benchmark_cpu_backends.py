#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare public CPU APIs, including Python/Rust integer conversion costs."""
import argparse
import importlib.util
import os
from pathlib import Path
import platform
import random
from statistics import median
from time import perf_counter

LEVELS = 4,8,12,16,20

ROOT = Path(__file__).resolve().parents[1]


def load_backend(name):
    previous = os.environ.get("CAYLEY_KERNEL_CPU_BACKEND")
    try:
        os.environ["CAYLEY_KERNEL_CPU_BACKEND"] = name
        spec = importlib.util.spec_from_file_location(
            "benchmark_" + name, ROOT / "CayleyKernel/_cpu.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            os.environ.pop("CAYLEY_KERNEL_CPU_BACKEND", None)
        else:
            os.environ["CAYLEY_KERNEL_CPU_BACKEND"] = previous


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def measure(function, loops):
    start = perf_counter()
    for _ in range(loops):
        function()
    return perf_counter() - start


def compare(functions, repeats, count):
    loops = []
    for function in functions:
        function()  # warm up separately from timings
        n = 1
        while measure(function, n) < 0.02 and n < 4096:
            n *= 2
        loops.append(n)
    samples = [[], []]
    for round_number in range(repeats):
        for index in ((0, 1) if round_number % 2 == 0 else (1, 0)):
            elapsed = measure(functions[index], loops[index])
            samples[index].append(elapsed * 1e6 / (loops[index] * count))
    return [median(values) for values in samples]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=positive, nargs="+", default=[2**p for p in LEVELS])
    parser.add_argument("--pairs", type=positive, default=128)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        backends = [load_backend(name) for name in ("python", "rust")]
    except ImportError as error:
        parser.error(str(error))
    print(f"{platform.platform()}; Python {platform.python_version()}; seed={args.seed}")
    print(f"pairs={args.pairs}; repeats={args.repeats}; median microseconds per pair")
    print(f"{'bits':>9} {'operation':>12} {'Python us':>12} {'Rust us':>12} {'speedup':>9}")
    rng = random.Random(args.seed)
    for bits in args.bits:
        pairs = [(rng.getrandbits(bits), rng.getrandbits(bits)) for _ in range(args.pairs)]
        labels = [((a + 1) * (-1 if i % 2 else 1), b + 1)
                  for i, (a, b) in enumerate(pairs)]
        for operation in ("twist", "basis", "twist_many", "basis_many"):
            inputs = labels if operation.startswith("basis") else pairs
            functions = []
            for backend in backends:
                fn = getattr(backend, operation)
                if operation.endswith("_many"):
                    functions.append(lambda fn=fn: fn(inputs))
                else:
                    functions.append(lambda fn=fn: [fn(a, b) for a, b in inputs])
            if functions[0]() != functions[1]():
                raise AssertionError(f"backend mismatch: bits={bits}, operation={operation}")
            python_us, rust_us = compare(functions, args.repeats, args.pairs)
            print(f"{bits:9d} {operation:>12} {python_us:12.3f} {rust_us:12.3f} "
                  f"{python_us / rust_us:8.2f}x")


if __name__ == "__main__":
    main()
