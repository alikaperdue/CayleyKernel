"""CPU backend parity, independent oracles, and optional-install behavior."""
import importlib.util
import itertools
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["python", "rust"])
def cpu(request, monkeypatch):
    if request.param == "rust" and importlib.util.find_spec("cayley_kernel_rust") is None:
        if os.environ.get("CAYLEY_TEST_REQUIRE_RUST") == "1":
            pytest.fail("Rust extension is required for this test run")
        pytest.skip("optional Rust extension is not installed")
    monkeypatch.setenv("CAYLEY_KERNEL_CPU_BACKEND", request.param)
    spec = importlib.util.spec_from_file_location("cpu_under_test", ROOT / "CayleyKernel/_cpu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.cpu_backend() == request.param
    return module


def doubling_sign(a, b):
    sign = 1
    for bit in range(max(a.bit_length(), b.bit_length()) - 1, -1, -1):
        half = 1 << bit
        upper_a, upper_b = bool(a & half), bool(b & half)
        a, b = a & (half - 1), b & (half - 1)
        if upper_a:
            if b:
                sign = -sign
            if upper_b:
                sign = -sign
        if upper_b:
            a, b = b, a
    return sign


def test_exhaustive_signed_basis(cpu):
    for a, b in itertools.product(range(64), repeat=2):
        expected = doubling_sign(a, b)
        assert cpu.twist(a, b) == expected
        for sa, sb in itertools.product((-1, 1), repeat=2):
            assert cpu.basis(sa * (a + 1), sb * (b + 1)) == sa * sb * expected * ((a ^ b) + 1)


def test_word_boundaries_and_random_bigints(cpu):
    rng = random.Random(2026)
    for bits in (31, 32, 63, 64, 65, 127, 128, 129, 1024, 4096):
        top = 1 << (bits - 1)
        pairs = [(0, top), (top, 0), (top, top), (top, top - 1),
                 ((1 << bits) - 1, top), (top + 1, top + 2),
                 (top + 1, 1), (top + 1, (top << 1) + 1), (top, top << 1)]
        pairs += [(rng.getrandbits(bits), rng.getrandbits(bits)) for _ in range(16)]
        expected = [doubling_sign(a, b) for a, b in pairs]
        assert cpu.twist_many(iter(pairs)) == expected
        assert [cpu.twist(a, b) for a, b in pairs] == expected
        labels = [(a + 1, -(b + 1)) for a, b in pairs]
        assert cpu.basis_many(labels) == [-s * ((a ^ b) + 1) for (a, b), s in zip(pairs, expected)]


def test_million_bit_indices(cpu):
    # Embedding small basis indices by a common bit shift preserves the sign.
    pairs = [(a << 1_000_000, b << 1_000_000)
             for a, b in ((0, 3), (3, 3), (3, 6), (6, 3), (7, 4))]
    assert cpu.twist_many(pairs) == [doubling_sign(a, b)
                                    for a, b in ((0, 3), (3, 3), (3, 6), (6, 3), (7, 4))]
    rng = random.Random(81)
    a, b = rng.getrandbits(1_000_000), rng.getrandbits(1_000_000)
    assert cpu.twist(a, b) == cpu._python_twist(a, b)
    assert cpu.basis(-(a + 1), b + 1) == cpu._python_basis(-(a + 1), b + 1)


def test_input_contract(cpu):
    class Index:
        def __index__(self):
            return 3

    assert cpu.twist(Index(), 6) == doubling_sign(3, 6)
    assert cpu.basis(Index(), -7) == cpu.basis(3, -7)
    assert cpu.twist_many([]) == []
    assert cpu.basis_many(iter(())) == []
    for fn in (cpu.twist, cpu.basis):
        for value in (1.5, "1", None):
            with pytest.raises(TypeError):
                fn(value, 1)
    with pytest.raises(ValueError, match="nonnegative"):
        cpu.twist(-1, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        cpu.twist_many([(0, 0), (1, -1)])
    for r, c in ((0, 1), (1, 0), (0, 0)):
        with pytest.raises(ValueError, match="nonzero"):
            cpu.basis(r, c)
        with pytest.raises(ValueError, match="nonzero"):
            cpu.basis_many([(1, 1), (r, c)])


@pytest.mark.parametrize("backend, succeeds", [("auto", True), ("python", True),
                                               ("rust", False), ("invalid", False)])
def test_import_without_optional_dependencies(backend, succeeds):
    code = """
import sys
for name in ('cayley_kernel_rust', 'numpy', 'cupy'):
    sys.modules[name] = None
import CayleyKernel as ck
assert ck.cpu_backend() == 'python'
assert ck.twist(3, 6) in (-1, 1)
assert ck.basis_many([(1, 1)]) == [1]
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            env={**os.environ, "CAYLEY_KERNEL_CPU_BACKEND": backend},
                            capture_output=True, text=True)
    assert (result.returncode == 0) == succeeds, result.stderr
    if backend == "rust":
        assert "Rust CPU backend requested but unavailable" in result.stderr
    if backend == "invalid":
        assert "must be auto, python, or rust" in result.stderr


def test_native_rejects_invalid_labels():
    native = pytest.importorskip("cayley_kernel_rust")
    with pytest.raises(ValueError, match="nonzero"):
        native.basis(0, 1)
    with pytest.raises(ValueError, match="nonzero"):
        native.basis_many([(1, 1), (0, 1)])
    # PyO3's unsigned integer conversion uses different CPython APIs across
    # Python versions, which can raise either exception for negative values.
    # The public wrapper separately guarantees ValueError before conversion.
    with pytest.raises((ValueError, OverflowError)):
        native.twist(-1, 2)
