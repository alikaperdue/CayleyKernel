# SPDX-License-Identifier: Apache-2.0
"""Thin native-wheel tagging shim; all package metadata lives in pyproject.toml."""

from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel


class NativeLibraryDistribution(Distribution):
    def has_ext_modules(self):
        # This also makes install select platlib, as required by auditwheel.
        return super().has_ext_modules() or any(
            (Path(__file__).parent / "CayleyKernel").glob("*.so")
        )


class NativeLibraryWheel(bdist_wheel):
    def get_tag(self):
        python, abi, platform = super().get_tag()
        if not self.root_is_pure:
            # ctypes uses the C ABI; the library does not link to Python's ABI.
            return "py3", "none", platform
        return python, abi, platform


setup(distclass=NativeLibraryDistribution, cmdclass={"bdist_wheel": NativeLibraryWheel})
