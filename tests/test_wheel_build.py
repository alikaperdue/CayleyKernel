"""Build-script regression checks without requiring a CUDA compiler."""

import importlib.util
from importlib import metadata as package_metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


class WheelBuildScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "CayleyKernel").mkdir()
        (self.root / "bin").mkdir()
        shutil.copy2(ROOT / "scripts/build_wheel.sh", self.root / "scripts")
        shutil.copy2(ROOT / "Makefile", self.root)
        self.command("nvcc", '''printf '%s\\n' "$@" > nvcc-args
while [ "$#" -gt 0 ]; do
    if [ "$1" = -o ]; then
        shift
        if [ "${OMIT_KERNEL:-}" != 1 ]; then touch "$1"; fi
        break
    fi
    shift
done
''')
        self.command("build-python", '''printf '%s\\n' "$@" > python-args
mkdir -p dist
touch dist/fixture-py3-none-linux_x86_64.whl
''')
        self.command("auditwheel", '''printf '%s\\n' "$@" > auditwheel-args
mkdir -p wheelhouse
touch wheelhouse/fixture-py3-none-manylinux_2_34_x86_64.whl
''')

    def command(self, name, body):
        path = self.root / "bin" / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + body)
        path.chmod(0o755)

    def run_build(self, **overrides):
        env = os.environ.copy()
        for name in ("ARCHS", "MAKEFLAGS", "MFLAGS", "LIB_NAME", "NVCC", "OMIT_KERNEL"):
            env.pop(name, None)
        env.update(PATH=f"{self.root / 'bin'}:{env['PATH']}", PYTHON="build-python")
        env.update(overrides)
        return subprocess.run(
            ["bash", "scripts/build_wheel.sh"], cwd=self.root, env=env,
            text=True, capture_output=True,
        )

    def test_default_architectures_and_packaged_output(self):
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = (self.root / "nvcc-args").read_text()
        self.assertIn("arch=compute_75,code=sm_75", args)
        self.assertIn("arch=compute_120,code=sm_120", args)
        self.assertTrue((self.root / "CayleyKernel/cuda_kernel.so").is_file())
        self.assertFalse((self.root / "cuda_kernel.so").exists())
        self.assertEqual((self.root / "python-args").read_text(), "-m\nbuild\n--wheel\n")
        self.assertIn("repair\n", (self.root / "auditwheel-args").read_text())

    def test_explicit_architectures(self):
        result = self.run_build(ARCHS="sm_80 sm_89")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = (self.root / "nvcc-args").read_text()
        self.assertIn("arch=compute_80,code=sm_80", args)
        self.assertIn("arch=compute_89,code=sm_89", args)
        self.assertNotIn("arch=compute_75", args)

    def test_empty_architectures_use_defaults(self):
        result = self.run_build(ARCHS="")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("arch=compute_75", (self.root / "nvcc-args").read_text())

    def test_missing_kernel_stops_before_packaging(self):
        result = self.run_build(OMIT_KERNEL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("kernel build did not produce", result.stderr)
        self.assertFalse((self.root / "python-args").exists())


@unittest.skipUnless(importlib.util.find_spec("build"), "requires Python build tooling")
class WheelMetadataTests(unittest.TestCase):
    def test_native_and_python_only_wheel_metadata(self):
        # --no-isolation keeps these tests offline, but requires the active
        # interpreter to supply every declared build-system dependency.
        from build import ProjectBuilder
        from packaging.requirements import Requirement

        missing = []
        for value in ProjectBuilder(str(ROOT)).build_system_requires:
            requirement = Requirement(value)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            try:
                version = package_metadata.version(requirement.name)
            except package_metadata.PackageNotFoundError:
                missing.append(value)
            else:
                if version not in requirement.specifier:
                    missing.append(value)
        if missing:
            self.skipTest(
                "missing build dependencies: " + ", ".join(sorted(missing))
                + "; install with: python -m pip install build 'setuptools>=77.0.3'"
            )
        for native in (False, True):
            with self.subTest(native=native), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                for name in ("setup.py", "pyproject.toml", "README.md", "LICENSE"):
                    shutil.copy2(ROOT / name, root)
                package = root / "CayleyKernel"
                package.mkdir()
                (package / "__init__.py").write_text("")
                if native:
                    (package / "cuda_kernel.so").write_bytes(b"native-library-fixture")
                result = subprocess.run(
                    [sys.executable, "-m", "build", "--wheel", "--no-isolation"],
                    cwd=root, text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                wheel, = (root / "dist").glob("*.whl")
                with zipfile.ZipFile(wheel) as archive:
                    metadata_path, = [p for p in archive.namelist() if p.endswith(".dist-info/WHEEL")]
                    metadata = archive.read(metadata_path).decode()
                    if native:
                        self.assertIn("Root-Is-Purelib: false", metadata)
                        self.assertIn("Tag: py3-none-", metadata)
                        self.assertNotIn("Tag: py3-none-any", metadata)
                        self.assertNotIn("-any.whl", wheel.name)
                        self.assertEqual(archive.read("CayleyKernel/cuda_kernel.so"), b"native-library-fixture")
                    else:
                        self.assertIn("Root-Is-Purelib: true", metadata)
                        self.assertIn("Tag: py3-none-any", metadata)


if __name__ == "__main__":
    unittest.main()
