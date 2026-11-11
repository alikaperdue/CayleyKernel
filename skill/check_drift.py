"""Check skill signatures and examples against the installed public API."""

import argparse
import ast
import inspect
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import CayleyKernel as ck


def public_function(name):
    if name not in ck.__all__:
        raise AssertionError(f"{name} is no longer a public export")
    return getattr(ck, name)


def check(folder):
    document = (folder / "SKILL.md").read_text(encoding="utf-8")
    blocks = re.findall(r"^```python\n(.*?)^```", document, re.MULTILINE | re.DOTALL)
    signatures = 0
    for block in blocks:
        for node in ast.parse(block).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            expected = f"({ast.unparse(node.args)})"
            actual = str(inspect.signature(public_function(node.name)))
            if expected != actual:
                raise AssertionError(f"{node.name}: documented {expected}, API {actual}")
            signatures += 1
    if not signatures:
        raise AssertionError("No documented signatures found")

    for name in ("cpu_octonion.py", "gpu_batch.py"):
        path = folder / "examples" / name
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), "exec")  # Syntax only; no CuPy import or bytecode files.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "ck":
                continue
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
                keyword.arg is None for keyword in node.keywords
            ):
                raise AssertionError("Use explicit API arguments so drift can be checked")
            signature = inspect.signature(public_function(node.func.attr))
            signature.bind(
                *[None for arg in node.args],
                **{keyword.arg: None for keyword in node.keywords},
            )

    # Isolated mode and a temporary working directory prevent checkout imports.
    with tempfile.TemporaryDirectory() as workdir:
        result = subprocess.run(
            [sys.executable, "-I", "-B", str(folder / "examples" / "cpu_octonion.py")],
            cwd=workdir,
            capture_output=True,
            text=True,
        )
    if result.returncode:
        raise AssertionError(f"CPU example failed:\n{result.stdout}{result.stderr}")
    print(result.stdout.strip())
    print(f"{signatures} public signatures matched; example API calls bind")
    print("GPU example: syntax checked; GPU execution NOT RUN (requires configured hardware)")


def self_test(folder):
    mutations = (
        ("SKILL.md", "def cpu_multiply(a, b):", "def cpu_multiply(a, b, stale=None):"),
        ("examples/cpu_octonion.py", "[-5, 10,", "[999, 10,"),
        ("examples/gpu_batch.py", "result = ck.multiply(a, b)", "result = ("),
    )
    for relative, before, after in mutations:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "skill"
            shutil.copytree(folder, copy)
            path = copy / relative
            original = path.read_text(encoding="utf-8")
            if before not in original:
                raise AssertionError(f"Self-test fixture no longer matches {relative}")
            path.write_text(original.replace(before, after, 1), encoding="utf-8")
            try:
                check(copy)
            except (AssertionError, SyntaxError, TypeError) as error:
                print(f"Detected intentional drift in {relative}: {str(error).splitlines()[0]}")
            else:
                raise AssertionError(f"Missed intentional drift in {relative}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Also inject temporary drift")
    args = parser.parse_args()
    skill = Path(__file__).resolve().parent
    check(skill)
    if args.self_test:
        self_test(skill)
