"""Exercise paper opt-in before imports, including explicitly selected files."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu
ROOT = Path(__file__).resolve().parents[1]


def test_papers_require_opt_in(tmp_path):
    tests = tmp_path / "tests"
    papers = tests / "papers"
    papers.mkdir(parents=True)
    shutil.copyfile(ROOT / "tests/conftest.py", tests / "conftest.py")
    (tests / "test_default.py").write_text(
        "import pytest\npytestmark = pytest.mark.cpu\ndef test_default(): pass\n")
    paper = papers / "test_paper.py"
    imported = tmp_path / "paper-imported"
    paper.write_text(
        "from pathlib import Path\nimport pytest\npytestmark = pytest.mark.cpu\n"
        f"Path({str(imported)!r}).touch()\n"
        "def test_paper(): pass\n")
    marked = tests / "test_marked.py"
    marked.write_text(
        "import pytest\npytestmark = [pytest.mark.cpu, pytest.mark.papers]\n"
        "def test_marked(): pass\n")
    base = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    env = {**os.environ, "PYTHONPATH": str(ROOT)}

    def run(*args):
        result = subprocess.run(base + list(args), cwd=tmp_path, env=env,
                                text=True, capture_output=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    assert "1 passed, 1 skipped" in run(str(tests))
    assert not imported.exists(), "Default discovery imported a paper module"
    assert "1 skipped" in run(str(paper))
    assert "1 passed" in run(str(paper), "--run-papers", "-m", "papers")
    assert "2 passed" in run(str(tests), "--run-papers", "-m", "papers")
