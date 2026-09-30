"""Every public example runs, and uses the public API only."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.py"))


def test_examples_are_numbered_and_present() -> None:
    names = [p.name for p in EXAMPLES]
    assert names[:7] == [
        "01_quickstart.py",
        "02_attribution.py",
        "03_intervention.py",
        "04_faithfulness.py",
        "05_concepts.py",
        "06_audit.py",
        "07_save_reload.py",
    ]


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_uses_public_api_and_runs(path: Path) -> None:
    source = path.read_text()
    assert "beyondnn._testing" not in source
    assert "beyondnn.core" not in source
    subprocess.run([sys.executable, str(path)], check=True, capture_output=True, timeout=600)
