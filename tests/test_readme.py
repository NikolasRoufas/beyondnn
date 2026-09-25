"""The README's runnable examples must actually run (not pseudo-code)."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
MARKER = "# runnable example"


def _runnable_blocks() -> list[str]:
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)
    return [b for b in blocks if b.startswith(MARKER)]


def test_readme_has_a_runnable_example() -> None:
    assert len(_runnable_blocks()) >= 1


def test_readme_examples_execute() -> None:
    for block in _runnable_blocks():
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(block, str(README), "exec"), {"__name__": "readme_example"})
        text = out.getvalue()
        assert "InputRecord" in text
        assert "NO_CAUSAL_EVIDENCE" in text
        assert "not a causal or attributed explanation" in text
