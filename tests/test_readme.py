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
    out = io.StringIO()
    for block in _runnable_blocks():
        with contextlib.redirect_stdout(out):
            exec(compile(block, str(README), "exec"), {"__name__": "readme_example"})
    text = out.getvalue()
    assert "InputRecord" in text
    assert "NO_CAUSAL_EVIDENCE" in text
    assert "not a causal or attributed explanation" in text
    assert "EvidenceStatus.INTERVENTIONAL" in text
    assert "ZERO_ABLATION_MAY_BE_OOD" in text
    assert "EvidenceStatus.ATTRIBUTED" in text
    assert "ATTRIBUTION_BASELINE_ASSUMPTION" in text
    assert "p receives attribution for y -> ['supported']" in text
    assert "p is necessary for y -> ['contradicted']" in text
    assert "NOT EVALUATED" in text
    assert "gradient (1,) contradicts 0.2" in text
    assert "integrated gradients (0,) supports 1.0" in text
    assert "{'unsupported': 1} ['attribution_is_not_intervention']" in text
    assert (
        "{'contradicted': 1} ['attribution_intervention_disagree', 'counterexamples_present']"
        in text
    )
