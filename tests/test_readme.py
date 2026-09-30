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


def test_readme_examples_use_the_public_api_only() -> None:
    for block in _runnable_blocks():
        assert "_testing" not in block
        assert "beyondnn.core" not in block


def test_readme_examples_execute() -> None:
    out = io.StringIO()
    for block in _runnable_blocks():
        with contextlib.redirect_stdout(out):
            exec(compile(block, str(README), "exec"), {"__name__": "readme_example"})
    text = out.getvalue()
    # quickstart: attribution alone is UNSUPPORTED; the intervention contradicts the claim
    assert "EvidenceStatus.ATTRIBUTED EvidenceStatus.INTERVENTIONAL -3.0" in text
    assert "{'unsupported': 1} ['attribution_is_not_intervention']" in text
    assert (
        "{'contradicted': 1} ['attribution_intervention_disagree', 'counterexamples_present']"
        in text
    )
    assert "AUDIT" in text
    assert "NOT EVALUATED" in text
    # end-to-end: content-token eligibility; PRIMARY decides, the stress test stays visible
    assert "{'supported': 3}" in text
    assert "['stress_test_reverses']" in text
    assert "(primary 1 of 1, stress_test 0 of 1)" in text
    assert "content_tokens (3, 5)" in text
    # save / reload / verify
    assert "{'supported': 1}" in text
    # concepts: decodable-but-unused stays proposed
    assert "0 supports supports validated_concept" in text
    assert "1 supports contradicts proposed_concept" in text
