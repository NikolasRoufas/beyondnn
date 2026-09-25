"""Phase 3: Captum is optional; BeyondNN works without it and refuses unverified versions."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

import beyondnn.attribution as A
import beyondnn.attribution.captum as C


def test_importing_beyondnn_attribution_does_not_import_captum() -> None:
    code = (
        "import sys, torch, beyondnn, beyondnn.attribution as A\n"
        "from beyondnn.interventions import metrics\n"
        "A.attribute(torch.nn.Linear(2, 1).eval(), torch.ones(1, 2),"
        " target=metrics.select([0, 0]), method=A.gradient())\n"
        "print('captum' in sys.modules)"
    )
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=root
    )
    assert out.stdout.strip() == "False"


def test_missing_captum_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "captum", None)
    with pytest.raises(C.CaptumUnavailableError, match="pip install"):
        C.saliency()


def test_unverified_captum_versions_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("captum")
    fake.__version__ = "0.8.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "captum", fake)
    with pytest.raises(C.CaptumUnavailableError, match="not a verified version"):
        C.integrated_gradients(baseline=A.zero_baseline())
