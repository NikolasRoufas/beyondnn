"""Phase 7.5 analysis helpers and frozen-policy consistency (plan §32 mutation targets)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load() -> Any:
    path = ROOT / "experiments" / "phase7_5" / "analysis75.py"
    spec = importlib.util.spec_from_file_location("analysis75", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AN = _load()


def _row(kind: str, gt: str, standing: str, **configs: str) -> dict[str, Any]:
    return {"kind": kind, "gt": gt, "standing": standing, "configs": configs}


def test_confusion_counts_and_ambiguity() -> None:
    rows = [
        _row("head", "necessary", "supported"),
        _row("head", "necessary", "contradicted"),
        _row("mlp", "necessary", "inconclusive"),
        _row("head", "not_necessary", "contradicted"),
        _row("head", "not_necessary", "supported"),
        _row("head", "not_necessary", "unsupported"),
        _row("head", "ambiguous", "supported"),
        _row("selection", "necessary", "supported"),
    ]
    c = AN.confusion(rows)
    assert c == {"tp": 1, "fn": 1, "pos_other": 1, "tn": 1, "fp": 1, "neg_other": 1, "ambiguous": 1}
    assert AN.confusion(rows, kinds=("selection",))["tp"] == 1
    with pytest.raises(ValueError, match="unknown ground-truth"):
        AN.confusion([_row("head", "maybe", "supported")])


def test_configuration_confusion_excludes_ambiguous() -> None:
    rows = [
        _row("head", "necessary", "supported", zero="supports", resample="supports"),
        _row("head", "not_necessary", "contradicted", zero="supports", resample="contradicts"),
        _row("head", "ambiguous", "supported", zero="supports", resample="supports"),
    ]
    c = AN.configuration_confusion(rows)
    assert c["zero"] == {"tp": 1, "pos": 1, "fp": 1, "neg": 1}
    assert c["resample"] == {"tp": 1, "pos": 1, "fp": 0, "neg": 1}


def test_reliable_rows_excludes_unreliable_and_failed_cases() -> None:
    cases = {
        "1": {"benchmark_reliable": True, "rows": [_row("head", "necessary", "supported")]},
        "2": {"benchmark_reliable": False, "rows": [_row("head", "necessary", "supported")]},
        "3": {"error": "boom"},
    }
    rows = AN.reliable_rows(cases)
    assert [r["case"] for r in rows] == ["1"]


def test_plausibility_is_never_ground_truth() -> None:
    samples = [{"plausibility": {"ig_token_f1": 0.5, "random_token_f1": 0.1}}]
    audits = {"human": {"human_necessary": {"distribution": {"supported": 1}}}}
    out = AN.rationale_summary(samples, audits)
    assert out["human_is_ground_truth"] is False
    assert set(out) == {"plausibility", "faithfulness", "human_is_ground_truth"}
    assert AN.token_f1({1, 2}, {2, 3}) == 0.5
    assert AN.iou({1, 2}, {2, 3}) == pytest.approx(1 / 3)


def test_perturbation_nativeness_labels() -> None:
    assert AN.NATIVE == {"zero": False, "mask": True, "pad": False, "unk": True, "delete": True}


def test_per_sample_support_counts() -> None:
    per_sample = [
        {
            "tested": 4,
            "outcomes": [
                ["primary", "supports", 1],
                ["alternative", "contradicts", 2],
                ["stress_test", "supports", 1],
            ],
        }
    ]
    assert AN.per_sample_support(per_sample) == [(2, 4)]


def test_development_cases_match_the_frozen_policy() -> None:
    """A held-out case must never become a development case after the freeze."""
    policy = (ROOT / "docs" / "PHASE_7_5_FROZEN_POLICY.md").read_text()
    script = (ROOT / "experiments" / "phase7_5" / "external_interpbench.py").read_text()
    dev = re.search(r"DEV_CASES = \(([^)]*)\)", script)
    assert dev is not None
    script_dev = sorted(x.strip().strip('"') for x in dev.group(1).split(",") if x.strip())
    assert script_dev == ["13", "7"]
    assert "**development** 7 and 13" in policy
    concepts = (ROOT / "experiments" / "phase7_5" / "external_tracr_concepts.py").read_text()
    assert 'CASES = {"dev": ("3",), "heldout": ("39",)}' in concepts
