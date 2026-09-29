"""Phase-7.5 results are immutable evidence (Phase 7.75 plan: never overwritten)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_phase75_results_are_unchanged() -> None:
    manifest = json.loads((ROOT / "tests" / "data" / "phase7_5_results_manifest.json").read_text())
    results = ROOT / "experiments" / "phase7_5" / "results"
    present = sorted(str(p.relative_to(ROOT)) for p in results.rglob("*") if p.is_file())
    assert present == sorted(manifest), "Phase-7.5 result files were added or removed"
    for name, digest in manifest.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
