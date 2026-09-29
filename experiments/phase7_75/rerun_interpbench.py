"""Phase 7.75: re-run InterpBench cases with the Phase-7.5 experiment code, unchanged, under the
Phase-7.75 audit (ADR-053/054). Phase-7.5 outputs are never overwritten: results go to
``experiments/phase7_75/results/`` and the Phase-7.5 per-case report artifacts are restored.

Usage (InterpBench env, from experiments/phase7_5/artifacts):
    python ../../phase7_75/rerun_interpbench.py dev|heldout [--cases 7,13]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P75 = HERE.parent / "phase7_5"
sys.path.insert(0, str(P75))

import external_interpbench as X  # noqa: E402

RESULTS = HERE / "results"
ARTIFACTS = HERE / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["dev", "heldout"])
    parser.add_argument("--cases", default=None)
    args = parser.parse_args()
    cases = (
        list(X.DEV_CASES)
        if args.split == "dev"
        else [c for c in X.eligible_cases() if c not in X.DEV_CASES]
    )
    if args.cases:
        cases = [c for c in cases if c in args.cases.split(",")]
    ARTIFACTS.mkdir(exist_ok=True)
    out: dict[str, object] = {
        "environment": X.environment() if hasattr(X, "environment") else {},
        "split": args.split,
        "code": "experiments/phase7_5/external_interpbench.py (unchanged); audit at HEAD",
        "cases": {},
    }
    for c in cases:
        p75_report = X.ART / f"interpbench_{c}_report.json"
        backup = ARTIFACTS / f"p75_backup_interpbench_{c}_report.json"
        if p75_report.exists():
            shutil.copy2(p75_report, backup)
        try:
            out["cases"][c] = X.run_case(c)  # type: ignore[index]
        except Exception as exc:  # reported, never hidden
            out["cases"][c] = {"case": c, "error": f"{type(exc).__name__}: {exc}"}  # type: ignore[index]
        finally:
            if p75_report.exists():
                shutil.move(p75_report, ARTIFACTS / f"interpbench_{c}_report.json")
            if backup.exists():
                shutil.move(backup, p75_report)
        print(c, out["cases"][c].get("error", "ok"), flush=True)  # type: ignore[index]
    name = f"interpbench_{args.split}" + (f"_{args.cases.replace(',', '_')}" if args.cases else "")
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
