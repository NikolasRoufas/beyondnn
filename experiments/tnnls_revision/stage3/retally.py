# ruff: noqa: B023
"""TNNLS revision, Stage 3: final tally of S3-CMP (protocol P2) from the per-sample rows.

The in-script tally of ``compare_workflows.py`` used one generic signal per workflow. This
re-tally applies the protocol's definitions exactly and keeps signals apart:
* affected samples: every sample whose evidence carries the fault (F5: all 20 samples, since
  all lack controls; the 6 where the uncontrolled test supports are the *erroneous conclusions*);
* B1 fault signal: the check written for that fault (F1 c3, F2 c1, F3 c2, F4 c5, F5 c4,
  F6 c7, F7 c6); c7 flags in other scenarios are configuration-dependence flags, reported
  separately, never as false alerts;
* BeyondNN signal: a standing other than SUPPORTED, an exclusion, or (F6) the
  ``alternative_reverses`` finding; for F7 the load refusal is study-level.

Usage: python3 retally.py RUN_DIR OUT.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CHECK = {
    "F1_attribution_as_necessity": "c3_no_intervention_evidence",
    "F2_wrong_checkpoint": "c1_checkpoint_mismatch",
    "F3_wrong_input": "c2_input_mismatch",
    "F4_wrong_selection_size": "c5_wrong_k",
    "F5_omitted_controls": "c4_primary_without_controls",
    "F6_configuration_dependence": "c7_alternative_contradicts",
    "F7_tampered_outcome": "c6_outcome_not_reproducible",
}


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = json.loads((run / "rows.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    clean = {r["sample"]: r["bnn"] for r in rows["F0_clean"]}
    table: dict[str, Any] = {}
    for name, rs in rows.items():
        if name == "F0_clean":
            table[name] = {
                "b1_fault_flags": dict(
                    Counter(
                        f for r in rs for f in r["b1_flags"] if f != "c7_alternative_contradicts"
                    )
                ),
                "b1_configuration_dependence_flags": sum(
                    "c7_alternative_contradicts" in r["b1_flags"] for r in rs
                ),
                "beyondnn_alternative_reverses": sum(
                    r["bnn"]["standing"] == "supported"
                    and "alternative_reverses" in r["bnn"]["codes"]
                    for r in rs
                ),
                "beyondnn_standings": dict(Counter(r["bnn"]["standing"] for r in rs)),
            }
            continue
        if name == "F5_omitted_controls":
            aff = {r["sample"] for r in rs}
            wrong = {r["sample"] for r in rs if r["b0_conclusion"] == "supports"}
        else:
            aff = {r["sample"] for r in rs if r["affected"]}
            wrong = {r["sample"] for r in rs if r["affected"] and r["b0_conclusion"] == "supports"}
        chk = CHECK[name]
        err = summary["scenarios"][name].get("beyondnn_error")

        def bnn_sig(r: dict[str, Any]) -> bool:
            if err:
                return True
            b = r["bnn"]
            if b is None:
                return True
            if name == "F6_configuration_dependence":
                return b["standing"] == "supported" and "alternative_reverses" in b["codes"]
            return (b["standing"] != "supported" and b != clean[r["sample"]]) or b["standing"] in (
                "not_evaluated",
                "unsupported",
            )

        def b1_sig(r: dict[str, Any]) -> bool:
            return chk in r["b1_flags"]

        entry = {
            "affected": len(aff),
            "erroneous_b0_support_conclusions": len(wrong),
            "b0": {"detected": 0, "missed": len(aff), "false_alerts": 0},
            "b1": {
                "detected": sum(b1_sig(r) for r in rs if r["sample"] in aff),
                "missed": sum(not b1_sig(r) for r in rs if r["sample"] in aff),
                "false_alerts": sum(b1_sig(r) for r in rs if r["sample"] not in aff),
                "check": chk,
            },
            "beyondnn": {
                "detected": sum(bnn_sig(r) for r in rs if r["sample"] in aff),
                "missed": sum(not bnn_sig(r) for r in rs if r["sample"] in aff),
                "false_alerts": 0 if err else sum(bnn_sig(r) for r in rs if r["sample"] not in aff),
                "erroneous_supports_prevented": sum(
                    1
                    for r in rs
                    if r["sample"] in wrong
                    and (err or (r["bnn"] or {}).get("standing") != "supported")
                ),
                "standings_affected": dict(
                    Counter(
                        (r["bnn"] or {}).get("standing", "refused")
                        for r in rs
                        if r["sample"] in aff
                    )
                ),
                "codes_affected": dict(
                    Counter(
                        c for r in rs if r["sample"] in aff and r["bnn"] for c in r["bnn"]["codes"]
                    )
                ),
                "study_level_refusal": err,
            },
        }
        if name == "F6_configuration_dependence":
            entry["b1_only"] = sorted(
                r["sample"] for r in rs if b1_sig(r) and r["sample"] not in aff
            )
            entry["b1_only_beyondnn_codes"] = {
                r["sample"]: r["bnn"]["codes"] for r in rs if b1_sig(r) and r["sample"] not in aff
            }
        table[name] = entry
    result = {
        "source": str(run),
        "protocol": summary["protocol"],
        "burden_loc": summary["burden_loc"],
        "cost": {k: v for k, v in summary["cost"].items() if k != "per_sample"},
        "table": table,
    }
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    for k, v in table.items():
        print(
            k,
            json.dumps({kk: vv for kk, vv in v.items() if kk not in ("b1_only_beyondnn_codes",)})[
                :400
            ],
        )


if __name__ == "__main__":
    main()
