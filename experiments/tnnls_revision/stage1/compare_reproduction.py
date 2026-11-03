"""TNNLS revision, Stage 1: compare a model re-run against a frozen central775 result.

INDEPENDENT MODEL-EXPERIMENT REPRODUCTION check (S1-REPRO-A). The re-run is produced by the
unchanged frozen script in an isolated copy of the repository (never in place); this script only
compares the two JSON files.

Usage: python3 compare_reproduction.py FROZEN.json RERUN.json OUT.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

FIELDS = ("standing", "codes", "outcomes", "reversal_axes", "tested")


def main() -> None:
    frozen_p, rerun_p, out_p = (Path(a) for a in sys.argv[1:4])
    frozen, rerun = json.loads(frozen_p.read_text()), json.loads(rerun_p.read_text())
    out: dict[str, Any] = {
        "frozen": str(frozen_p),
        "frozen_environment": frozen["environment"],
        "rerun_environment": rerun["environment"],
        "frozen_model": {
            k: frozen["model"].get(k) for k in ("checkpoint_state_digest", "accuracy")
        },
        "rerun_model": {k: rerun["model"].get(k) for k in ("checkpoint_state_digest", "accuracy")},
        "sites": {},
    }
    for site, s in frozen["settings"].items():
        r = rerun["settings"][site]
        site_out: dict[str, Any] = {
            "verify_report": [s.get("verify_report"), r.get("verify_report")],
            "n_results": [s.get("n_results"), r.get("n_results")],
            "claims": {},
        }
        for claim, c in s["audit"].items():
            fa = {x["index"]: x for x in c["per_sample"]}
            ra = {x["index"]: x for x in r["audit"][claim]["per_sample"]}
            diffs = {f: sum(fa[i].get(f) != ra.get(i, {}).get(f) for i in fa) for f in FIELDS}
            site_out["claims"][claim] = {
                "samples": len(fa),
                "same_sample_set": set(fa) == set(ra),
                "per_sample_field_differences": diffs,
                "distribution": [c["distribution"], r["audit"][claim]["distribution"]],
            }
        out["sites"][site] = site_out
    Path(out_p).write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
