"""TNNLS revision, Stage 4 aggregation (S4-SEED, S4-COST, S4-XPLAT; protocol P1).

Reads the run directories written by ``run_central_configs.py`` (summary.json, timing.json,
configs.csv) and writes per-checkpoint robustness tables, the cost breakdown and the
cross-platform comparison against the frozen macOS results. Pure post-processing.

Usage: python aggregate.py RUNS_DIR FROZEN_RESULTS_DIR OUT.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage2"))
import decompose as D

SITES = {"A": ("input", "net.1"), "B": ("pixels", "relu2"), "C": ("tokens",), "D": ("tokens",)}


def merged(runs: Path, prefix: str) -> tuple[dict[str, Any], list[Path], list[dict[str, Any]]]:
    """Merge shard runs: per-sample entries are shard-independent (instance scope)."""
    dirs = sorted(p for p in runs.iterdir() if p.name == prefix or p.name.startswith(prefix + "_s"))
    docs = [
        json.loads((d / "summary.json").read_text()) for d in dirs if (d / "summary.json").exists()
    ]
    if not docs:
        return {}, [], []
    out = {
        "model": docs[0]["model"],
        "environment": docs[0]["environment"],
        "settings": {},
        "shards": len(docs),
        "dirs": [d.name for d in dirs],
    }
    for site in docs[0]["settings"]:
        per = {}
        for doc in docs:
            for claim, c in doc["settings"][site]["audit"].items():
                per.setdefault(claim, []).extend(c["per_sample"])
        out["settings"][site] = {
            "audit": {c: {"per_sample": v} for c, v in per.items()},
            "samples": [m for doc in docs for m in doc["settings"][site]["samples"]],
            "n_results": sum(doc["settings"][site]["n_results"] for doc in docs),
        }
    timings = [
        json.loads((d / "timing.json").read_text()) for d in dirs if (d / "timing.json").exists()
    ]
    cfgs = [d / "configs.csv" for d in dirs if (d / "configs.csv").exists()]
    return out, cfgs, timings


def site_stats(doc: dict[str, Any], site: str) -> dict[str, Any]:
    a = doc["settings"][site]["audit"]
    ig, r, g = (a[f"{m}_necessary"]["per_sample"] for m in ("ig", "r", "g"))
    sup = [x for x in ig if x["standing"] == "supported"]
    return {
        "n": len(ig),
        "ig_supported": len(sup),
        "g_supported": sum(x["standing"] == "supported" for x in g),
        "r_supported": sum(x["standing"] == "supported" for x in r),
        "ig_supported_alternative_reverses": sum("alternative_reverses" in x["codes"] for x in sup),
        "ig_standings": dict(Counter(x["standing"] for x in ig)),
        "ig_control_unattainable": sum("control_criterion_unattainable" in x["codes"] for x in ig),
    }


def compare(doc: dict[str, Any], frozen: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for site in doc["settings"]:
        diffs = {}
        for claim, c in doc["settings"][site]["audit"].items():
            fa = {x["index"]: x for x in frozen["settings"][site]["audit"][claim]["per_sample"]}
            ra = {x["index"]: x for x in c["per_sample"]}
            diffs[claim] = {
                "samples": len(ra),
                "same_samples": set(fa) == set(ra),
                "standing_diff": sorted(
                    i for i in ra if i in fa and ra[i]["standing"] != fa[i]["standing"]
                ),
                "codes_or_outcomes_diff": sorted(
                    i
                    for i in ra
                    if i in fa
                    and (ra[i]["codes"], ra[i]["outcomes"]) != (fa[i]["codes"], fa[i]["outcomes"])
                ),
            }
        out[site] = diffs
    return out


def main() -> None:
    runs, frozen_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    result: dict[str, Any] = {
        "protocol": "docs/tnnls_revision/protocols/P1_CENTRAL_RERUN_PROTOCOL.md",
        "runs": {},
    }
    for model in "ABCD":
        for tag in [f"S2_{model}_frozen"] + (
            [f"S4_{model}_seed{s}" for s in range(5)] if model in "AB" else []
        ):
            doc, cfgs, timings = merged(runs, tag)
            if not doc:
                continue
            entry: dict[str, Any] = {
                "checkpoint_state_digest": doc["model"].get("checkpoint_state_digest"),
                "accuracy": doc["model"].get("accuracy"),
                "platform": doc["environment"].get("platform"),
                "shards": doc["shards"],
                "sites": {s: site_stats(doc, s) for s in doc["settings"]},
            }
            if cfgs:
                groups = D.load(cfgs)
                units = {
                    k: a
                    for k, rows in groups.items()
                    if k[4] == "ig_necessary" and (a := D.analyse_group(rows))
                }
                entry["decomposition"] = {
                    s: D.summarise({k: v for k, v in units.items() if k[1] == s})
                    for s in doc["settings"]
                }
            if timings:
                t: dict[str, Any] = {}
                for site in doc["settings"]:
                    rows = [x["settings"][site] for x in timings if site in x["settings"]]
                    t[site] = (
                        {
                            k: sum(r.get(k, 0) for r in rows)
                            for k in (
                                "samples",
                                "results",
                                "evidence_seconds",
                                "audit_seconds",
                                "verify_seconds",
                                "report_to_json_seconds",
                                "report_bytes",
                            )
                        }
                        | {
                            k: sum(r[k] for r in rows if k in r)
                            for k in (
                                "save_evidence_seconds",
                                "evidence_bytes",
                                "load_and_reaudit_seconds",
                                "verify_saved_report_from_disk_seconds",
                            )
                            if any(k in r for r in rows)
                        }
                        | {
                            "peak_rss_mb_max_process": max(
                                r.get("peak_rss_mb_so_far", 0) for r in rows
                            ),
                            "reaudit_from_disk_identical": [
                                r["reaudit_from_disk_identical"]
                                for r in rows
                                if "reaudit_from_disk_identical" in r
                            ],
                        }
                    )
                entry["timing"] = t
            fz = frozen_dir / f"central775_{model}_heldout_all.json"
            if tag.endswith("frozen") and fz.exists():
                frozen = json.loads(fz.read_text())
                entry["frozen_checkpoint_digest_equal"] = (
                    frozen["model"]["checkpoint_state_digest"] == entry["checkpoint_state_digest"]
                )
                entry["vs_frozen_macos"] = compare(doc, frozen)
            result["runs"][tag] = entry
    out_path.write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    for tag, e in result["runs"].items():
        print(
            tag,
            e["checkpoint_state_digest"][:18],
            json.dumps(
                {
                    s: (
                        v["ig_supported"],
                        v["n"],
                        v["ig_supported_alternative_reverses"],
                        v["r_supported"],
                    )
                    for s, v in e["sites"].items()
                }
            ),
        )


if __name__ == "__main__":
    main()
