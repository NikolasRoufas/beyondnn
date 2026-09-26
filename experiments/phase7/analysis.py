"""Phase 7 descriptive analysis of the central audit reports (not pre-registered as a
hypothesis; reported as description only). Reads artifacts/central_<M>_<site>_report.json.

Per claim and model/site:
* the per-sample share of the *recorded* decisive tests that SUPPORT, in bins
  (descriptive, never a score: it weights configurations equally, which nothing justifies);
* for ASSUMPTION_SENSITIVE samples: the set of axes with a finding (e.g. only "threshold");
* configuration-level necessity/sufficiency disagreement: same method, sample, replacement
  and k, count-controlled comprehensiveness SUPPORTS while sufficiency CONTRADICTS (or the
  reverse), counted per sample (the plan-level check compares standings only);
* absolute vs controls: same sample, method and k under r1, the uncontrolled
  comprehensiveness SUPPORTS while the count-controlled one CONTRADICTS.

Usage: python experiments/phase7/analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"
BINS = ((0.0, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.0))


def bin_of(x: float) -> str:
    if x == 0.0:
        return "0"
    if x == 1.0:
        return "1"
    for lo, hi in BINS[1:-1]:
        if lo < x <= hi:
            return f"({lo},{hi}]"
    raise AssertionError(x)


def recorded(group: dict[str, Any]) -> list[dict[str, Any]]:
    return [t for t in group["tests"] if t["alternative"] is None]


def axes(t: dict[str, Any]) -> dict[str, str]:
    return dict(t["axes"])


def analyse(report: dict[str, Any]) -> dict[str, Any]:
    claims = {c["claim"]: c for c in report["claims"]}
    out: dict[str, Any] = {}
    for name, c in claims.items():
        shares: dict[str, int] = {}
        only_axes: dict[str, int] = {}
        for g in c["groups"]:
            decisive = [t for t in recorded(g) if t["outcome"] in ("supports", "contradicts")]
            if decisive:
                share = sum(t["outcome"] == "supports" for t in decisive) / len(decisive)
                key = bin_of(share)
                shares[key] = shares.get(key, 0) + 1
            if g["standing"] == "assumption_sensitive":
                found = sorted(
                    {f["axis"] for f in g["findings"] if f["kind"] == "assumption_sensitive"}
                )
                key = "+".join(found)
                only_axes[key] = only_axes.get(key, 0) + 1
        out[name] = {
            "supports_share_bins": dict(sorted(shares.items())),
            "sensitive_axis_sets": dict(sorted(only_axes.items(), key=lambda kv: -kv[1])),
        }
    for tag in ("g", "ig", "r"):
        nec, suf = claims[f"{tag}_necessary"], claims[f"{tag}_sufficient"]
        by_sample = {g["sample"]: g for g in suf["groups"]}
        n_pairs = n_disagree = samples_disagree = 0
        absolute_vs_controls = 0
        for g in nec["groups"]:
            comp = {
                (axes(t)["replacement"], axes(t)["k"]): t["outcome"]
                for t in recorded(g)
                if axes(t)["null"].startswith("count")
            }
            suff = {
                (axes(t)["replacement"], axes(t)["k"]): t["outcome"]
                for t in recorded(by_sample[g["sample"]])
            }
            disagree = 0
            for key, outcome in comp.items():
                other = suff.get(key)
                if other is None:
                    continue
                n_pairs += 1
                if {outcome, other} == {"supports", "contradicts"}:
                    disagree += 1
            n_disagree += disagree
            samples_disagree += disagree > 0
            none = {axes(t)["k"]: t["outcome"] for t in recorded(g) if axes(t)["null"] == "none"}
            r1_count = {
                axes(t)["k"]: t["outcome"]
                for t in recorded(g)
                if axes(t)["null"].startswith("count") and axes(t)["replacement"] == "zero"
            }
            absolute_vs_controls += any(
                none.get(k) == "supports" and r1_count.get(k) == "contradicts" for k in none
            )
        out[f"{tag}_configuration_level"] = {
            "necessity_sufficiency_pairs": n_pairs,
            "pairs_disagreeing": n_disagree,
            "samples_with_a_disagreeing_pair": samples_disagree,
            "samples_absolute_pass_but_controls_reject": absolute_vs_controls,
        }
    return out


def main() -> None:
    payload: dict[str, Any] = {
        "note": "descriptive; not a pre-registered hypothesis; shares are not scores",
    }
    for path in sorted(ARTIFACTS.glob("central_*_report.json")):
        if "limit" in path.name:
            continue
        key = path.name.removeprefix("central_").removesuffix("_report.json")
        payload[key] = analyse(json.loads(path.read_text()))
        print(key, json.dumps(payload[key]["ig_necessary"]), flush=True)
    (RESULTS / "analysis.json").write_text(json.dumps(payload, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
