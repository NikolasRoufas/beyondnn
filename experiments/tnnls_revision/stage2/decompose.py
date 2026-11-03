"""TNNLS revision, Stage 2 (S2-CFG analysis; protocol P1): fixed-selection decomposition of the
configuration reversals of PRIMARY-supported necessity claims.

Input: one or more ``configs.csv`` files written by ``run_central_configs.py`` (every tested
configuration of every ``*_necessary`` claim, per sample, with its axis values and role).
Pure post-processing; no model is run.

Definitions (protocol P1):
* unit = (model, site, seed, sample) whose claim is PRIMARY-SUPPORTED;
* PRIMARY configuration = the role-``primary`` entry without a threshold re-evaluation;
* a configuration's *changed axes* = the axes whose value differs from the PRIMARY entry;
* fixed subject = k unchanged (IG/G top-k at the same k selects the same units: the ranking
  does not depend on the replacement, null or threshold);
* reversal = outcome ``contradicts``; ``inconclusive`` is kept as its own category.

Usage: python decompose.py OUT.json CONFIGS.csv [CONFIGS.csv ...]
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

AXES = ("replacement", "k", "null", "threshold")


def load(paths: list[Path]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for p in paths:
        with p.open() as f:
            for r in csv.DictReader(f):
                r["ax"] = json.loads(r["axes_json"])
                key = (r["model"], r["site"], r["seed"], r["sample_index"], r["claim"])
                groups[key].append(r)
    return groups


def family(changed: frozenset[str]) -> str:
    if not changed:
        return "none"
    if len(changed) == 1:
        return f"only:{next(iter(changed))}"
    return "multi:" + "+".join(sorted(changed))


def analyse_group(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    prim = [r for r in rows if r["role"] == "primary" and not r["alternative"]]
    if len(prim) != 1:
        return None
    p = prim[0]
    if p["outcome"] != "supports":
        return None
    out: dict[str, Any] = {"standing": p["standing"], "configs": []}
    for r in rows:
        if r is p:
            continue
        changed = frozenset(a for a in AXES if r["ax"].get(a) != p["ax"].get(a))
        out["configs"].append(
            {
                "role": r["role"],
                "family": family(changed),
                "fixed_subject": "k" not in changed,
                "outcome": r["outcome"],
                "values": {a: r["ax"].get(a) for a in changed},
            }
        )
    return out


def summarise(units: dict[tuple[str, ...], dict[str, Any]]) -> dict[str, Any]:
    """Per unit flags, then counts. A unit is 'reversed by X' if >= 1 configuration of X
    contradicts PRIMARY."""
    flags: dict[tuple[str, ...], dict[str, bool]] = {}
    fam_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for key, u in units.items():
        f: dict[str, bool] = {}
        for role in ("alternative", "stress_test"):
            cs = [c for c in u["configs"] if c["role"] == role]
            f[f"{role}:any"] = any(c["outcome"] == "contradicts" for c in cs)
            f[f"{role}:fixed_subject"] = any(
                c["outcome"] == "contradicts" and c["fixed_subject"] for c in cs
            )
            f[f"{role}:subject_change"] = any(
                c["outcome"] == "contradicts" and not c["fixed_subject"] for c in cs
            )
            by_fam: dict[str, list[str]] = defaultdict(list)
            for c in cs:
                by_fam[c["family"]].append(c["outcome"])
            for fam, outs in by_fam.items():
                k = f"{role}|{fam}"
                fam_counts[k]["units_tested"] += 1
                fam_counts[k]["units_reversed"] += "contradicts" in outs
                fam_counts[k]["units_only_inconclusive_or_support"] += "contradicts" not in outs
                fam_counts[k]["configs"] += len(outs)
                fam_counts[k]["configs_contradict"] += outs.count("contradicts")
                fam_counts[k]["configs_inconclusive"] += outs.count("inconclusive")
                f[f"{role}|{fam}"] = "contradicts" in outs
        flags[key] = f
    n = len(units)
    alt_any = sum(f["alternative:any"] for f in flags.values())
    fixed = sum(f["alternative:fixed_subject"] for f in flags.values())
    subj = sum(f["alternative:subject_change"] for f in flags.values())
    only_subj = sum(
        f["alternative:subject_change"] and not f["alternative:fixed_subject"]
        for f in flags.values()
    )
    single_fixed = {
        a: sum(f.get(f"alternative|only:{a}", False) for f in flags.values())
        for a in ("replacement", "null", "threshold", "k")
    }
    any_single_fixed = sum(
        any(f.get(f"alternative|only:{a}", False) for a in ("replacement", "null", "threshold"))
        for f in flags.values()
    )
    multi_fixed_only = sum(
        f["alternative:fixed_subject"]
        and not any(
            f.get(f"alternative|only:{a}", False) for a in ("replacement", "null", "threshold")
        )
        for f in flags.values()
    )
    return {
        "units": n,
        "standing_of_units": dict(Counter(u["standing"] for u in units.values())),
        "alternative_any_reversal": alt_any,
        "alternative_fixed_subject_reversal": fixed,
        "alternative_subject_change_reversal": subj,
        "alternative_reversed_only_by_subject_change": only_subj,
        "alternative_single_axis_reversal": single_fixed,
        "alternative_any_single_fixed_axis_reversal": any_single_fixed,
        "alternative_fixed_subject_reversal_only_multi_axis": multi_fixed_only,
        "alternative_not_reversed": n - alt_any,
        "stress_any_reversal": sum(f["stress_test:any"] for f in flags.values()),
        "stress_fixed_subject_reversal": sum(
            f["stress_test:fixed_subject"] for f in flags.values()
        ),
        "families": {k: dict(v) for k, v in sorted(fam_counts.items())},
    }


def main() -> None:
    out_path = Path(sys.argv[1])
    groups = load([Path(p) for p in sys.argv[2:]])
    result: dict[str, Any] = {
        "protocol": "docs/tnnls_revision/protocols/P1_CENTRAL_RERUN_PROTOCOL.md"
    }
    for claim in ("ig_necessary", "g_necessary", "r_necessary"):
        units = {}
        for key, rows in groups.items():
            if key[4] != claim:
                continue
            a = analyse_group(rows)
            if a is not None:
                units[key] = a
        per_site: dict[str, Any] = {}
        for site_key in sorted({k[:3] for k in units}):
            sub = {k: v for k, v in units.items() if k[:3] == site_key}
            per_site["/".join(site_key)] = summarise(sub)
        n_samples = Counter(k[:3] for k in groups if k[4] == claim)
        result[claim] = {
            "samples_per_site": {"/".join(k): v for k, v in sorted(n_samples.items())},
            "per_site": per_site,
            "pooled_descriptive_only": summarise(units),
        }
    out_path.write_text(json.dumps(result, indent=1, sort_keys=True))
    ig = result["ig_necessary"]["pooled_descriptive_only"]
    print(json.dumps({k: v for k, v in ig.items() if k != "families"}, indent=1))


if __name__ == "__main__":
    main()
