# ruff: noqa: E501
"""Evaluate the frozen Phase-7.75 hypotheses (docs/PHASE_7_75_PLAN.md) on the held-out results.

Pure post-processing of committed result JSON; nothing is tuned. Missing inputs are reported as
"not_available". Writes ``results/hypotheses775.json``.

Usage (from the repo root): uv run --no-project --python 3.12 --with-editable . --with numpy python experiments/phase7_75/evaluate775.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import beyondnn as bnn

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
P75 = HERE.parent / "phase7_5" / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase7_5"))
import analysis75  # noqa: E402
import analysis775 as AN  # noqa: E402

AU = bnn.audits
PRIMARY = "tensor/resample|none|recorded"


def load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def wilson(k: int, n: int, what: str, unit: str) -> dict[str, Any]:
    if n == 0:
        return {"k": 0, "n": 0, "text": "n = 0"}
    iv = AU.wilson(k, n, quantity=what, unit=unit)
    return {
        "k": k,
        "n": n,
        "estimate": iv.estimate,
        "low": iv.low,
        "high": iv.high,
        "text": f"{k}/{n} = {iv.estimate:.4f} [95% Wilson {iv.low:.4f}, {iv.high:.4f}]",
    }


# ------------------------------------------------------------------ TD (§3)


def td() -> dict[str, Any]:
    d = load(RESULTS / "td_heldout.json")
    if d is None:
        return {"status": "not_available"}
    progs = d["programs"]
    rows = [x for r in progs.values() if "rows" in r for x in r["rows"]]
    nodes = AN.truth_table(rows)
    sels = AN.truth_table(rows, kinds=("selection",))
    comps = AN.component_support(rows)
    t_true = sum(nodes["known_true"].values())
    t_false = sum(nodes["known_false"].values())
    out: dict[str, Any] = {
        "truth_source": "program source + weight structure",
        "audited_operation": "resample",
        "independent": AN.truth_is_independent("program source + weight structure", "resample"),
        "errors": {p: r["error"] for p, r in progs.items() if "error" in r},
        "compiled_matches_interpreter": {
            p: r.get("compiled_matches_interpreter") for p, r in progs.items()
        },
        "results_excluded": sum(r.get("results_excluded", 0) for r in progs.values()),
        "node_truth_table": nodes,
        "selection_truth_table": sels,
        "components": comps,
    }
    true_comps = [v for v in comps.values() if v["truth"] == "known_true"]
    out["TD1"] = {
        "holds": all(v["supported"] >= 1 for v in true_comps),
        "min_supported_of_40": min(v["supported"] for v in true_comps),
    }
    out["TD2"] = wilson(
        nodes["known_false"]["supported"], t_false, "PRIMARY SUPPORTED on decoys", "node instances"
    ) | {"threshold": "<= 0.05; failure > 0.10"}
    out["TD2"]["holds"] = out["TD2"]["estimate"] <= 0.05
    out["TD3"] = wilson(
        nodes["known_true"]["supported"],
        t_true,
        "PRIMARY SUPPORTED on used components",
        "node instances",
    ) | {"threshold": ">= 0.60"}
    out["TD3"]["holds"] = out["TD3"]["estimate"] >= 0.60
    corr = [
        x
        for x in rows
        if x["program"].startswith("ho_corr")
        and x["kind"] != "selection"
        and x["truth"] == "known_false"
    ]
    out["TD4"] = wilson(
        sum(x["standing"] == "supported" for x in corr),
        len(corr),
        "PRIMARY SUPPORTED on correlated decoys",
        "node instances",
    )
    out["TD4"]["holds"] = out["TD4"]["estimate"] <= 0.05
    zk, zn = AN.configuration_rates(rows, "zero|none|recorded", truth="known_false")
    out["TD5"] = {
        "zero_fp": wilson(zk, zn, "zero ablation SUPPORTS decoys", "node instances"),
        "primary_fp": out["TD2"]["estimate"],
        "holds": zk / zn >= out["TD2"]["estimate"] + 0.05,
    }
    empty_supported = sum(
        v == "supports" for x in rows if x["truth"] == "empty" for v in x["configs"].values()
    )
    ao = Counter()
    for r in progs.values():
        for c, dist in (r.get("attribution_only") or {}).items():
            if "ig_top" in c:
                ao.update(dist)
    out["TD6"] = {
        "attribution_only_ig": dict(ao),
        "empty_supports_any_configuration": empty_supported,
        "holds": set(ao) == {"unsupported"} and empty_supported == 0,
    }
    agree = Counter(
        (x["configs"].get(PRIMARY), x["tl_interchange_flips"])
        for x in rows
        if x["kind"] != "selection"
    )
    decisive = {k: v for k, v in agree.items() if k[0] in ("supports", "contradicts")}
    ok = sum(v for (o, f), v in decisive.items() if (o == "supports") == bool(f))
    out["TD7"] = wilson(
        ok,
        sum(decisive.values()),
        "PRIMARY outcome agrees with TransformerLens interchange",
        "decisive node instances",
    ) | {"table": {f"{k[0]}|tl_flip={k[1]}": v for k, v in agree.items()}}
    out["TD7"]["holds"] = out["TD7"]["estimate"] >= 0.99
    out["TD8"] = wilson(
        sels["known_false"]["supported"],
        sum(sels["known_false"].values()),
        "IG-selected decoy heads PRIMARY SUPPORTED",
        "selection instances",
    ) | {
        "ig_selected_decoy": f"{sum(sels['known_false'].values())} of {sum(sum(v.values()) for v in sels.values())} selections"
    }
    out["TD8"]["holds"] = out["TD8"]["estimate"] <= 0.05
    heads = [
        x for x in rows if x["kind"] == "head" and "tensor/resample|count|recorded" in x["configs"]
    ]
    unatt = sum("control_criterion_unattainable" in x["codes"] for x in heads)
    out["TD9"] = {
        "count_null_head_instances": len(heads),
        "with_unattainable_finding": unatt,
        "count_null_supports_on_used": AN.configuration_rates(
            rows, "tensor/resample|count|recorded", truth="known_true"
        ),
    }
    out["per_configuration"] = {
        cfg: {
            t: AN.configuration_rates(rows, cfg, truth=t)
            for t in ("known_true", "known_false", "empty")
        }
        for cfg in sorted({c for x in rows for c in x["configs"]})
    }
    out["claim_F"] = (
        "supported_with_qualifier"
        if all(out[h]["holds"] for h in ("TD1", "TD2", "TD4", "TD8"))
        else ("rejected" if out["TD2"]["estimate"] > 0.10 else "narrowed")
    )
    return out


# ------------------------------------------------------------------ concepts (§4)


def concepts() -> dict[str, Any]:
    d = load(RESULTS / "td_concepts_heldout.json")
    out: dict[str, Any] = {
        "calibration": {
            k: v
            for k, v in (load(RESULTS / "concept_calibration.json") or {}).items()
            if k != "grid"
        }
    }
    if d is None:
        return out | {"status": "not_available"}
    table = {}
    kc1 = kc2 = True
    for p, r in d["programs"].items():
        if "error" in r:
            table[p] = r["error"]
            kc1 = kc2 = False
            continue
        for name, c in r["concepts"].items():
            table[f"{p}/{name}"] = {
                "known": c["known"],
                "encoding": c["encoding"],
                "rel": {
                    k: c["rules"]["rel"][k]
                    for k in ("min_change", "use", "use_effect", "validation")
                },
                "abs": {
                    k: c["rules"]["abs"][k]
                    for k in ("min_change", "use", "use_effect", "validation")
                },
                "audit": {k: v[name]["standing"] for k, v in r["audits"].items()},
            }
            if c["known"] == "positive":
                kc1 &= c["rules"]["rel"]["validation"]["covariance"] == "validated_concept"
            else:
                kc2 &= all(
                    v != "validated_concept"
                    for rule in c["rules"].values()
                    for v in rule["validation"].values()
                )
                kc2 &= all(v[name]["standing"] != "supported" for v in r["audits"].values())
    out |= {"table": table, "KC1": {"holds": kc1}, "KC2": {"holds": kc2}}
    abs_pos = {
        k: v["abs"]["validation"]["covariance"]
        for k, v in table.items()
        if isinstance(v, dict) and v["known"] == "positive"
    }
    out["KC3"] = {"abs_rule_known_positive_validation": abs_pos}
    out["claim_G"] = (
        "supported_with_qualifier"
        if kc1 and kc2
        else ("rejected" if not kc2 else "narrowed_to_testing")
    )
    c39 = load(RESULTS / "td_concepts_case39.json")
    if c39 is not None:
        out["case39_not_blind"] = {
            m: {
                name: {
                    "encoding": c["encoding"],
                    "rel": {
                        k: c["rules"]["rel"][k]
                        for k in ("min_change", "use", "use_effect", "validation")
                    },
                    "abs": c["rules"]["abs"]["validation"],
                }
                for name, c in c39[m]["concepts"].items()
            }
            for m in ("hl", "ll")
        }
    return out


# ------------------------------------------------------------------ central (§5-§6)


def central_site(doc: dict[str, Any], site: str) -> dict[str, Any]:
    s = doc["settings"][site]
    ig, r, g = (s["audit"][f"{m}_necessary"]["per_sample"] for m in ("ig", "r", "g"))
    n = len(ig)
    sup = [x for x in ig if x["standing"] == "supported"]
    rev = sum("alternative_reverses" in x["codes"] for x in sup)
    return {
        "n": n,
        "results_excluded": s["results_excluded"],
        "ig_supported": wilson(len(sup), n, "IG_necessary PRIMARY SUPPORTED", "samples"),
        "r_supported": wilson(
            sum(x["standing"] == "supported" for x in r),
            n,
            "R_necessary PRIMARY SUPPORTED",
            "samples",
        ),
        "g_supported": sum(x["standing"] == "supported" for x in g),
        "ig_supported_alternative_reverses": f"{rev}/{len(sup)}",
        "ig_configuration_level_disagreement": sum(
            "configuration_level_disagreement" in x["codes"] for x in ig
        ),
        "ig_control_unattainable": sum("control_criterion_unattainable" in x["codes"] for x in ig),
        "ig_effect_without_competitive_advantage": sum(
            "effect_without_competitive_advantage" in x["codes"] for x in ig
        ),
        "distributions": {k: v["distribution"] for k, v in s["audit"].items()},
    }


def central() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model, sites in (
        ("A", ("input", "net.1")),
        ("B", ("pixels", "relu2")),
        ("C", ("tokens",)),
        ("D", ("tokens",)),
    ):
        for elig in ("all", "content_tokens"):
            doc = load(RESULTS / f"central775_{model}_heldout_{elig}.json")
            if doc is None:
                continue
            for site in sites:
                out[f"{model}/{site}/{elig}"] = central_site(doc, site)
        p75 = load(P75 / f"central75_{model}_heldout.json")
        if p75 is not None:
            for site in sites:
                s = p75["settings"][site]["audit"]["ig_necessary"]["per_sample"]
                out[f"{model}/{site}/phase7_5"] = {
                    "ig_supported": sum(x["standing"] == "supported" for x in s),
                    "n": len(s),
                }
    d1, d2 = out.get("D/tokens/all"), out.get("D/tokens/content_tokens")
    if d1 and d2:
        out["SX1"] = {
            "d1": d1["ig_supported"]["k"],
            "d2": d2["ig_supported"]["k"],
            "holds": d2["ig_supported"]["k"] <= d1["ig_supported"]["k"],
        }
        k, n = (int(v) for v in d2["ig_supported_alternative_reverses"].split("/"))
        out["SX2"] = {"reversed": f"{k}/{n}", "holds": None if n < 5 else k / n >= 0.2}
        out["SX3"] = {"holds": d2["ig_supported"]["k"] >= d2["r_supported"]["k"]}
    return out


def special_tokens() -> dict[str, Any]:
    """Which D1 / D2 selections contain [CLS] / [SEP] (from the recorded PRIMARY units)."""
    out: dict[str, Any] = {}
    for elig in ("all", "content_tokens"):
        doc = load(RESULTS / f"central775_D_heldout_{elig}.json")
        if doc is None:
            continue
        s = doc["settings"]["tokens"]
        by_index = {x["index"]: x for x in s["audit"]["ig_necessary"]["per_sample"]}
        sep = sup_sep = sup = 0
        for m in s["samples"]:
            units = m["primary_single_configuration"]["units"]
            has = (m["n_units"] - 1) in units or 0 in units
            sep += has
            if by_index[m["index"]]["standing"] == "supported":
                sup += 1
                sup_sep += has
        out[elig] = {
            "primary_selections_with_special_token": sep,
            "n": len(s["samples"]),
            "ig_supported": sup,
            "ig_supported_with_special_token": sup_sep,
        }
    return out


# ------------------------------------------------------------------ InterpBench (§6)


def interpbench() -> dict[str, Any]:
    new = load(RESULTS / "interpbench_heldout.json")
    if new is None:
        return {"status": "not_available"}
    old = dict(load(P75 / "external_interpbench_heldout.json")["cases"])
    old["124"] = load(P75 / "external_interpbench_heldout_case124_dv1.json")["cases"]["124"]
    cases = dict(new["cases"])
    rows_new = analysis75.reliable_rows(cases)
    rows_old = analysis75.reliable_rows(old)
    heads_new = [x for x in rows_new if x["kind"] == "head" and x["gt"] == "necessary"]
    changed = 0
    old_map = {(x["case"], x["claim"], x["sample"]): x["standing"] for x in rows_old}
    for x in rows_new:
        changed += old_map.get((x["case"], x["claim"], x["sample"])) != x["standing"]
    return {
        "truth_independent": AN.truth_is_independent(
            "resample (LL filter) + interchange", "resample (LL filter) + interchange"
        ),
        "note": "implementation-consistency check only (Phase 7.5 circularity); not confirmatory",
        "primary_confusion_7_5": analysis75.confusion(rows_old),
        "primary_confusion_7_75": analysis75.confusion(rows_new),
        "primary_standings_changed": changed,
        "count_null_on_necessary_heads": dict(
            Counter(x["configs"].get("tensor/resample|count|recorded") for x in heads_new)
        ),
        "unattainable_findings": sum(
            "control_criterion_unattainable" in x["codes"] for x in rows_new
        ),
        "errors": {c: r["error"] for c, r in cases.items() if "error" in r},
    }


def main() -> None:
    out = {
        "policy": "docs/PHASE_7_75_PLAN.md (frozen at b355129)",
        "TD": td(),
        "concepts": concepts(),
        "central": central(),
        "special_tokens": special_tokens(),
        "interpbench": interpbench(),
        "nli_strata": load(RESULTS / "nli_strata.json"),
    }
    t = out["TD"]
    out["gate_criteria"] = {
        "TD2_fp_above_10pct": t.get("TD2", {}).get("estimate", 0) > 0.10 if "TD2" in t else None,
        "KC2_fails": (not out["concepts"]["KC2"]["holds"]) if "KC2" in out["concepts"] else None,
        "attribution_only_supported": (
            "supported" in t.get("TD6", {}).get("attribution_only_ig", {})
        )
        if "TD6" in t
        else None,
    }
    (RESULTS / "hypotheses775.json").write_text(
        json.dumps(out, indent=1, sort_keys=True, default=str)
    )
    for k in ("TD1", "TD2", "TD3", "TD4", "TD5", "TD6", "TD7", "TD8", "TD9", "claim_F"):
        print(k, json.dumps(t.get(k))[:300])
    c = out["concepts"]
    print("concepts", json.dumps({k: c.get(k) for k in ("KC1", "KC2", "KC3", "claim_G")})[:800])
    print(
        "central",
        json.dumps(
            {
                k: (
                    v["ig_supported"].get("text")
                    if isinstance(v, dict) and isinstance(v.get("ig_supported"), dict)
                    else v
                )
                for k, v in out["central"].items()
            }
        )[:2500],
    )
    print("special", out["special_tokens"])
    print("interpbench", json.dumps(out["interpbench"])[:800])
    print("gate", out["gate_criteria"])


if __name__ == "__main__":
    main()
