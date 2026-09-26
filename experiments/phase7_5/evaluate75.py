# ruff: noqa: E501
"""Evaluate the pre-registered Phase 7.5 hypotheses (frozen policy §9) on the held-out results.

Pure post-processing of committed result JSON: no model is run, nothing is tuned. Writes
``results/hypotheses75.json``. Missing inputs (e.g. NLP results not yet run) are reported as
``"not_available"``, never guessed.

Usage (from the repo root): uv run --python 3.12 --with-editable . python experiments/phase7_5/evaluate75.py
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
sys.path.insert(0, str(HERE))
import analysis75 as AN  # noqa: E402

AU = bnn.audits
BONFERRONI_LEVEL = 0.9875  # frozen policy §5: 4 confirmatory interval tests
CENTRAL_SITES = [("A", "input"), ("A", "net.1"), ("B", "pixels"), ("B", "relu2"), ("C", "tokens")]
REVERSAL_CODES = {"alternative_reverses", "stress_test_reverses"}


def load(name: str) -> Any:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def interval(k: int, n: int, quantity: str, unit: str, level: float = 0.95) -> dict[str, Any]:
    if n == 0:
        return {"k": 0, "n": 0, "text": "n = 0"}
    iv = AU.wilson(k, n, quantity=quantity, unit=unit, level=level)
    return {"k": k, "n": n, "estimate": iv.estimate, "low": iv.low, "high": iv.high, "level": level, "text": f"{k}/{n} = {iv.estimate:.4f} ({level:.2%} Wilson [{iv.low:.4f}, {iv.high:.4f}])"}


# --------------------------------------------------------------------------- E1 (InterpBench)


def e1_cases() -> dict[str, Any]:
    cases = dict(load("external_interpbench_heldout.json")["cases"])
    rerun = load("external_interpbench_heldout_case124_dv1.json")["cases"]["124"]
    cases["124"] = dict(rerun, rerun_after="DV-1")
    return cases


def e1(cases: dict[str, Any]) -> dict[str, Any]:
    rows = AN.reliable_rows(cases)
    unreliable = sorted(c for c, r in cases.items() if not r.get("benchmark_reliable", False))
    conf = AN.confusion(rows)
    cfg = AN.configuration_confusion(rows)
    heads = AN.configuration_confusion(rows, kinds=("head",))
    sel = AN.confusion(rows, kinds=("selection",))
    pos = conf["tp"] + conf["fn"] + conf["pos_other"]
    neg = conf["tn"] + conf["fp"] + conf["neg_other"]
    out: dict[str, Any] = {
        "reliable_cases": sorted({r["case"] for r in rows}, key=int),
        "unreliable_cases": unreliable,
        "case_124": "re-run after DV-1",
        "node_confusion": conf,
        "selection_confusion": sel,
    }
    out["EH1"] = interval(conf["tp"], pos, "PRIMARY SUPPORTED on necessary", "node instances") | {"threshold": ">= 0.90"}
    out["EH1"]["holds"] = out["EH1"]["estimate"] >= 0.90
    out["EH2"] = interval(conf["fp"], neg, "PRIMARY SUPPORTED on not-necessary", "node instances") | {"threshold": "<= 0.05"}
    out["EH2"]["holds"] = out["EH2"]["estimate"] <= 0.05
    sel_neg = sel["tn"] + sel["fp"] + sel["neg_other"]
    out["EH3"] = interval(sel["tn"], sel_neg, "IG top-1 wrong-head selections CONTRADICTED", "selection instances") | {"threshold": ">= 0.90"}
    out["EH3"]["holds"] = sel_neg > 0 and out["EH3"]["estimate"] >= 0.90
    count_cfg = [c for c in heads if "|count|recorded" in c and c.startswith("tensor/resample")]
    cell = heads[count_cfg[0]]
    head_pos = cell["pos"]
    contradicts = sum(1 for r in rows if r["kind"] == "head" and r["gt"] == AN.NECESSARY and r["configs"].get(count_cfg[0]) == "contradicts")
    out["EH4"] = interval(contradicts, head_pos, f"{count_cfg[0]} CONTRADICTS necessary heads", "head instances") | {"configuration": count_cfg[0], "threshold": ">= 0.50"}
    out["EH4"]["holds"] = out["EH4"]["estimate"] >= 0.50
    zero = cfg["zero|none|recorded"]
    primary_rate = conf["fp"] / neg
    out["EH5"] = {
        "stress_zero": interval(zero["fp"], zero["neg"], "zero|none|recorded SUPPORTS not-necessary", "node instances"),
        "primary_fp_rate": primary_rate,
        "threshold": "zero FP rate >= PRIMARY + 0.05",
    }
    out["EH5"]["holds"] = zero["fp"] / zero["neg"] >= primary_rate + 0.05
    out["per_configuration"] = {c: v | {"tp_rate": v["tp"] / v["pos"], "fp_rate": v["fp"] / v["neg"]} for c, v in sorted(cfg.items())}
    ao = Counter()
    for c in out["reliable_cases"]:
        for claim, dist in cases[c]["attribution_only"].items():
            if "ig_top" in claim:
                for standing, n in dist.items():
                    ao[standing] += n
    out["EH6"] = {"selection_claims_attribution_only": dict(ao), "holds": set(ao) == {"unsupported"}}
    out["attribution_only_any_supported"] = any(
        dist.get("supported", 0) for c in cases.values() if "attribution_only" in c for dist in c["attribution_only"].values()
    )
    # RQ3 (descriptive): reversal findings on clear vs ambiguous, reliable vs unreliable
    all_rows = [dict(r, case=c, reliable=v.get("benchmark_reliable", False)) for c, v in cases.items() if "rows" in v for r in v["rows"] if r["kind"] in ("head", "mlp")]

    def rate(sub: list[dict[str, Any]], label: str) -> dict[str, Any]:
        k = sum(1 for r in sub if REVERSAL_CODES & set(r["codes"]))
        return interval(k, len(sub), f"node instances with a reversal finding ({label})", "node instances")

    out["RQ3"] = {
        "clear (reliable)": rate([r for r in all_rows if r["reliable"] and r["gt"] != AN.AMBIGUOUS], "clear"),
        "ambiguous (reliable)": rate([r for r in all_rows if r["reliable"] and r["gt"] == AN.AMBIGUOUS], "ambiguous"),
        "reliable cases": rate([r for r in all_rows if r["reliable"]], "reliable"),
        "unreliable cases (held-out)": rate([r for r in all_rows if not r["reliable"]], "unreliable"),
    }
    dev = load("external_interpbench_dev.json")["cases"]
    dev_rows = [dict(r, reliable=v.get("benchmark_reliable", False)) for v in dev.values() if "rows" in v for r in v["rows"] if r["kind"] in ("head", "mlp")]
    out["RQ3"]["development: reliable case 7"] = rate([r for r in dev_rows if r["reliable"]], "dev reliable")
    out["RQ3"]["development: unreliable case 13"] = rate([r for r in dev_rows if not r["reliable"]], "dev unreliable")
    # every PRIMARY FP / FN (qualitative inspection list) and where the STRESS/ALT FPs sit
    out["primary_errors"] = [
        {k: r[k] for k in ("case", "claim", "sample", "gt", "standing", "codes")}
        for r in rows
        if r["kind"] in ("head", "mlp") and ((r["gt"] == AN.NECESSARY and r["standing"] == "contradicted") or (r["gt"] == AN.NOT_NECESSARY and r["standing"] == "supported"))
    ]
    out["other_on_clear"] = [
        {k: r[k] for k in ("case", "claim", "sample", "gt", "standing", "codes")}
        for r in rows
        if r["kind"] in ("head", "mlp") and r["gt"] != AN.AMBIGUOUS and r["standing"] not in ("supported", "contradicted")
    ]
    zero_fp = Counter((r["case"], r["claim"]) for r in rows if r["kind"] in ("head", "mlp") and r["gt"] == AN.NOT_NECESSARY and r["configs"].get("zero|none|recorded") == "supports")
    out["stress_zero_fp_by_node"] = [{"case": c, "claim": cl, "samples": n} for (c, cl), n in zero_fp.most_common()]
    mean_fp = Counter((r["case"], r["claim"]) for r in rows if r["kind"] in ("head", "mlp") and r["gt"] == AN.NOT_NECESSARY and r["configs"].get("tensor/mean|none|recorded") == "supports")
    out["alternative_mean_fp_by_node"] = [{"case": c, "claim": cl, "samples": n} for (c, cl), n in mean_fp.most_common()]
    return out


# --------------------------------------------------------------------------- E2 (Tracr concepts)


def e2() -> dict[str, Any]:
    case = load("external_tracr_concepts_heldout.json")["cases"]["39"]
    out: dict[str, Any] = {"outcomes": {}}
    ke1 = True
    for model, m in case["models"].items():
        for name, c in m["concepts"].items():
            out["outcomes"][f"{model}/{name}"] = {
                "known": c["known"], "encoding": c["encoding"], "use": c["use"], "validation": c["validation"],
                "audit": {cap: m["audits"][cap][name]["standing"] for cap in m["audits"]},
            }
            if c["known"] != "positive":
                ke1 &= all(v != "validated_concept" for v in c["validation"].values())
                ke1 &= all(m["audits"][cap][name]["standing"] != "supported" for cap in m["audits"])
    hl = case["models"]["tracr_hl"]["concepts"]["K+"]
    ll = case["models"]["siit_ll"]["concepts"]["K+"]
    out["KE1"] = {"holds": ke1}
    out["KE2"] = {"tracr_K+_covariance": hl["validation"]["covariance"], "holds": hl["validation"]["covariance"] != "validated_concept"}
    out["KE3"] = {"siit_K+": ll["validation"], "encoding": ll["encoding"], "use": ll["use"], "use_effect": ll["use_effect"]}
    return out


# --------------------------------------------------------------------------- C1 (central)


def claim_of(model: str, site: str, split: str, name: str) -> dict[str, Any] | None:
    data = load(f"central75_{model}_{split}.json")
    if data is None or site not in data["settings"]:
        return None
    return data["settings"][site]["audit"][name]


def central() -> dict[str, Any]:
    sites = [(m, s) for m, s in CENTRAL_SITES]
    d = load("central75_D_heldout.json")
    if d is not None:
        sites += [("D", s) for s in d["settings"]]
    out: dict[str, Any] = {"sites": {}}
    ch7_qualifying, ch7_pass = [], []
    ch8, ch9 = True, True
    for model, site in sites:
        ig = claim_of(model, site, "heldout", "ig_necessary")
        r = claim_of(model, site, "heldout", "r_necessary")
        if ig is None or r is None:
            continue
        n = len(ig["per_sample"])
        ig_sup = [s for s in ig["per_sample"] if s["standing"] == "supported"]
        rev = sum(1 for s in ig_sup if "alternative_reverses" in s["codes"])
        r_sup = sum(1 for s in r["per_sample"] if s["standing"] == "supported")
        disagree = sum(1 for s in ig["per_sample"] if "configuration_level_disagreement" in s["codes"])
        entry = {
            "n": n,
            "ig_necessary_distribution": ig["distribution"],
            "r_necessary_distribution": r["distribution"],
            "ig_supported": interval(len(ig_sup), n, "IG_necessary PRIMARY SUPPORTED", "samples"),
            "r_supported": interval(r_sup, n, "R_necessary PRIMARY SUPPORTED", "samples"),
            "ig_supported_with_alternative_reverses": interval(rev, len(ig_sup), "PRIMARY-supported IG samples with alternative_reverses", "samples"),
            "ig_configuration_level_disagreement": interval(disagree, n, "IG samples with configuration-level disagreement", "samples"),
            "all_claims": {name: c["distribution"] for name, c in load(f"central75_{model}_heldout.json")["settings"][site]["audit"].items()},
        }
        out["sites"][f"{model}/{site}"] = entry
        if model != "D" and len(ig_sup) >= 5:
            ch7_qualifying.append(f"{model}/{site}")
            if rev / len(ig_sup) >= 0.20:
                ch7_pass.append(f"{model}/{site}")
        ch8 &= r_sup / n <= 0.05
        ch9 &= len(ig_sup) >= r_sup
    out["CH7"] = {"qualifying_sites": ch7_qualifying, "passing_sites": ch7_pass, "holds": len(ch7_pass) >= 3}
    out["CH8"] = {"holds": ch8, "D_included": d is not None}
    out["CH9"] = {"holds": ch9, "D_included": d is not None}
    return out


# --------------------------------------------------------------------------- NLP


def nlp() -> dict[str, Any]:
    out: dict[str, Any] = {}
    es = load("esnli.json")
    probes = load("nlp_probes.json")
    if es is None:
        out["N1"] = out["N2"] = "not_available"
    else:
        out["N1"] = es.get("N1", "see esnli.json")
        out["N2"] = es.get("N2", "see esnli.json")
    if probes is None:
        out["N3"] = out["N4"] = "not_available"
    else:
        out["N3"] = probes.get("N3", "see nlp_probes.json")
        out["N4"] = probes.get("N4", "see nlp_probes.json")
    return out


def main() -> None:
    cases = e1_cases()
    out = {
        "policy": "docs/PHASE_7_5_FROZEN_POLICY.md §9 (frozen at ee3f91e; DV-1)",
        "E1": e1(cases),
        "E2": e2(),
        "C1": central(),
        "NLP": nlp(),
    }
    e = out["E1"]
    out["gate_criteria_§10"] = {
        "EH2_primary_fp_rate_above_10pct": e["EH2"]["estimate"] > 0.10,
        "EH1_below_80pct": e["EH1"]["estimate"] < 0.80,
        "attribution_only_claim_supported": e["attribution_only_any_supported"],
        "KE1_fails": not out["E2"]["KE1"]["holds"],
    }
    (RESULTS / "hypotheses75.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "E1"}, indent=1, default=str)[:6000])
    print({k: (v.get("holds"), v.get("text")) for k, v in e.items() if k.startswith("EH")})
    print("RQ3", {k: v["text"] for k, v in e["RQ3"].items()})
    print("primary_errors", len(e["primary_errors"]), "other_on_clear", len(e["other_on_clear"]), Counter(r["standing"] for r in e["other_on_clear"]))


if __name__ == "__main__":
    main()
