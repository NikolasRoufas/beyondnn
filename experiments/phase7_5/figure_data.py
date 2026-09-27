# ruff: noqa: E501
"""Figure data for Phase 7.5 (request §29): tables only, derived from committed result JSON by
this script; no figure is drawn and no number is typed by hand. Writes CSV files into
``results/figure_data/`` and an index ``results/figure_data/INDEX.json`` (source file and fields
for every table). Run ``evaluate75.py`` first.

Usage (from the repo root): python3 experiments/phase7_5/figure_data.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "figure_data"
sys.path.insert(0, str(HERE))
import analysis75 as AN  # noqa: E402

INDEX: dict[str, dict[str, str]] = {}


def load(name: str) -> Any:
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def write(name: str, header: list[str], rows: list[list[Any]], source: str, shows: str) -> None:
    with (OUT / name).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    INDEX[name] = {"source": source, "shows": shows, "rows": str(len(rows))}


def e1_tables(h: dict[str, Any]) -> None:
    e = h["E1"]
    write(
        "fd1_e1_per_configuration.csv",
        [
            "configuration",
            "role_replacement",
            "tp",
            "necessary",
            "tp_rate",
            "fp",
            "not_necessary",
            "fp_rate",
        ],
        [
            [
                c,
                {"tensor/resample": "PRIMARY", "tensor/mean": "ALTERNATIVE", "zero": "STRESS_TEST"}[
                    c.split("|")[0]
                ],
                v["tp"],
                v["pos"],
                round(v["tp_rate"], 4),
                v["fp"],
                v["neg"],
                round(v["fp_rate"], 4),
            ]
            for c, v in sorted(e["per_configuration"].items())
        ],
        "results/hypotheses75.json -> E1.per_configuration",
        "each single configuration as if it were the only one: SUPPORTS on necessary / not-necessary node instances (held-out reliable cases, clear instances)",
    )
    cases = json.loads((RESULTS / "external_interpbench_heldout.json").read_text())["cases"]
    cases["124"] = load("external_interpbench_heldout_case124_dv1.json")["cases"]["124"]
    rows = []
    for c in sorted(cases, key=int):
        r = cases[c]
        conf = AN.confusion(r["rows"])
        sel = AN.confusion(r["rows"], kinds=("selection",))
        rows.append(
            [
                c,
                r["task"],
                r["benchmark_reliable"],
                r["ll_hl_clean_agreement"],
                conf["tp"],
                conf["fn"],
                conf["tn"],
                conf["fp"],
                conf["pos_other"] + conf["neg_other"],
                conf["ambiguous"],
                sel["tn"],
                sel["fp"],
                sel["tp"],
                sel["fn"],
                "DV-1" if c == "124" else "",
            ]
        )
    write(
        "fd2_e1_confusion_by_case.csv",
        [
            "case",
            "task",
            "reliable",
            "clean_agreement",
            "tp",
            "fn",
            "tn",
            "fp",
            "other",
            "ambiguous",
            "selection_tn",
            "selection_fp",
            "selection_tp",
            "selection_fn",
            "note",
        ],
        rows,
        "results/external_interpbench_heldout.json (+ _case124_dv1.json) -> cases.<c>.rows",
        "PRIMARY-standing confusion per held-out case (nodes and IG top-1 selections)",
    )
    write(
        "fd3_e1_rq3_reversals.csv",
        ["group", "k", "n", "estimate", "wilson_low", "wilson_high"],
        [
            [g, v["k"], v["n"], v.get("estimate"), v.get("low"), v.get("high")]
            for g, v in e["RQ3"].items()
        ],
        "results/hypotheses75.json -> E1.RQ3",
        "share of node instances with a reversal finding, clear vs ambiguous (descriptive)",
    )


def central_tables(h: dict[str, Any]) -> None:
    rows = []
    for site, v in h["C1"]["sites"].items():
        for name, key in (
            ("IG_necessary", "ig_supported"),
            ("R_necessary", "r_supported"),
            ("IG supported with alternative_reverses", "ig_supported_with_alternative_reverses"),
            ("IG configuration-level disagreement", "ig_configuration_level_disagreement"),
        ):
            iv = v[key]
            rows.append(
                [site, name, iv["k"], iv["n"], iv.get("estimate"), iv.get("low"), iv.get("high")]
            )
    write(
        "fd4_central_primary_supported.csv",
        ["site", "quantity", "k", "n", "estimate", "wilson_low", "wilson_high"],
        rows,
        "results/hypotheses75.json -> C1.sites",
        "PRIMARY-supported samples (IG vs R), alternative reversals and configuration-level disagreement per held-out site, with 95% Wilson intervals",
    )
    rows = []
    for model in "ABCD":
        data = load(f"central75_{model}_heldout.json")
        if data is None:
            continue
        for site, s in data["settings"].items():
            for claim in ("ig_necessary", "r_necessary", "g_necessary"):
                for p in s["audit"][claim]["per_sample"]:
                    by_role: dict[str, list[int]] = {}
                    for role, outcome, n in p["outcomes"]:
                        cell = by_role.setdefault(role, [0, 0])
                        cell[1] += n
                        cell[0] += n if outcome == "supports" else 0
                    rows.append(
                        [
                            f"{model}/{site}",
                            claim,
                            p["index"],
                            p["standing"],
                            *[
                                f"{by_role.get(r, [0, 0])[0]}/{by_role.get(r, [0, 0])[1]}"
                                for r in ("primary", "alternative", "stress_test")
                            ],
                            p["tested"],
                            ";".join(p["sensitive_axes"]),
                        ]
                    )
    write(
        "fd5_central_profiles_per_sample.csv",
        [
            "site",
            "claim",
            "sample",
            "primary_standing",
            "primary_supports",
            "alternative_supports",
            "stress_test_supports",
            "tested",
            "sensitive_axes",
        ],
        rows,
        "results/central75_<M>_heldout.json -> settings.<site>.audit.<claim>.per_sample",
        "per-sample sensitivity profiles by role (supporting / tested configurations); descriptive, not a score",
    )


def concept_table(h: dict[str, Any]) -> None:
    rows = []
    for split in ("dev", "heldout"):
        data = load(f"external_tracr_concepts_{split}.json")
        for case, c in data["cases"].items():
            for model, m in c["models"].items():
                for name, k in m["concepts"].items():
                    rows.append(
                        [
                            split,
                            case,
                            model,
                            name,
                            k["known"],
                            round(k["auroc"], 4),
                            k["encoding"]["covariance"],
                            k["encoding"]["isotropic"],
                            k["use"],
                            round(k["use_effect"], 4),
                            k["validation"]["covariance"],
                            k["validation"]["isotropic"],
                            m["audits"]["strict"][name]["standing"],
                            m["audits"]["lenient"][name]["standing"],
                        ]
                    )
    write(
        "fd6_e2_concepts.csv",
        [
            "split",
            "case",
            "model",
            "concept",
            "known",
            "auroc",
            "encoding_cov",
            "encoding_iso",
            "use",
            "use_effect",
            "validation_cov (PRIMARY)",
            "validation_iso (ALTERNATIVE)",
            "audit_strict",
            "audit_lenient",
        ],
        rows,
        "results/external_tracr_concepts_{dev,heldout}.json -> cases.<c>.models",
        "known-positive / known-negative concepts under the frozen concept policy",
    )


def nlp_tables() -> None:
    es = load("esnli.json")
    if es is not None:
        INDEX["fd7_esnli.json"] = {
            "source": "results/esnli.json",
            "shows": "plausibility (token F1 vs annotator 1) and faithfulness (audited standings) side by side; never merged",
            "rows": "-",
        }
        summary = {k: es[k] for k in es if k not in ("samples", "environment")}
        (OUT / "fd7_esnli.json").write_text(
            json.dumps(summary, indent=1, sort_keys=True, default=str)
        )
    rows = []
    for m in "CDE":
        p = load(f"nlp_probes_{m}.json")
        if p is None:
            continue
        for key, v in sorted(p.get("summary", {}).items()):
            m_ = re.search(
                r": (-?[\d.e+-]+) \(n = (\d+); ([\d.]+)% .*\[(-?[\d.e+-]+), (-?[\d.e+-]+)\]", str(v)
            )
            if m_:
                est, n, level, low, high = m_.groups()
                rows.append(
                    [m, key, float(est), float(low), float(high), float(level) / 100, int(n)]
                )
    if rows:
        write(
            "fd8_token_ood.csv",
            ["model", "quantity", "estimate", "low", "high", "level", "n"],
            rows,
            "results/nlp_probes_{C,D,E}.json -> summary",
            "token-perturbation OOD and special-token probes (bootstrap intervals)",
        )


def performance_table() -> None:
    p = load("performance75.json")
    if p is None:
        return
    keys = sorted({k for v in p.values() for k in v})
    write(
        "fd9_performance.csv",
        ["setting", *keys],
        [[s, *[v.get(k) for k in keys]] for s, v in sorted(p.items())],
        "results/performance75.json",
        "end-to-end cost per stage (single runs, one machine)",
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    h = load("hypotheses75.json")
    e1_tables(h)
    central_tables(h)
    concept_table(h)
    nlp_tables()
    performance_table()
    (OUT / "INDEX.json").write_text(json.dumps(INDEX, indent=1, sort_keys=True))
    for name, v in INDEX.items():
        print(name, v["rows"])


if __name__ == "__main__":
    main()
