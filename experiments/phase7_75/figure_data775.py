# ruff: noqa: E501
"""Figure data for Phase 7.75: CSV tables derived from committed result JSON by this script
(no figure is drawn, no number typed by hand). Run ``evaluate775.py`` first.

Usage (from the repo root): python3 experiments/phase7_75/figure_data775.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "figure_data"
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


def main() -> None:
    OUT.mkdir(exist_ok=True)
    h = load("hypotheses775.json")
    td = h["TD"]
    if "node_truth_table" in td:
        rows = [
            [
                kind,
                truth,
                *[
                    cells[s]
                    for s in ("supported", "contradicted", "unsupported", "inconclusive", "other")
                ],
            ]
            for kind, table in (
                ("node", td["node_truth_table"]),
                ("ig_selection", td["selection_truth_table"]),
            )
            for truth, cells in table.items()
        ]
        write(
            "fd10_td_truth_table.csv",
            [
                "claims",
                "truth",
                "supported",
                "contradicted",
                "unsupported",
                "inconclusive",
                "other",
            ],
            rows,
            "hypotheses775.json -> TD.node_truth_table / selection_truth_table",
            "every truth x PRIMARY-standing cell on the held-out TD programs (program-defined truth)",
        )
        write(
            "fd11_td_components.csv",
            ["component", "truth", "supported", "n"],
            [[k, v["truth"], v["supported"], v["n"]] for k, v in sorted(td["components"].items())],
            "hypotheses775.json -> TD.components",
            "per component: samples PRIMARY-SUPPORTED (mechanism-level view)",
        )
        write(
            "fd12_td_per_configuration.csv",
            [
                "configuration",
                "supports_on_used",
                "used",
                "supports_on_decoy",
                "decoy",
                "supports_on_empty",
                "empty",
            ],
            [
                [c, *v["known_true"], *v["known_false"], *v["empty"]]
                for c, v in sorted(td["per_configuration"].items())
            ],
            "hypotheses775.json -> TD.per_configuration",
            "each configuration as if it were the only one, under program-defined truth (zero ablation supports every decoy)",
        )
    c = h["concepts"]
    if "table" in c:
        rows = []
        for k, v in sorted(c["table"].items()):
            if isinstance(v, dict):
                rows.append(
                    [
                        k,
                        v["known"],
                        v["encoding"]["covariance"],
                        v["rel"]["use"],
                        round(v["rel"]["use_effect"], 4),
                        round(v["rel"]["min_change"], 4),
                        v["rel"]["validation"]["covariance"],
                        v["abs"]["validation"]["covariance"],
                        v["audit"]["rel|strict"],
                    ]
                )
        write(
            "fd13_td_concepts.csv",
            [
                "concept",
                "known",
                "encoding_cov",
                "use_rel",
                "use_effect",
                "rel_min_change",
                "validation_rel_cov",
                "validation_abs_cov",
                "audit_rel_strict",
            ],
            rows,
            "hypotheses775.json -> concepts.table",
            "known used / decodable-unused / negative concepts under the frozen relative rule (and the absolute 0.1 for comparison)",
        )
    cal = load("concept_calibration.json")
    if cal:
        rows = [
            [
                g["w"],
                g["s"],
                n,
                g[n]["truth"],
                g[n]["abs"]["validation"],
                g[n]["rel"]["validation"],
                g[n]["rel"]["effect"],
                g["rel_min_change"],
            ]
            for g in cal["grid"]
            for n in ("used", "decodable_unused", "negative")
        ]
        write(
            "fd14_concept_calibration.csv",
            [
                "use_weight",
                "output_scale",
                "concept",
                "truth",
                "abs_rule",
                "rel_rule",
                "effect",
                "rel_min_change",
            ],
            rows,
            "concept_calibration.json -> grid",
            "constructed models: the absolute rule changes with output scale, the relative rule does not",
        )
    rows = []
    for k, v in sorted(h["central"].items()):
        if isinstance(v, dict) and isinstance(v.get("ig_supported"), dict):
            rows.append(
                [
                    k,
                    v["n"],
                    v["ig_supported"]["k"],
                    v["r_supported"]["k"],
                    v["g_supported"],
                    v["ig_supported_alternative_reverses"],
                    v["ig_configuration_level_disagreement"],
                    v["ig_control_unattainable"],
                    v["ig_effect_without_competitive_advantage"],
                ]
            )
    if rows:
        write(
            "fd15_central_775.csv",
            [
                "site/eligibility",
                "n",
                "ig_supported",
                "r_supported",
                "g_supported",
                "ig_supported_with_alternative_reverses",
                "ig_config_disagreement",
                "ig_control_unattainable",
                "ig_effect_without_competitive_advantage",
            ],
            rows,
            "hypotheses775.json -> central",
            "central held-out re-runs under ADR-053/054 (all units vs content tokens)",
        )
    n = h.get("nli_strata")
    if n:
        rows = [
            [a, stratum, cells.get("supported", 0), sum(cells.values())]
            for a, strata in n["necessity_by_stratum"].items()
            for stratum, cells in strata.items()
        ]
        write(
            "fd16_nli_strata.csv",
            ["selection", "stratum", "supported", "n"],
            rows,
            "nli_strata.json",
            "e-SNLI necessity supports by empty-premise predictability (analysis only)",
        )
    (OUT / "INDEX.json").write_text(json.dumps(INDEX, indent=1, sort_keys=True))
    for name, v in INDEX.items():
        print(name, v["rows"])


if __name__ == "__main__":
    main()
