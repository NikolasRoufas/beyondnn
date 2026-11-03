"""TNNLS revision: LaTeX tables of paper_tnnls/ generated from committed result JSON.

Outputs paper_tnnls/tables/{tab_decomp,tab_cmp,tab_seeds,tab_cost}.tex.
Usage: python3 experiments/tnnls_revision/figures/make_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "experiments/tnnls_revision"
OUT = ROOT / "paper_tnnls/tables"
OUT.mkdir(parents=True, exist_ok=True)
SITES = [
    ("A", "input", "MLP input"),
    ("A", "net.1", "MLP hidden"),
    ("B", "pixels", "CNN pixels"),
    ("B", "relu2", "CNN relu2"),
    ("C", "tokens", "BERT-tiny"),
    ("D", "tokens", "BERT-base"),
]


def tab_decomp(agg: dict) -> None:
    lines = []
    tot = [0] * 9
    for m, s, lab in SITES:
        e = agg["runs"].get(f"S2_{m}_frozen")
        if not e or s not in e.get("decomposition", {}):
            continue
        d = e["decomposition"][s]
        sa = d["alternative_single_axis_reversal"]
        n_site = e["sites"][s]["n"]
        vals = [
            n_site,
            d["units"],
            d["alternative_any_reversal"],
            d["alternative_fixed_subject_reversal"],
            sa["replacement"],
            sa["null"],
            sa["threshold"],
            sa["k"],
            d["alternative_reversed_only_by_subject_change"],
        ]
        tot = [a + b for a, b in zip(tot, vals, strict=True)]
        lines.append(f"{lab} & " + " & ".join(str(v) for v in vals) + r" \\")
    lines.append(r"\midrule")
    lines.append("Total & " + " & ".join(str(v) for v in tot) + r" \\")
    body = "\n".join(lines)
    (OUT / "tab_decomp.tex").write_text(
        r"""\begin{table}[t]
\centering
\caption{Decomposition of ALTERNATIVE reversals of PRIMARY-supported IG necessity claims (re-run of the frozen central design, held-out samples). \emph{Any}: $\ge 1$ ALTERNATIVE configuration contradicts. \emph{Fixed}: a contradicting ALTERNATIVE keeps $k$, hence the selected units. Single-axis columns: contradiction by a configuration that differs from PRIMARY only on that axis. \emph{$k$ only}: reversed only by configurations that change $k$. Counts per sample; samples are nested in four checkpoints.}
\label{tab:decomp}
\setlength{\tabcolsep}{2.6pt}
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrrrrrr}
\toprule
 & & & \multicolumn{2}{c}{reversed} & \multicolumn{4}{c}{single-axis reversal} & \\
\cmidrule(lr){4-5}\cmidrule(lr){6-9}
site & $n$ & sup. & any & fixed & repl. & null & thr. & $k$ & $k$ only \\
\midrule
"""
        + body
        + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    )


def tab_cmp(t: dict) -> None:
    names = [
        ("F1_attribution_as_necessity", "F1 attribution as necessity"),
        ("F2_wrong_checkpoint", "F2 wrong checkpoint"),
        ("F3_wrong_input", "F3 wrong input"),
        ("F4_wrong_selection_size", "F4 wrong selection size"),
        ("F5_omitted_controls", "F5 omitted controls"),
        ("F6_configuration_dependence", "F6 configuration dependence"),
        ("F7_tampered_outcome", "F7 tampered outcome"),
    ]
    rows = []
    for key, lab in names:
        e = t["table"][key]
        b1, bn = e["b1"], e["beyondnn"]
        bn_s = f"{bn['detected']}/{e['affected']}" + (
            r"$^\ast$" if bn["study_level_refusal"] else ""
        )
        rows.append(
            f"{lab} & {e['affected']} & {e['erroneous_b0_support_conclusions']} & 0 & {b1['detected']}/{e['affected']} ({b1['false_alerts']}) & {bn_s} ({bn['false_alerts'] if not bn['study_level_refusal'] else '--'}) \\\\"
        )
    loc = t["burden_loc"]
    (OUT / "tab_cmp.tex").write_text(
        r"""\begin{table}[t]
\centering
\caption{Injected workflow faults on identical measurements (MLP, 20 held-out samples). Detected/affected, false alerts in parentheses. B0: single-configuration report. B1: conventional log with """
        + str(loc["b1_checks"])
        + r""" lines of fault-specific checks. BeyondNN: the frozen plan, no fault-specific code. $^\ast$Study-level refusal: the edited trace fails its content hash and the evidence set is not loaded.}
\label{tab:cmp}
\setlength{\tabcolsep}{2.4pt}
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrr}
\toprule
fault & aff. & wrong sup. & B0 & B1 & BeyondNN \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    )


def tab_seeds(agg: dict) -> None:
    rows = []
    for m, s, lab in SITES[:4]:
        cells = []
        for seed in range(5):
            e = agg["runs"][f"S4_{m}_seed{seed}"]["sites"][s]
            cells.append(
                f"{e['ig_supported']}/{e['ig_supported_alternative_reverses']}/{e['r_supported']}"
            )
        fz = agg["runs"][f"S2_{m}_frozen"]["sites"][s] if f"S2_{m}_frozen" in agg["runs"] else None
        f = (
            f"{fz['ig_supported']}/{fz['ig_supported_alternative_reverses']}/{fz['r_supported']}"
            if fz
            else "--"
        )
        rows.append(f"{lab} & {f} & " + " & ".join(cells) + r" \\")
    (OUT / "tab_seeds.tex").write_text(
        r"""\begin{table}[t]
\centering
\caption{Independently trained checkpoints (Linux x86-64; seeds 0--4) and the frozen macOS checkpoint. Each cell: IG necessity PRIMARY-supported / of which ALTERNATIVE-reversed / random selection supported, out of 51 (MLP) or 60 (CNN) held-out samples drawn by the same rule per checkpoint.}
\label{tab:seeds}
\setlength{\tabcolsep}{2.2pt}
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
site & frozen & seed 0 & seed 1 & seed 2 & seed 3 & seed 4 \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    )


def tab_cost(agg: dict, cmp_cost: dict) -> None:
    rows = []
    for m, s, lab in SITES:
        e = agg["runs"].get(f"S2_{m}_frozen")
        if not e or "timing" not in e or s not in e["timing"]:
            continue
        t = e["timing"][s]
        n = t["samples"]
        ev, au, ve = t["evidence_seconds"] / n, t["audit_seconds"] / n, t["verify_seconds"] / n
        rep = t["report_bytes"] / 1e6
        rows.append(
            f"{lab} & {n} & {t['results'] // n} & {ev:.1f} & {au:.3f} & {ve:.3f} & {100 * (au + ve) / (ev + au + ve):.1f} & {rep:.1f} & {t['peak_rss_mb_max_process']:.0f} \\\\"
        )
    (OUT / "tab_cost.tex").write_text(
        r"""\begin{table}[t]
\centering
\caption{Cost per held-out sample of the full central design (one CPU thread, Linux x86-64). Evidence: model execution and BeyondNN record construction for all results of the sample. Audit and verify: model-free. Share: audit + verify as \% of the total. Report: size of the site's report (MB). RSS: peak resident memory of one process (MB).}
\label{tab:cost}
\setlength{\tabcolsep}{2.2pt}
\footnotesize
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrrrrrrr}
\toprule
site & $n$ & res. & evid. (s) & audit (s) & verify (s) & share & report & RSS \\
\midrule
"""
        + "\n".join(rows)
        + r"""
\bottomrule
\end{tabular}}
\end{table}
"""
    )


def numbers(agg: dict, cmp: dict, ib: dict) -> None:
    """Key numbers of the text as LaTeX macros (paper_tnnls/tables/numbers.tex)."""
    m: dict[str, Any] = {}
    tot = {
        "Sup": 0,
        "Rev": 0,
        "Fixed": 0,
        "KOnly": 0,
        "Repl": 0,
        "Null": 0,
        "Thr": 0,
        "KSingle": 0,
        "NotRev": 0,
    }
    diffs = 0
    for model, site, _ in SITES:
        e = agg["runs"].get(f"S2_{model}_frozen")
        if not e or site not in e.get("decomposition", {}):
            continue
        d = e["decomposition"][site]
        sa = d["alternative_single_axis_reversal"]
        for k, v in (
            ("Sup", d["units"]),
            ("Rev", d["alternative_any_reversal"]),
            ("Fixed", d["alternative_fixed_subject_reversal"]),
            ("KOnly", d["alternative_reversed_only_by_subject_change"]),
            ("Repl", sa["replacement"]),
            ("Null", sa["null"]),
            ("Thr", sa["threshold"]),
            ("KSingle", sa["k"]),
            ("NotRev", d["alternative_not_reversed"]),
        ):
            tot[k] += v
        tag = {"A": "Mlp", "B": "Cnn", "C": "Tiny", "D": "Base"}[model] + (
            "Hid" if site in ("net.1", "relu2") else ""
        )
        m[f"{tag}Sup"], m[f"{tag}Rev"], m[f"{tag}Fixed"], m[f"{tag}KOnly"] = (
            d["units"],
            d["alternative_any_reversal"],
            d["alternative_fixed_subject_reversal"],
            d["alternative_reversed_only_by_subject_change"],
        )
        for c in e.get("vs_frozen_macos", {}).get(site, {}).values():
            diffs += len(c["standing_diff"]) + len(c["codes_or_outcomes_diff"])
    for k, v in tot.items():
        m[f"Tot{k}"] = v
    m["ReproDiffs"] = diffs
    c = cmp["cost"]
    m["CostBnnMs"] = f"{1000 * c['median_beyondnn_s']:.1f}"
    m["CostPlainMs"] = f"{1000 * c['median_plain_s']:.2f}"
    m["CostRatio"] = f"{c['median_beyondnn_s'] / c['median_plain_s']:.0f}"
    m["BOneLoc"], m["PlanLoc"] = (
        cmp["burden_loc"]["b1_checks"],
        cmp["burden_loc"]["beyondnn_plan_and_roles_reused"],
    )
    p = ib["primary_vs_spec_input"]
    m["IbTp"], m["IbPos"], m["IbFp"], m["IbNeg"] = (
        p["tp_rate"][0],
        p["tp_rate"][1],
        p["fp_rate"][0],
        p["fp_rate"][1],
    )
    m["IbFpPct"] = f"{100 * p['fp_rate'][0] / p['fp_rate'][1]:.2f}"
    z = ib["per_configuration_vs_spec_input"]["zero|none|recorded"]["not_necessary"]
    m["IbZeroFp"], m["IbZeroFpPct"] = z[0], f"{100 * z[0] / z[1]:.1f}"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (OUT / "numbers.tex").write_text(
        "% generated by make_tables.py -- do not edit\n" + "\n".join(lines) + "\n"
    )


def main() -> None:
    agg = json.loads((R / "results/aggregate.json").read_text())
    tab_decomp(agg)
    tab_cmp(json.loads((R / "stage3/results/comparative_tally.json").read_text()))
    tab_seeds(agg)
    tab_cost(agg, {})
    numbers(
        agg,
        json.loads((R / "stage3/results/comparative_tally.json").read_text()),
        json.loads((R / "stage3/results/interpbench_specification_labels.json").read_text()),
    )
    print(sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
