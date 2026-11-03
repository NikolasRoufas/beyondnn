"""TNNLS revision: every empirical figure of paper_tnnls/, generated only from committed result
JSON (no number is typed in by hand).

Inputs:
  experiments/tnnls_revision/results/aggregate.json          (Stage 2 / 4; aggregate.py)
  experiments/tnnls_revision/stage1/results/recalculation.json (TD per configuration)
  experiments/tnnls_revision/stage3/results/interpbench_specification_labels.json
Outputs: paper_tnnls/figures/fig_{decomposition,reference,seeds,cost}.pdf

Usage: uv run --with matplotlib python experiments/tnnls_revision/figures/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
R = ROOT / "experiments/tnnls_revision"
OUT = ROOT / "paper_tnnls/figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "pdf.fonttype": 42,
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
W1 = 3.45  # IEEE single column (in)
DARK, MID, LIGHT, ACCENT = "#2b2b2b", "#7a7a7a", "#d0d0d0", "#1f5aa6"
SITE_LABEL = {
    ("A", "input"): "MLP input",
    ("A", "net.1"): "MLP hidden",
    ("B", "pixels"): "CNN pixels",
    ("B", "relu2"): "CNN relu2",
    ("C", "tokens"): "BERT-tiny",
    ("D", "tokens"): "BERT-base",
}


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def fig_decomposition(agg: dict) -> None:
    rows = []
    for (m, s), lab in SITE_LABEL.items():
        e = agg["runs"].get(f"S2_{m}_frozen")
        if not e or s not in e.get("decomposition", {}):
            continue
        d = e["decomposition"][s]
        rows.append(
            (
                lab,
                d["units"],
                d["alternative_fixed_subject_reversal"],
                d["alternative_reversed_only_by_subject_change"],
                d["alternative_not_reversed"],
            )
        )
    fig, ax = plt.subplots(figsize=(W1, 2.15))
    y = list(range(len(rows)))[::-1]
    for yi, (lab, n, fixed, subj, none) in zip(y, rows, strict=True):
        ax.barh(yi, fixed, color=ACCENT, edgecolor="white", linewidth=0.4)
        ax.barh(yi, subj, left=fixed, color=MID, edgecolor="white", linewidth=0.4)
        ax.barh(yi, none, left=fixed + subj, color=LIGHT, edgecolor="white", linewidth=0.4)
        ax.text(n + 0.3, yi, f"{fixed + subj}/{n}", va="center", fontsize=6.5)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("samples with IG necessity PRIMARY-SUPPORTED")
    ax.set_xlim(0, max(r[1] for r in rows) + 3)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=ACCENT, label="reversed with the selected units fixed"),
            Patch(color=MID, label="reversed only when k changes the units"),
            Patch(color=LIGHT, label="no ALTERNATIVE reversal"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.42, -0.30),
        ncol=1,
        frameon=False,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_reference(recalc: dict, ib: dict) -> None:
    td = recalc["manuscript_checks"]["M10_td_configuration_rates"]
    cfgs = [
        ("tensor/resample|none|recorded", "resample\n(P)"),
        ("tensor/mean|none|recorded", "mean\n(A)"),
        ("zero|none|recorded", "zero\n(S)"),
    ]

    def pct(s: str) -> float:
        k, n = (int(v) for v in s.split("/"))
        return 100 * k / n

    fig, axes = plt.subplots(1, 2, figsize=(W1, 2.0), sharey=True)
    for ax, title, pos, neg in (
        (
            axes[0],
            "(a) Tracr: program labels",
            [pct(td[f"{c}|known_true"]) for c, _ in cfgs],
            [pct(td[f"{c}|known_false"]) for c, _ in cfgs],
        ),
        (
            axes[1],
            "(b) InterpBench: spec. labels",
            [
                100
                * ib["per_configuration_vs_spec_input"][c]["necessary"][0]
                / ib["per_configuration_vs_spec_input"][c]["necessary"][1]
                for c, _ in cfgs
            ],
            [
                100
                * ib["per_configuration_vs_spec_input"][c]["not_necessary"][0]
                / ib["per_configuration_vs_spec_input"][c]["not_necessary"][1]
                for c, _ in cfgs
            ],
        ),
    ):
        x = range(len(cfgs))
        ax.bar([i - 0.19 for i in x], pos, width=0.36, color=DARK, label="used / necessary")
        ax.bar(
            [i + 0.19 for i in x],
            neg,
            width=0.36,
            color=LIGHT,
            edgecolor=MID,
            linewidth=0.4,
            label="decoy / not necessary",
        )
        for i, (p, q) in enumerate(zip(pos, neg, strict=True)):
            ax.text(i - 0.19, p + 2, f"{p:.0f}", ha="center", fontsize=5.5)
            ax.text(
                i + 0.19, q + 2, f"{q:.1f}" if 0 < q < 10 else f"{q:.0f}", ha="center", fontsize=5.5
            )
        ax.set_xticks(list(x), [lab for _, lab in cfgs])
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 112)
    axes[0].set_ylabel("instances SUPPORTED (%)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.25), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.88, bottom=0.30, wspace=0.12)
    fig.savefig(OUT / "fig_reference.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_seeds(agg: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(W1, 2.6))
    for ax, (m, s) in zip(
        axes.flat, [("A", "input"), ("A", "net.1"), ("B", "pixels"), ("B", "relu2")], strict=True
    ):
        ig, rr, rev, fixed = [], [], [], []
        for seed in range(5):
            run = agg["runs"][f"S4_{m}_seed{seed}"]
            e = run["sites"][s]
            ig.append(e["ig_supported"])
            rr.append(e["r_supported"])
            rev.append(e["ig_supported_alternative_reverses"])
            fixed.append(run["decomposition"][s]["alternative_fixed_subject_reversal"])
        x = list(range(5))
        ax.bar(
            [i - 0.2 for i in x],
            ig,
            width=0.38,
            color=LIGHT,
            edgecolor=DARK,
            linewidth=0.5,
            label="IG supported",
        )
        ax.bar([i - 0.2 for i in x], rev, width=0.38, color=MID, label="alt. reversed (any)")
        ax.bar(
            [i - 0.2 for i in x],
            fixed,
            width=0.38,
            color=ACCENT,
            label="alt. reversed, units fixed",
        )
        ax.bar(
            [i + 0.2 for i in x],
            rr,
            width=0.38,
            color="white",
            edgecolor=DARK,
            hatch="////",
            linewidth=0.5,
            label="random supported",
        )
        ax.set_title(SITE_LABEL[(m, s)], loc="left")
        ax.set_xticks(x, [str(i) for i in x])
        ax.yaxis.get_major_locator().set_params(integer=True)
    for ax in axes[1]:
        ax.set_xlabel("training seed")
    for ax in axes[:, 0]:
        ax.set_ylabel("samples")
    axes[1, 0].legend(loc="upper center", bbox_to_anchor=(1.12, -0.42), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.93, bottom=0.25, wspace=0.28, hspace=0.55)
    fig.savefig(OUT / "fig_seeds.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_cost(agg: dict) -> None:
    rows = []
    for (m, s), lab in SITE_LABEL.items():
        e = agg["runs"].get(f"S2_{m}_frozen")
        if not e or "timing" not in e or s not in e["timing"]:
            continue
        t = e["timing"][s]
        n = t["samples"]
        rows.append(
            (lab, t["evidence_seconds"] / n, (t["audit_seconds"] + t["verify_seconds"]) / n)
        )
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    y = list(range(len(rows)))[::-1]
    for yi, (lab, ev, au) in zip(y, rows, strict=True):
        ax.barh(yi + 0.18, ev, height=0.34, color=MID)
        ax.barh(yi - 0.18, au, height=0.34, color=ACCENT)
        ax.text(max(ev, au) * 1.15, yi, f"{100 * au / (ev + au):.1f}%", va="center", fontsize=6)
    ax.set_xscale("log")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("seconds per sample (log scale; one CPU thread)")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=MID, label="evidence generation (model + records)"),
            Patch(color=ACCENT, label="audit + verify_report (model-free)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.45, -0.32),
        ncol=1,
        frameon=False,
    )
    ax.set_xlim(right=max(max(r[1], r[2]) for r in rows) * 6)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_cost.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    agg = load(R / "results/aggregate.json")
    fig_decomposition(agg)
    fig_reference(
        load(R / "stage1/results/recalculation.json"),
        load(R / "stage3/results/interpbench_specification_labels.json"),
    )
    fig_seeds(agg)
    fig_cost(agg)
    print(sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
