"""Phase 5.5 analysis of the pre-registered hypotheses H1-H9 (docs/PHASE_5_5_PLAN.md
§2, §12, §13; operational details written in §18 before this script was run on the
full results). Reads results/faithfulness_*.json.gz and results/diagnostics_*.json.gz;
writes results/analysis.json and results/analysis.md. No "significant" wording.

Usage: python analysis.py
"""

# ruff: noqa: E501, RUF001  (report-table strings)
from __future__ import annotations

import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations, pairwise
from typing import Any

import common

from beyondnn.faithfulness import stats

ATTR = ("gradient", "input_x_gradient", "integrated_gradients")
METHODS = (*ATTR, "ablation", "random")
SHORT = {
    "gradient": "grad",
    "input_x_gradient": "IxG",
    "integrated_gradients": "IG",
    "ablation": "ablation",
    "random": "random",
}
GRID = (0.1, 0.25, 0.5, 0.75, 0.9)


def load(name: str) -> dict[str, Any] | None:
    path = common.RESULTS / f"{name}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def rows_of(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    fields = result["row_fields"]
    out: dict[str, list[dict[str, Any]]] = {}
    for site, block in result["settings"].items():
        rows = []
        for s in block["samples"]:
            for raw in s["rows"]:
                row = dict(zip(fields, raw, strict=False))
                row["sample"] = s["index"]
                row["margin"] = s["margin"]
                row["p"] = tuple(row["p"])
                rows.append(row)
        out[site] = rows
    return out


def med(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return statistics.median(vals) if vals else None


def iqr(values: list[float]) -> list[float] | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return [stats.quantile(vals, 0.25), stats.quantile(vals, 0.75)]


def select(rows: list[dict[str, Any]], **kw: Any) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        ok = True
        for key, want in kw.items():
            if key == "p":
                ok = ok and want in r["p"]
            elif key == "replacement_prefix":
                ok = ok and r["replacement"].startswith(want)
            else:
                ok = ok and r[key] == want
        if ok:
            out.append(r)
    return out


def outcome_at(row: dict[str, Any], t: float) -> str:
    if row["no_op"]:
        return "inconclusive"
    threshold = t * row["margin"]
    if row["test"] == "comprehensiveness":
        return "supports" if row["drop"] >= threshold else "contradicts"
    return "supports" if row["drop"] <= threshold else "contradicts"


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rho with average ranks for ties (Pearson on ranks)."""

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for m in range(i, j + 1):
                out[order[m]] = (i + j) / 2
            i = j + 1
        return out

    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if sx == 0 or sy == 0 else sxy / (sx * sy)


# ------------------------------------------------------------------ per-site analyses


def superiority_table(rows: list[dict[str, Any]], p: float, rep: str, match: str) -> dict[str, Any]:
    out = {}
    for m in METHODS:
        sel = select(
            rows, method=m, p=p, test="comprehensiveness", controls=match, replacement_prefix=rep
        )
        sups = [r["superiority"] for r in sel]
        ps = [r["mc_p_value"] for r in sel]
        out[m] = {
            "n": len(sel),
            "median_superiority": med(sups),
            "iqr_superiority": iqr(sups),
            "median_mc_p": med(ps),
            "share_mc_p_le_0.05": (sum(1 for x in ps if x <= 0.05) / len(ps)) if ps else None,
            "median_drop": med([r["drop"] for r in sel]),
            "outcomes": dict(Counter(r["outcome"] for r in sel)),
        }
    return out


def h2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for rep in sorted({r["replacement"] for r in rows}):
        k1 = select(rows, k=1, test="comprehensiveness", controls="count", replacement=rep)
        medians = {m: med([r["drop"] for r in k1 if r["method"] == m]) for m in METHODS}
        order = ["ablation", "integrated_gradients", "input_x_gradient", "gradient"]
        holds = all(
            medians[a] is not None and medians[b] is not None and medians[a] >= medians[b]
            for a, b in pairwise(order)
        )
        out[rep] = {"median_drop_k1": medians, "predicted_order_holds": holds}
    return out


def h3(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[Any, ...], dict[str, str]] = defaultdict(dict)
    for r in select(rows, controls="count"):
        cells[(r["sample"], r["method"], r["k"], r["replacement"])][r["test"]] = r["outcome"]
    decided = disagree = excluded = 0
    kinds: Counter[str] = Counter()
    for c in cells.values():
        a, b = c.get("comprehensiveness"), c.get("sufficiency")
        if a is None or b is None or "inconclusive" in (a, b):
            excluded += 1
            continue
        decided += 1
        if a != b:
            disagree += 1
            kinds[f"comp={a}/suff={b}"] += 1
    return {
        "cells": len(cells),
        "decided_cells": decided,
        "excluded_inconclusive": excluded,
        "disagreeing": disagree,
        "share": disagree / decided if decided else None,
        "kinds": dict(kinds),
    }


def h4(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comp = select(rows, test="comprehensiveness", controls="count")
    cells: dict[tuple[Any, ...], dict[str, str]] = defaultdict(dict)
    for r in comp:
        cells[(r["sample"], r["method"], r["k"])][r["replacement"]] = r["outcome"]
    changed = sum(1 for c in cells.values() if len(set(c.values())) > 1)
    reps = sorted({r["replacement"] for r in comp})
    reversals = []
    for k in sorted({r["k"] for r in comp}):
        med_by = {
            rep: {
                m: med(
                    [
                        r["drop"]
                        for r in comp
                        if r["k"] == k and r["replacement"] == rep and r["method"] == m
                    ]
                )
                for m in METHODS
            }
            for rep in reps
        }
        for r1, r2 in combinations(reps, 2):
            for a, b in combinations(METHODS, 2):
                x1, y1, x2, y2 = med_by[r1][a], med_by[r1][b], med_by[r2][a], med_by[r2][b]
                if None in (x1, y1, x2, y2):
                    continue
                if (x1 - y1) * (x2 - y2) < 0:
                    reversals.append(
                        {
                            "k": k,
                            "pair": [a, b],
                            "replacements": [r1, r2],
                            "medians": [[x1, y1], [x2, y2]],
                        }
                    )
    return {
        "cells": len(cells),
        "changed": changed,
        "share_changed": changed / len(cells) if cells else None,
        "median_ordering_reversals": reversals,
    }


def h5(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = [r for r in select(rows, controls="count") if not r["no_op"]]
    changed = sum(1 for r in cells if len({outcome_at(r, t) for t in (0.25, 0.5, 0.75)}) > 1)
    grid = {}
    for test in ("comprehensiveness", "sufficiency"):
        sub = [r for r in cells if r["test"] == test]
        grid[test] = {
            str(t): sum(1 for r in sub if outcome_at(r, t) == "supports") / len(sub)
            if sub
            else None
            for t in GRID
        }
    return {
        "cells": len(cells),
        "changed_across_0.25_0.5_0.75": changed,
        "share": changed / len(cells) if cells else None,
        "share_supports_by_threshold": grid,
    }


def h6(
    result_site: dict[str, Any], rows: list[dict[str, Any]], methods: tuple[str, ...]
) -> dict[str, Any]:
    drop = {}
    for r in select(rows, test="comprehensiveness", controls="count", replacement_prefix="r1"):
        drop[(r["sample"], r["method"], r["k"])] = r["drop"]
    jac, delta = [], []
    for s in result_site["samples"]:
        r1 = next(key for key in s["orders"] if key.startswith("r1"))
        orders = dict(s["orders"]["all"]) | s["orders"][r1]
        for k in sorted({key[2] for key in drop if key[0] == s["index"]}):
            for a, b in combinations(methods, 2):
                if (s["index"], a, k) in drop and (s["index"], b, k) in drop:
                    jac.append(stats.jaccard(orders[a][:k], orders[b][:k]))
                    delta.append(abs(drop[(s["index"], a, k)] - drop[(s["index"], b, k)]))
    rho = spearman(jac, delta)
    return {
        "methods": list(methods),
        "cells": len(jac),
        "spearman_jaccard_vs_abs_delta_drop": rho,
        "predicted": "rho < 0 and |rho| < 0.5",
        "holds": rho is not None and rho < 0 and abs(rho) < 0.5,
        "median_jaccard": med(jac),
        "share_jaccard_1_with_abs_delta_gt_0": (
            sum(1 for j, d in zip(jac, delta, strict=True) if j == 1.0 and d > 0) / len(jac)
            if jac
            else None
        ),
    }


def h9(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for r in select(rows, controls="count"):
        cells[(r["method"], r["k"], r["replacement"], r["test"])].append(r["outcome"])
    mixed = {key: v for key, v in cells.items() if "supports" in v and "contradicts" in v}
    hidden = total = 0
    for v in mixed.values():
        majority = Counter(v).most_common(1)[0][0]
        hidden += sum(1 for o in v if o != majority)
        total += len(v)
    return {
        "cells": len(cells),
        "mixed_cells": len(mixed),
        "share_mixed": len(mixed) / len(cells) if cells else None,
        "per_sample_results_in_mixed_cells": total,
        "hidden_by_majority_aggregate": hidden,
        "share_hidden": hidden / total if total else None,
    }


def paired(rows: list[dict[str, Any]], a: str, b: str, p: float, rep: str) -> dict[str, Any]:
    da = {
        r["sample"]: r["drop"]
        for r in select(
            rows, method=a, p=p, test="comprehensiveness", controls="count", replacement_prefix=rep
        )
    }
    db = {
        r["sample"]: r["drop"]
        for r in select(
            rows, method=b, p=p, test="comprehensiveness", controls="count", replacement_prefix=rep
        )
    }
    diffs = [da[s] - db[s] for s in sorted(da) if s in db]
    if not diffs:
        return {}
    return {
        "n": len(diffs),
        "median_difference": med(diffs),
        "win_rate": sum(1 for d in diffs if d > 0) / len(diffs),
        "ties": sum(1 for d in diffs if d == 0),
        "sign_flip_p_one_sided": stats.sign_flip_p(diffs, 5000, 0),
    }


def counterexamples(
    result_site: dict[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Natural cases, found by fixed rules (labelled case studies; not selected by hand)."""
    out: list[dict[str, Any]] = []
    comp = select(rows, test="comprehensiveness", controls="count", replacement_prefix="r1")
    # 1. no-op selections: a top-ranked unit whose removal changes nothing
    noop = [r for r in comp if r["no_op"] and r["method"] in ATTR]
    if noop:
        r = noop[0]
        out.append(
            {
                "rule": "attribution top-k whose removal is a no-op (first by sample order)",
                "count": len(noop),
                "example": {k: r[k] for k in ("sample", "method", "k", "selected", "drop")},
            }
        )
    # 2. IG top-k CONTRADICTS although IG's completeness is tight: worst superiority
    ig = [r for r in comp if r["method"] == "integrated_gradients" and not r["no_op"]]
    if ig:
        r = min(ig, key=lambda r: (r["superiority"], r["sample"], r["k"]))
        out.append(
            {
                "rule": "IG selection with the lowest superiority vs count controls (r1)",
                "example": {
                    k: r[k]
                    for k in (
                        "sample",
                        "k",
                        "selected",
                        "drop",
                        "superiority",
                        "median_control_drop",
                        "margin",
                    )
                },
            }
        )
    # 3. count-matched SUPPORTS-like superiority but magnitude-matched much lower
    mag = {
        (r["sample"], r["method"], r["k"]): r
        for r in select(rows, test="comprehensiveness", controls="magnitude")
    }
    gaps = []
    for r in comp:
        m = mag.get((r["sample"], r["method"], r["k"]))
        if m is not None and r["method"] in ATTR:
            gaps.append((r["superiority"] - m["superiority"], r, m))
    if gaps:
        gap, r, m = max(gaps, key=lambda g: (g[0], -g[1]["sample"]))
        out.append(
            {
                "rule": "largest drop in superiority from count- to magnitude-matched controls",
                "gap": gap,
                "example": {
                    "sample": r["sample"],
                    "method": r["method"],
                    "k": r["k"],
                    "count_superiority": r["superiority"],
                    "magnitude_superiority": m["superiority"],
                    "drop": r["drop"],
                },
            }
        )
    # 4. methods agree on the top-k set (Jaccard 1) but the comprehensiveness claim fails
    for s in result_site["samples"]:
        orders = s["orders"]["all"]
        for k in sorted({r["k"] for r in comp if r["sample"] == s["index"]}):
            sets = {m: set(orders[m][:k]) for m in ATTR}
            if len({frozenset(v) for v in sets.values()}) == 1:
                rs = [
                    r
                    for r in comp
                    if r["sample"] == s["index"] and r["k"] == k and r["method"] in ATTR
                ]
                if rs and all(r["outcome"] == "contradicts" for r in rs):
                    out.append(
                        {
                            "rule": "all three attribution methods agree on the top-k set, and the claim fails (first found)",
                            "example": {
                                "sample": s["index"],
                                "k": k,
                                "selected": sorted(sets["gradient"]),
                                "drop": rs[0]["drop"],
                                "margin": s["margin"],
                            },
                        }
                    )
                    break
        else:
            continue
        break
    # 5. replacement flips the outcome for the same selection
    by = defaultdict(dict)
    for r in select(
        rows, test="comprehensiveness", controls="count", method="integrated_gradients"
    ):
        by[(r["sample"], r["k"])][r["replacement"]] = r
    flips = [(key, v) for key, v in by.items() if len({x["outcome"] for x in v.values()}) > 1]
    if flips:
        key, v = flips[0]
        out.append(
            {
                "rule": "same IG selection, outcome changes with the replacement (first by sample order)",
                "count": len(flips),
                "example": {
                    "sample": key[0],
                    "k": key[1],
                    "by_replacement": {rep: [x["outcome"], x["drop"]] for rep, x in v.items()},
                },
            }
        )
    return out


def pair_table(
    result_site: dict[str, Any], rows: list[dict[str, Any]], p: float = 0.10
) -> dict[str, Any]:
    """Plan §13 / request §19: per comparable method pair (r1, count controls, p)."""
    key = {}
    for r in select(rows, p=p, controls="count", replacement_prefix="r1"):
        key[(r["sample"], r["method"], r["test"])] = r
    out = {}
    methods = (*ATTR, "ablation")
    for a, b in combinations(methods, 2):
        rho, jac, dc, ds, dsup = [], [], [], [], []
        for s in result_site["samples"]:
            r1 = next(x for x in s["orders"] if x.startswith("r1"))
            orders = dict(s["orders"]["all"]) | s["orders"][r1]
            ca, cb = (
                key.get((s["index"], a, "comprehensiveness")),
                key.get((s["index"], b, "comprehensiveness")),
            )
            sa, sb = (
                key.get((s["index"], a, "sufficiency")),
                key.get((s["index"], b, "sufficiency")),
            )
            if not (ca and cb and sa and sb):
                continue
            k = ca["k"]
            rho.append(stats.spearman_of_orders(orders[a], orders[b]))
            jac.append(stats.jaccard(orders[a][:k], orders[b][:k]))
            dc.append(ca["drop"] - cb["drop"])
            ds.append(sa["drop"] - sb["drop"])
            dsup.append(ca["superiority"] - cb["superiority"])
        out[f"{a}|{b}"] = {
            "n": len(rho),
            "median_spearman_full_ranking": med(rho),
            "median_topk_jaccard": med(jac),
            "iqr_topk_jaccard": iqr(jac),
            "median_comprehensiveness_drop_difference": med(dc),
            "median_sufficiency_drop_difference": med(ds),
            "median_superiority_difference": med(dsup),
            "share_first_larger_comprehensiveness_drop": (sum(1 for d in dc if d > 0) / len(dc))
            if dc
            else None,
        }
    return out


def curve_disagreement(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Removal says a > b if a's removal AOPC is larger; retention says a > b if a's
    retention AOPC is smaller (keeping a's top units loses less). Ties are skipped."""
    methods = (*ATTR, "ablation")
    decided = disagree = 0
    for r in recs:
        c = r["curves"]
        for a, b in combinations(methods, 2):
            ra, rb = c[f"remove/{a}"]["aopc_mean_drop"], c[f"remove/{b}"]["aopc_mean_drop"]
            ta, tb = c[f"retain/{a}"]["aopc_mean_drop"], c[f"retain/{b}"]["aopc_mean_drop"]
            if ra == rb or ta == tb:
                continue
            decided += 1
            if (ra > rb) != (ta < tb):
                disagree += 1
    return {
        "decided_pairs": decided,
        "disagreeing": disagree,
        "share": disagree / decided if decided else None,
    }


def stability_vs_agreement(
    result_site: dict[str, Any], recs: list[dict[str, Any]]
) -> dict[str, Any]:
    by = {r["index"]: r for r in recs}
    cases = []
    for s in result_site["samples"]:
        st = by.get(s["index"], {}).get("stability")
        if not st:
            continue
        r1 = next(x for x in s["orders"] if x.startswith("r1"))
        k = max(1, math.ceil(0.10 * s["n_units"]))
        agree = stats.jaccard(
            s["orders"]["all"]["integrated_gradients"][:k], s["orders"][r1]["ablation"][:k]
        )
        cases.append((agree, st["topk_jaccard"], s["index"]))
    hit = [c for c in cases if c[0] >= 0.5 and c[1] <= 0.3]
    return {
        "n": len(cases),
        "rule": "IG-vs-ablation top-10% Jaccard >= 0.5 but stability top-k Jaccard <= 0.3",
        "count": len(hit),
        "examples": [
            {"sample": i, "ig_ablation_jaccard": a, "stability_jaccard": j} for a, j, i in hit[:5]
        ],
        "spearman_agreement_vs_stability": spearman([c[0] for c in cases], [c[1] for c in cases]),
    }


def failure_correlates(result_site: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive only: no cause is inferred."""
    margin = {s["index"]: s["margin"] for s in result_site["samples"]}
    compl = {
        s["index"]: abs(s["completeness_delta"]["integrated_gradients"])
        for s in result_site["samples"]
    }
    ig = select(
        rows,
        method="integrated_gradients",
        p=0.10,
        test="comprehensiveness",
        controls="count",
        replacement_prefix="r1",
    )
    sup = [r for r in ig if r["outcome"] == "supports"]
    con = [r for r in ig if r["outcome"] == "contradicts"]
    other = defaultdict(list)
    for r in select(
        rows, method="integrated_gradients", p=0.10, test="comprehensiveness", controls="count"
    ):
        other[r["sample"]].append(r)
    rescued = sum(
        1
        for r in con
        if any(
            x["outcome"] == "supports"
            for x in other[r["sample"]]
            if not x["replacement"].startswith("r1")
        )
    )
    return {
        "ig_k10_r1_supports": len(sup),
        "ig_k10_r1_contradicts": len(con),
        "median_margin_supports": med([margin[r["sample"]] for r in sup]),
        "median_margin_contradicts": med([margin[r["sample"]] for r in con]),
        "spearman_margin_vs_superiority": spearman(
            [margin[r["sample"]] for r in ig], [r["superiority"] for r in ig]
        ),
        "spearman_margin_vs_drop_fraction": spearman(
            [margin[r["sample"]] for r in ig], [r["drop"] / margin[r["sample"]] for r in ig]
        ),
        "median_abs_ig_completeness_delta_supports": med([compl[r["sample"]] for r in sup]),
        "median_abs_ig_completeness_delta_contradicts": med([compl[r["sample"]] for r in con]),
        "r1_contradicts_that_support_under_another_replacement": rescued,
    }


def diagnostics_summary(diag: dict[str, Any] | None) -> dict[str, Any]:
    if diag is None:
        return {}
    out: dict[str, Any] = {}
    for site, recs in diag["settings"].items():
        curves: dict[str, Any] = {}
        for key in recs[0]["curves"]:
            vals = [r["curves"][key]["aopc_mean_drop"] for r in recs]
            worse = [r["curves"][key].get("aopc_fraction_controls_worse") for r in recs]
            curves[key] = {
                "median_aopc_mean_drop": med(vals),
                "median_of_per_sample_median_control_aopc": med(
                    [
                        med(list(r["curves"][key]["control_aopc_mean_drop"]))
                        for r in recs
                        if r["curves"][key].get("control_aopc_mean_drop")
                    ]
                ),
                "median_fraction_controls_worse": med([w for w in worse if w is not None]),
            }
        agree = defaultdict(list)
        errors = Counter()
        for r in recs:
            for pair, v in r["method_agreement"].items():
                if "error" in v:
                    errors[v["error"].split(":")[0]] += 1
                else:
                    agree[pair].append(v["topk_jaccard"])
        igs = [r["ig_step_sensitivity"] for r in recs]
        out[site] = {
            "curves_r1": curves,
            "method_agreement_median_topk_jaccard_(sum_reduction)": {
                p: med(v) for p, v in agree.items()
            },
            "diagnostic_errors": dict(errors)
            | Counter(v["error"].split(":")[0] for v in igs if "error" in v),
            "ig_32_vs_64": {
                "median_max_abs_difference": med(
                    [v["max_abs_difference"] for v in igs if "error" not in v]
                ),
                "median_topk_jaccard": med([v["topk_jaccard"] for v in igs if "error" not in v]),
            },
            "composition_verified": sum(r["composition_verified"] for r in recs),
        }
        if "stability" in recs[0]:
            st = [r["stability"] for r in recs]
            same = [o[0] == o[1] for o in (s["claim_outcomes"] for s in st) if o]
            out[site]["stability_roll_1px"] = {
                "n": len(st),
                "median_prediction_change": med([s["prediction_change"] for s in st]),
                "median_rank_correlation": med([s["rank_correlation"] for s in st]),
                "median_topk_jaccard": med([s["topk_jaccard"] for s in st]),
                "share_same_claim_outcome": sum(same) / len(same) if same else None,
                "share_prediction_change_gt_half_margin": sum(
                    1
                    for s, r in zip(st, recs, strict=True)
                    if s["prediction_change"] > 0.5 * r["margin"]
                )
                / len(st),
            }
    return out


def main() -> None:
    analysis: dict[str, Any] = {"models": {}}
    for kind in "ABC":
        result = load(f"faithfulness_{kind}")
        if result is None:
            continue
        per_site: dict[str, Any] = {}
        all_rows = rows_of(result)
        diag = load(f"diagnostics_{kind}")
        for site, rows in all_rows.items():
            block = result["settings"][site]
            per_site[site] = {
                "samples": len(block["samples"]),
                "n_units": block["samples"][0]["n_units"],
                "k_values": sorted({r["k"] for r in rows}),
                "composition_verified": sum(s["composition_verified"] for s in block["samples"]),
                "composition_failures": sum(s["composition_failures"] for s in block["samples"]),
                "superiority_k10_r1_count": superiority_table(rows, 0.10, "r1", "count"),
                "superiority_k10_r1_magnitude": superiority_table(rows, 0.10, "r1", "magnitude"),
                "superiority_all_k_r1_count": {
                    str(p): superiority_table(rows, p, "r1", "count")
                    for p in (0.01, 0.05, 0.10, 0.20)
                },
                "superiority_all_k_r1_magnitude": {
                    str(p): superiority_table(rows, p, "r1", "magnitude")
                    for p in (0.01, 0.05, 0.10, 0.20)
                },
                "superiority_k10_by_replacement": {
                    rep: superiority_table(rows, 0.10, rep, "count")
                    for rep in sorted({r["replacement"] for r in rows})
                },
                "H2": h2(rows),
                "H3": h3(rows),
                "H4": h4(rows),
                "H5": h5(rows),
                "H6_attribution_methods": h6(block, rows, ATTR),
                "H6_with_ablation": h6(block, rows, (*ATTR, "ablation")),
                "H9": h9(rows),
                "paired_k10_r1": {
                    f"{a}-{b}": paired(rows, a, b, 0.10, "r1")
                    for a, b in (
                        ("integrated_gradients", "gradient"),
                        ("integrated_gradients", "input_x_gradient"),
                        ("ablation", "integrated_gradients"),
                        ("integrated_gradients", "random"),
                        ("gradient", "random"),
                    )
                },
                "counterexamples": counterexamples(block, rows),
                "method_pairs_k10_r1": pair_table(block, rows),
                "failure_correlates": failure_correlates(block, rows),
                "curve_removal_vs_retention": curve_disagreement(diag["settings"][site])
                if diag
                else None,
                "stability_vs_agreement": stability_vs_agreement(block, diag["settings"][site])
                if diag
                else None,
                "median_seconds_per_sample": med([s["seconds"] for s in block["samples"]]),
                "no_op_share": sum(1 for r in rows if r["no_op"]) / len(rows),
                "completeness_delta_ig_median_abs": med(
                    [abs(s["completeness_delta"]["integrated_gradients"]) for s in block["samples"]]
                ),
                "median_margin": med([s["margin"] for s in block["samples"]]),
            }
        analysis["models"][kind] = {
            "sites": per_site,
            "runtime_seconds": result["runtime_seconds"],
            "environment": result["environment"],
            "diagnostics": diagnostics_summary(diag),
        }
    analysis["hypotheses"] = verdicts(analysis)
    (common.RESULTS / "analysis.json").write_text(
        json.dumps(analysis, indent=1, sort_keys=True, default=str)
    )
    (common.RESULTS / "analysis.md").write_text(markdown(analysis))
    print(markdown(analysis))


def verdicts(a: dict[str, Any]) -> dict[str, Any]:
    v: dict[str, Any] = {}
    h1 = {}
    for kind, m in a["models"].items():
        bar = 0.6 if kind == "C" else 0.8
        for site, s in m["sites"].items():
            t = s["superiority_k10_r1_count"]
            h1[f"{kind}/{site}"] = {
                "bar": bar,
                "IG": t["integrated_gradients"]["median_superiority"],
                "ablation": t["ablation"]["median_superiority"],
                "holds": all(
                    t[x]["median_superiority"] is not None and t[x]["median_superiority"] >= bar
                    for x in ("integrated_gradients", "ablation")
                ),
            }
    v["H1"] = h1
    h1b = {}
    for kind, m in a["models"].items():
        for site, s in m["sites"].items():
            c, g = s["superiority_k10_r1_count"], s["superiority_k10_r1_magnitude"]

            def drop(x: str, c: dict[str, Any] = c, g: dict[str, Any] = g) -> float | None:
                cx, gx = c[x]["median_superiority"], g[x]["median_superiority"]
                return None if cx is None or gx is None else cx - gx

            dx, di = drop("input_x_gradient"), drop("integrated_gradients")
            h1b[f"{kind}/{site}"] = {
                "IxG_decrease": dx,
                "IG_decrease": di,
                "holds": dx is not None and di is not None and dx > di,
            }
    v["H1b"] = h1b
    v["H2"] = {
        f"{k}/{s}": {rep: x["predicted_order_holds"] for rep, x in site["H2"].items()}
        for k, m in a["models"].items()
        for s, site in m["sites"].items()
    }
    v["H3"] = {
        f"{k}/{s}": site["H3"]["share"]
        for k, m in a["models"].items()
        for s, site in m["sites"].items()
    }
    pooled = [site["H3"] for m in a["models"].values() for site in m["sites"].values()]
    dec = sum(x["decided_cells"] for x in pooled)
    v["H3_pooled_share"] = sum(x["disagreeing"] for x in pooled) / dec if dec else None
    v["H4"] = {
        f"{k}/{s}": {
            "share_changed": site["H4"]["share_changed"],
            "reversals": len(site["H4"]["median_ordering_reversals"]),
        }
        for k, m in a["models"].items()
        for s, site in m["sites"].items()
    }
    v["H5"] = {
        f"{k}/{s}": site["H5"]["share"]
        for k, m in a["models"].items()
        for s, site in m["sites"].items()
    }
    v["H6"] = {
        f"{k}/{s}": [
            site["H6_attribution_methods"]["spearman_jaccard_vs_abs_delta_drop"],
            site["H6_attribution_methods"]["holds"],
        ]
        for k, m in a["models"].items()
        for s, site in m["sites"].items()
    }
    v["H7"] = {
        f"{k}/{s}": site["superiority_k10_r1_count"]["random"]["median_superiority"]
        for k, m in a["models"].items()
        for s, site in m["sites"].items()
    }
    v["H9"] = {
        f"{k}/{s}": [
            site["H9"]["mixed_cells"],
            site["H9"]["share_hidden"],
            site["H3"]["disagreeing"],
        ]
        for k, m in a["models"].items()
        for s, site in m["sites"].items()
    }
    return v


def fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def markdown(a: dict[str, Any]) -> str:
    lines = ["# Phase 5.5 analysis tables (generated by analysis.py)", ""]
    lines += ["## Median superiority vs count-matched controls, comprehensiveness, r1 (IQR)", ""]
    for p in ("0.01", "0.05", "0.1", "0.2"):
        lines += [
            f"### p = {p}",
            "",
            "| model/site | k | " + " | ".join(SHORT[m] for m in METHODS) + " |",
            "|---|---|" + "---|" * len(METHODS),
        ]
        for kind, m in a["models"].items():
            for site, s in m["sites"].items():
                t = s["superiority_all_k_r1_count"][p]
                ks = max(1, math.ceil(float(p) * s["n_units"]))
                cells = []
                for meth in METHODS:
                    q = t[meth]
                    cells.append(
                        f"{fmt(q['median_superiority'])} ({fmt((q['iqr_superiority'] or [None, None])[0])}–{fmt((q['iqr_superiority'] or [None, None])[1])})"
                    )
                lines.append(f"| {kind}/{site} | {ks} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## Count- vs magnitude-matched controls (comprehensiveness, r1, p = 10%): median superiority",
        "",
        "| model/site | " + " | ".join(f"{SHORT[m]} count→mag" for m in METHODS) + " |",
        "|---|" + "---|" * len(METHODS),
    ]
    for kind, m in a["models"].items():
        for site, s in m["sites"].items():
            c, g = s["superiority_k10_r1_count"], s["superiority_k10_r1_magnitude"]
            lines.append(
                f"| {kind}/{site} | "
                + " | ".join(
                    f"{fmt(c[x]['median_superiority'])}→{fmt(g[x]['median_superiority'])}"
                    for x in METHODS
                )
                + " |"
            )
    lines += ["", "## Median superiority by replacement (comprehensiveness, count, p = 10%)", ""]
    for kind, m in a["models"].items():
        for site, s in m["sites"].items():
            lines.append(f"**{kind}/{site}**")
            lines.append("")
            lines += [
                "| replacement | "
                + " | ".join(SHORT[x] for x in METHODS)
                + " | outcome SUPPORTS share (IG) |",
                "|---|" + "---|" * (len(METHODS) + 1),
            ]
            for rep, t in s["superiority_k10_by_replacement"].items():
                o = t["integrated_gradients"]["outcomes"]
                share = o.get("supports", 0) / max(1, sum(o.values()))
                lines.append(
                    f"| {rep} | "
                    + " | ".join(fmt(t[x]["median_superiority"]) for x in METHODS)
                    + f" | {fmt(share)} |"
                )
            lines.append("")
    lines += [
        "## Hypothesis summary",
        "",
        "```",
        json.dumps(a["hypotheses"], indent=1, default=str),
        "```",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
