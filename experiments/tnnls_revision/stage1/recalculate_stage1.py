"""TNNLS revision, Stage 1: artifact-level recalculation of headline numbers.

Pure standard-library re-derivation from the frozen result JSON committed under
``experiments/phase7*/results``. It deliberately does NOT import the repository's own
aggregation helpers (``evaluate775.py``, ``analysis775.py``, ``analysis75.py``) so that each
count is recomputed from the per-row / per-sample records by independent code.

This is RECALCULATION FROM FROZEN ARTIFACTS. It does not rerun any model and is not an
independent model-experiment reproduction.

Usage (from the repo root):  python3 experiments/tnnls_revision/stage1/recalculate_stage1.py
Writes: experiments/tnnls_revision/stage1/results/recalculation.json
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
P7 = ROOT / "experiments/phase7/results"
P75 = ROOT / "experiments/phase7_5/results"
P775 = ROOT / "experiments/phase7_75/results"
OUT = Path(__file__).resolve().parent / "results"
PRIMARY_TD = "tensor/resample|none|recorded"
ZERO_TD = "zero|none|recorded"


def load(p: Path) -> Any:
    return json.loads(p.read_text())


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def frac(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "wilson95": wilson(k, n)}


SOURCES: set[Path] = set()


def src(p: Path) -> Path:
    SOURCES.add(p)
    return p


# --------------------------------------------------------------------------- central 7.75


SITES = (
    ("A", "input"),
    ("A", "net.1"),
    ("B", "pixels"),
    ("B", "relu2"),
    ("C", "tokens"),
    ("D", "tokens"),
)


def central_775() -> dict[str, Any]:
    out: dict[str, Any] = {"per_site": {}, "pooled": {}}
    pooled_sup = pooled_rev = 0
    rev_axes: Counter[str] = Counter()
    axis_sets: Counter[str] = Counter()
    every_axis_involves_k = 0
    some_axis_without_k = 0
    no_axis_recorded = 0
    for elig in ("all", "content_tokens"):
        for model, site in SITES:
            p = P775 / f"central775_{model}_heldout_{elig}.json"
            if not p.exists():
                continue
            doc = load(src(p))
            s = doc["settings"][site]
            ig = s["audit"]["ig_necessary"]["per_sample"]
            r = s["audit"]["r_necessary"]["per_sample"]
            sup = [x for x in ig if x["standing"] == "supported"]
            rev = [x for x in sup if "alternative_reverses" in x["codes"]]
            key = f"{model}/{site}/{elig}"
            out["per_site"][key] = {
                "n_samples": len(ig),
                "ig_necessary_primary_supported": frac(len(sup), len(ig)),
                "of_which_alternative_reverses": f"{len(rev)}/{len(sup)}",
                "r_necessary_primary_supported": sum(x["standing"] == "supported" for x in r),
                "verify_report": s.get("verify_report"),
                "results_excluded": s.get("results_excluded"),
                "standing_distribution": dict(Counter(x["standing"] for x in ig)),
            }
            if elig == "all":
                pooled_sup += len(sup)
                pooled_rev += len(rev)
                for x in rev:
                    # Each entry is the axis label of one reversal finding: a comma-joined list
                    # of single explaining axes ("k,replacement") or "combined:a+b" when no
                    # single axis explains it. NOTE: the per-sample field merges ALTERNATIVE and
                    # STRESS_TEST reversal findings (central75.py), so it cannot attribute the
                    # alternative reversal alone.
                    axes = x.get("reversal_axes", [])
                    axis_sets[" | ".join(axes) or "(none)"] += 1
                    for a in axes:
                        rev_axes[a] += 1
                    explanations = [
                        set(a.removeprefix("combined:").split("+"))
                        if a.startswith("combined:")
                        else set(a.split(","))
                        for a in axes
                    ]
                    if not explanations:
                        no_axis_recorded += 1
                    elif all(
                        "k" in e if a.startswith("combined:") else e == {"k"}
                        for a, e in zip(axes, explanations, strict=True)
                    ):
                        every_axis_involves_k += 1
                    else:
                        some_axis_without_k += 1
    out["pooled"]["headline_all_tokens"] = {
        "alternative_reverses": pooled_rev,
        "primary_supported": pooled_sup,
        "text": f"{pooled_rev}/{pooled_sup}",
    }
    out["pooled"]["reversal_axis_labels_on_reversed_samples"] = dict(rev_axes)
    out["pooled"]["reversal_axis_label_sets_per_reversed_sample"] = dict(axis_sets)
    out["pooled"]["reversed_samples_only_k_explanations"] = every_axis_involves_k
    out["pooled"]["reversed_samples_with_some_explanation_without_k"] = some_axis_without_k
    out["pooled"]["reversed_samples_no_axis_recorded"] = no_axis_recorded
    out["pooled"]["note"] = (
        "Axis labels are recorded explanations of reversal findings, merged over ALTERNATIVE "
        "and STRESS_TEST roles; they are descriptive and are not experimentally isolated "
        "effects (Stage 2)."
    )
    return out


def alternative_multiplicity() -> dict[str, Any]:
    """Descriptive: how many ALTERNATIVE configurations each PRIMARY-supported IG sample had,
    and how many of them contradicted it (per-sample ``outcomes`` = (role, outcome, count))."""
    alts: Counter[int] = Counter()
    contra: list[int] = []
    for model, site in SITES:
        doc = load(src(P775 / f"central775_{model}_heldout_all.json"))
        for x in doc["settings"][site]["audit"]["ig_necessary"]["per_sample"]:
            if x["standing"] != "supported":
                continue
            o = {(r, v): n for r, v, n in x["outcomes"]}
            tot = sum(n for (r, _), n in o.items() if r == "alternative")
            alts[tot] += 1
            if "alternative_reverses" in x["codes"]:
                contra.append(o.get(("alternative", "contradicts"), 0))
    return {
        "alternative_configurations_per_supported_sample": dict(alts),
        "reversed_samples": len(contra),
        "contradicting_alternatives_median": statistics.median(contra),
        "contradicting_alternatives_min_max": [min(contra), max(contra)],
        "reversed_by_exactly_one_alternative": sum(c == 1 for c in contra),
    }


def manuscript_checks() -> dict[str, Any]:
    """Numbers stated in the TNNLS manuscript (rows M01-M17 of the verification table)."""
    out: dict[str, Any] = {}
    docs = {
        e: load(src(P775 / f"central775_D_heldout_{e}.json"))["settings"]["tokens"]
        for e in ("all", "content_tokens")
    }
    ex = {}
    for e, s in docs.items():
        m = next(x for x in s["samples"] if x["index"] == 423)
        a = next(x for x in s["audit"]["ig_necessary"]["per_sample"] if x["index"] == 423)
        ex[e] = {
            "text": m["text"],
            "n_units": m["n_units"],
            "margin": m["margin"],
            "primary": m["primary_single_configuration"],
            "standing": a["standing"],
            "outcomes": a["outcomes"],
            "tested": a["tested"],
        }
    out["M01_sample_423"] = ex
    st = {
        e: {x["index"]: x["standing"] for x in s["audit"]["ig_necessary"]["per_sample"]}
        for e, s in docs.items()
    }
    sup1 = [i for i, v in st["all"].items() if v == "supported"]
    out["M07_D1_supports_in_D2"] = dict(Counter(st["content_tokens"][i] for i in sup1))
    out["M07_D2_newly_supported"] = sum(
        v == "supported" and st["all"][i] != "supported" for i, v in st["content_tokens"].items()
    )
    shares: list[float] = []
    more = none = n_samples = 0
    for model, site in SITES:
        s = load(src(P775 / f"central775_{model}_heldout_all.json"))["settings"][site]
        n_samples += len(s["audit"]["ig_necessary"]["per_sample"])
        for x in s["audit"]["ig_necessary"]["per_sample"]:
            if x["standing"] != "supported":
                continue
            o = {(r, v): k for r, v, k in x["outcomes"]}
            tot = sum(k for (r, _), k in o.items() if r == "alternative")
            c = o.get(("alternative", "contradicts"), 0)
            shares.append(c / tot)
            more += c > o.get(("alternative", "supports"), 0)
            none += c == 0
        if (model, site) == ("B", "pixels"):
            g = s["audit"]["g_necessary"]["per_sample"]
            out["M06_B_pixels_gradient_supported"] = (
                f"{sum(x['standing'] == 'supported' for x in g)}/{len(g)}"
            )
    out["M03_held_out_samples"] = n_samples
    out["M05_median_share_contradicting"] = round(statistics.median(shares), 4)
    out["M05_more_contradict_than_support"] = more
    out["M05_no_contradicting_alternative"] = none
    n_results = 0
    for model, site in SITES:
        for e in ("all", "content_tokens"):
            p = P775 / f"central775_{model}_heldout_{e}.json"
            if p.exists():
                n_results += load(src(p))["settings"][site]["n_results"]
    out["M15_audited_results_8_central_reports"] = n_results
    td_rows = [
        x for r in load(src(P775 / "td_heldout.json"))["programs"].values() for x in r["rows"]
    ]
    sel = [x for x in td_rows if x["kind"] == "selection" and x["truth"] == "known_true"]
    out["M08_ig_selected_used_supported"] = (
        f"{sum(x['standing'] == 'supported' for x in sel)}/{len(sel)}"
    )
    comps = {
        (x["program"], x["component"], x["truth"]) for x in td_rows if x["kind"] != "selection"
    }
    out["M09_components_by_truth"] = dict(Counter(t for _, _, t in comps))
    cfg_rates = {}
    for cfg in ("tensor/mean|none|recorded", "zero|none|recorded", PRIMARY_TD):
        for t in ("known_true", "known_false"):
            xs = [
                x
                for x in td_rows
                if x["kind"] != "selection" and x["truth"] == t and cfg in x["configs"]
            ]
            cfg_rates[f"{cfg}|{t}"] = (
                f"{sum(x['configs'][cfg] == 'supports' for x in xs)}/{len(xs)}"
            )
    out["M10_td_configuration_rates"] = cfg_rates
    grid = load(src(P775 / "concept_calibration.json"))["grid"]
    out["M14_rel_validated_weights"] = sorted(
        {x["w"] for x in grid if x["used"]["rel"]["validation"] == "validated_concept"}
    )
    out["M14_nonused_cells_validated_any_rule"] = sum(
        x[k][r]["validation"] == "validated_concept"
        for x in grid
        for k in ("decodable_unused", "negative")
        for r in ("abs", "rel")
    )
    out["M14_grid_cells"] = len(grid)
    out["M17_nli_strata"] = load(src(P775 / "nli_strata.json"))["necessity_by_stratum"]["ig"]
    return out


def special_tokens_D() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for elig in ("all", "content_tokens"):
        doc = load(src(P775 / f"central775_D_heldout_{elig}.json"))
        s = doc["settings"]["tokens"]
        st = {x["index"]: x["standing"] for x in s["audit"]["ig_necessary"]["per_sample"]}
        sel_special = sup = sup_special = 0
        sup_idx = []
        for m in s["samples"]:
            u = m["primary_single_configuration"]["units"]
            has = 0 in u or (m["n_units"] - 1) in u
            sel_special += has
            if st[m["index"]] == "supported":
                sup += 1
                sup_special += has
                sup_idx.append(m["index"])
        out[elig] = {
            "n": len(s["samples"]),
            "primary_selections_with_CLS_or_SEP": sel_special,
            "ig_supported": sup,
            "ig_supported_with_CLS_or_SEP": sup_special,
            "supported_indices": sorted(sup_idx),
        }
    surv = set(out["all"]["supported_indices"]) & set(out["content_tokens"]["supported_indices"])
    out["all_token_supports_also_supported_content_tokens"] = (
        f"{len(surv)}/{out['all']['ig_supported']}"
    )
    return out


# --------------------------------------------------------------------------- TD benchmark


def td() -> dict[str, Any]:
    d = load(src(P775 / "td_heldout.json"))
    rows = [x for r in d["programs"].values() for x in r.get("rows", [])]
    nodes = [x for x in rows if x["kind"] != "selection"]
    sels = [x for x in rows if x["kind"] == "selection"]
    dec = [x for x in nodes if x["truth"] == "known_false"]
    used = [x for x in nodes if x["truth"] == "known_true"]
    empty = [x for x in nodes if x["truth"] == "empty"]
    corr = [x for x in dec if x["program"].startswith("ho_corr")]
    corr_by_comp = Counter((x["program"], x["component"]) for x in corr)
    descr = {p: r["description"] for p, r in d["programs"].items()}
    ig_dec = [x for x in sels if x["truth"] == "known_false"]
    tl = Counter((x["configs"].get(PRIMARY_TD), x["tl_interchange_flips"]) for x in nodes)
    comp_sup: dict[str, list[int]] = {}
    for x in used:
        c = comp_sup.setdefault(f"{x['program']}/{x['component']}", [0, 0])
        c[0] += x["standing"] == "supported"
        c[1] += 1
    ao: Counter[str] = Counter()
    for r in d["programs"].values():
        for c, dist in (r.get("attribution_only") or {}).items():
            if "ig_top" in c:
                ao.update(dist)
    return {
        "environment": d["environment"],
        "n_programs": len(d["programs"]),
        "program_errors": [p for p, r in d["programs"].items() if "error" in r],
        "compiled_matches_interpreter": {
            p: r.get("compiled_matches_interpreter") for p, r in d["programs"].items()
        },
        "node_rows": len(nodes),
        "selection_rows": len(sels),
        "decoys_primary_supported": frac(sum(x["standing"] == "supported" for x in dec), len(dec)),
        "decoy_standing_distribution": dict(Counter(x["standing"] for x in dec)),
        "used_primary_supported": frac(sum(x["standing"] == "supported" for x in used), len(used)),
        "used_standing_distribution": dict(Counter(x["standing"] for x in used)),
        "used_min_supported_per_component": min(v[0] for v in comp_sup.values()),
        "used_max_supported_per_component": max(v[0] for v in comp_sup.values()),
        "empty_primary_supported": frac(
            sum(x["standing"] == "supported" for x in empty), len(empty)
        ),
        "zero_ablation_supports_decoys": frac(
            sum(x["configs"].get(ZERO_TD) == "supports" for x in dec),
            sum(ZERO_TD in x["configs"] for x in dec),
        ),
        "correlated_program_decoys_primary_supported": frac(
            sum(x["standing"] == "supported" for x in corr), len(corr)
        ),
        "correlated_program_decoy_components": {
            f"{p}/{c}": n for (p, c), n in corr_by_comp.items()
        },
        "correlated_program_descriptions": {p: descr[p] for p in descr if p.startswith("ho_corr")},
        "ig_selected_decoys_primary_supported": frac(
            sum(x["standing"] == "supported" for x in ig_dec), len(ig_dec)
        ),
        "ig_selection_rows_by_truth": dict(Counter(x["truth"] for x in sels)),
        "attribution_only_ig_selection_claims": dict(ao),
        "primary_vs_transformerlens_interchange": {
            f"{a}|tl_flip={b}": v for (a, b), v in tl.items()
        },
        "programs": {p: descr[p] for p in descr},
    }


def td_concepts() -> dict[str, Any]:
    d = load(src(P775 / "td_concepts_heldout.json"))
    pos = neg = pos_val = neg_val = 0
    by_known: Counter[tuple[str, str]] = Counter()
    for r in d["programs"].values():
        for c in r["concepts"].values():
            v = c["rules"]["rel"]["validation"]["covariance"]
            by_known[(c["known"], v)] += 1
            if c["known"] == "positive":
                pos += 1
                pos_val += v == "validated_concept"
            else:
                neg += 1
                neg_val += any(
                    vv == "validated_concept"
                    for rule in c["rules"].values()
                    for vv in rule["validation"].values()
                )
    return {
        "known_positive_validated_rel_rule": f"{pos_val}/{pos}",
        "known_negative_validated_under_any_rule": f"{neg_val}/{neg}",
        "by_known_and_validation": {f"{a}|{b}": n for (a, b), n in by_known.items()},
        "n_programs": len(d["programs"]),
    }


# --------------------------------------------------------------------------- Phase 7.5


def interpbench() -> dict[str, Any]:
    def rows_of(cases: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for c, r in cases.items():
            if "error" in r or not r.get("benchmark_reliable", False):
                continue
            out += [dict(x, case=c) for x in r["rows"]]
        return out

    def conf(rows: list[dict[str, Any]]) -> dict[str, int]:
        c: Counter[str] = Counter()
        for x in rows:
            if x["kind"] not in ("head", "mlp"):
                continue
            g, s = x["gt"], x["standing"]
            if g == "ambiguous":
                c["ambiguous"] += 1
            elif g == "necessary":
                c["tp" if s == "supported" else "fn" if s == "contradicted" else "pos_other"] += 1
            else:
                c["tn" if s == "contradicted" else "fp" if s == "supported" else "neg_other"] += 1
        return dict(c)

    old = dict(load(src(P75 / "external_interpbench_heldout.json"))["cases"])
    old["124"] = load(src(P75 / "external_interpbench_heldout_case124_dv1.json"))["cases"]["124"]
    rows = rows_of(old)
    sel = [x for x in rows if x["kind"] == "selection"]
    sel_wrong = [x for x in sel if x["gt"] == "not_necessary"]
    ao: Counter[str] = Counter()
    for r in old.values():
        for c, dist in (r.get("attribution_only") or {}).items():
            if "ig_top" in c:
                ao.update(dist)
    new = load(src(P775 / "interpbench_heldout.json"))["cases"]
    return {
        "n_cases": len(old),
        "reliable_cases": sum(1 for r in old.values() if r.get("benchmark_reliable")),
        "phase7_5_primary_confusion_nodes": conf(rows),
        "phase7_75_rerun_primary_confusion_nodes": conf(rows_of(new)),
        "ig_selection_rows": len(sel),
        "ig_selection_rows_by_gt": dict(Counter(x["gt"] for x in sel)),
        "ig_selected_unnecessary_contradicted": frac(
            sum(x["standing"] == "contradicted" for x in sel_wrong), len(sel_wrong)
        ),
        "attribution_only_ig": dict(ao),
        "per_configuration_as_if_primary": per_config(rows),
    }


def per_config(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """SUPPORTS rate of each configuration on clear node instances (gt necessary / not)."""
    nodes = [x for x in rows if x["kind"] in ("head", "mlp")]
    out = {}
    for cfg in sorted({c for x in nodes for c in x["configs"]}):
        cell = {}
        for gt in ("necessary", "not_necessary"):
            xs = [x for x in nodes if x["gt"] == gt and cfg in x["configs"]]
            k = sum(x["configs"][cfg] == "supports" for x in xs)
            cell[gt] = f"{k}/{len(xs)}"
        out[cfg] = cell
    return out


def esnli(seed: int = 7501, draws: int = 10000) -> dict[str, Any]:
    d = load(src(P75 / "esnli.json"))
    s = d["samples"]
    ig = [x["plausibility"]["ig_token_f1"] for x in s]
    rnd = [x["plausibility"]["random_token_f1"] for x in s]
    diff = [a - b for a, b in zip(ig, rnd, strict=True)]
    rng = random.Random(seed)
    boots = []
    n = len(diff)
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(diff[i] for i in idx) / n)
    boots.sort()
    dist = {g: d["audits"][g][f"{g}_necessary"]["distribution"] for g in ("human", "ig", "random")}
    return {
        "n": n,
        "ig_token_f1_mean": round(statistics.fmean(ig), 4),
        "ig_minus_random_f1_mean": round(statistics.fmean(diff), 4),
        "own_bootstrap_95_percentile": [
            round(boots[int(0.025 * draws)], 4),
            round(boots[int(0.975 * draws) - 1], 4),
        ],
        "own_bootstrap_98_75_percentile": [
            round(boots[int(0.00625 * draws) - 1], 4),
            round(boots[int(0.99375 * draws) - 1], 4),
        ],
        "inter_annotator_iou_mean_first_entry": round(
            statistics.fmean(
                v[0] if isinstance(v, list) else v
                for v in (x["plausibility"]["inter_annotator_iou"] for x in s)
            ),
            4,
        ),
        "own_bootstrap_note": (
            "own RNG stream; matches the recorded interval only approximately "
            "(different resampling code)"
        ),
        "recorded_plausibility": d["plausibility"],
        "necessary_distribution": dist,
        "model": d["model"],
    }


def esnli_bonferroni_recorded() -> dict[str, Any]:
    h = load(src(P75 / "hypotheses75.json"))["NLP"]
    return {
        "N2_bonferroni": h["N2"]["paired_difference_bonferroni"]["text"],
        "N1_paired": h["N1"]["paired_difference_95"]["text"],
    }


def central_75() -> dict[str, Any]:
    out = {}
    tot_s = tot_r = 0
    for model, site in SITES:
        doc = load(src(P75 / f"central75_{model}_heldout.json"))
        ig = doc["settings"][site]["audit"]["ig_necessary"]["per_sample"]
        sup = [x for x in ig if x["standing"] == "supported"]
        rev = sum("alternative_reverses" in x["codes"] for x in sup)
        out[f"{model}/{site}"] = f"{rev}/{len(sup)} (n={len(ig)})"
        tot_s += len(sup)
        tot_r += rev
    out["pooled"] = f"{tot_r}/{tot_s}"
    return out


# --------------------------------------------------------------------------- Phase 7


def phase7_central() -> dict[str, Any]:
    naive = audited = n = 0
    per = {}
    for model in ("A", "B", "C"):
        doc = load(src(P7 / f"central_{model}.json"))
        for site, s in doc["settings"].items():
            dist = s["audit"]["claims"]["ig_necessary"]["distribution"]
            k = s["naive_single_configuration_supports"]
            ns = s["n_samples"]
            naive += k
            audited += dist.get("supported", 0)
            n += ns
            per[f"{model}/{site}"] = {
                "naive": k,
                "audited_supported": dist.get("supported", 0),
                "n": ns,
            }
    return {
        "per_site": per,
        "naive_total": f"{naive}/{n}",
        "audited_supported_total": f"{audited}/{n}",
    }


# --------------------------------------------------------------------------- mutations / verify


def mutations() -> dict[str, Any]:
    out = {}
    for p in (
        P7 / "mutations.json",
        P75 / "mutations75.json",
        P775 / "mutations775.json",
        P775 / "mutations_pre8.json",
    ):
        d = load(src(p))
        out[str(p.relative_to(ROOT))] = summarize_mut(d)
    return out


def summarize_mut(d: Any) -> Any:
    if isinstance(d, dict):
        keys = {
            k: d[k] for k in d if k in ("killed", "survived", "total", "n", "summary", "all_killed")
        }
        if keys:
            return keys
        muts = d.get("mutations") or d.get("results")
        if isinstance(muts, list):
            st = Counter(str(m.get("status", m.get("killed"))) for m in muts if isinstance(m, dict))
            return {"n": len(muts), "status": dict(st)}
        return {k: summarize_mut(v) for k, v in list(d.items())[:8]}
    if isinstance(d, list):
        st = Counter(str(m.get("status", m.get("killed"))) for m in d if isinstance(m, dict))
        return {"n": len(d), "status": dict(st)}
    return str(d)[:200]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {
        "kind": "RECALCULATED FROM FROZEN ARTIFACTS (no model rerun)",
        "central_775": central_775(),
        "alternative_multiplicity": alternative_multiplicity(),
        "manuscript_checks": manuscript_checks(),
        "special_tokens_D": special_tokens_D(),
        "central_75_heldout_for_comparison": central_75(),
        "td": td(),
        "td_concepts": td_concepts(),
        "interpbench": interpbench(),
        "esnli": esnli(),
        "esnli_recorded_intervals": esnli_bonferroni_recorded(),
        "phase7_central": phase7_central(),
        "mutations": mutations(),
    }
    res["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(SOURCES)}
    (OUT / "recalculation.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    print(
        json.dumps({k: v for k, v in res.items() if k != "source_sha256"}, indent=1, default=str)[
            :20000
        ]
    )


if __name__ == "__main__":
    main()
