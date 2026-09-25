"""Phase 5.5 diagnostics (docs/PHASE_5_5_PLAN.md §8): removal/retention curves under r1
for every declared sample and method, stability (model B only), and the library's
method diagnostics (method agreement, IG step sensitivity) on every sample.

Usage: python diagnostics.py A|B|C [--limit N]
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from typing import Any

import common
import run_faithfulness as rf
import torch

import beyondnn as bnn
from beyondnn.faithfulness import stats

A, F, iv = bnn.attribution, bnn.faithfulness, bnn.interventions
CURVE_CONTROLS = 20  # random rankings at the same points (logged in plan §18)


def _points(n: int, retain: bool) -> list[int]:
    pts = {0, *rf.ks_for(n)}
    if retain:
        pts.add(n)
    return sorted(pts)


def _ablation_curve(
    model: Any,
    setting: rf.Setting,
    sample: rf.Sample,
    metric: Any,
    order: list[int],
    pts: list[int],
    retain: bool,
) -> list[float]:
    """Workaround (API review): F.curve refuses declared rankings, so the ablation
    ranking's curve is built from public interventions in one comparison family."""
    n = len(order)
    specs = []
    kept = []
    for k in pts:
        units = tuple(sorted(order[:k]))
        if (not retain and k == 0) or (retain and k == n):
            continue
        axes = setting.unit_axes if units else None
        if setting.module is None:
            spec = iv.zero_input(units=units or None, retain=retain and bool(units), unit_axes=axes)
        else:
            spec = iv.zero(
                setting.module, units=units or None, retain=retain and bool(units), unit_axes=axes
            )
        specs.append(spec)
        kept.append(k)
    fam = iv.compare_family(model, [(sample.inputs, sample.kwargs, specs)], metric)
    drops = dict(zip(kept, [-e.effect for e in fam.effects[0]], strict=True))
    return [drops.get(k, 0.0) for k in pts]


def _roll() -> Any:
    unit_map = [h * 8 + (w - 1) % 8 for h in range(8) for w in range(8)]
    return F.transformation(
        "roll_right_1px_circular",
        lambda x: torch.roll(x, shifts=1, dims=3),
        implementation_revision="phase5_5-v1",
        config={"shifts": 1, "dims": 3, "circular": True},
        unit_map=unit_map,
    )


def run(
    model: Any, setting: rf.Setting, sample: rf.Sample, path: str, verify: bool
) -> dict[str, Any]:
    kw = sample.kwargs
    with torch.no_grad():
        out = model(*sample.inputs, **kw)
    metric, _pred, _runner, margin = common.margin_target(out.logits if path else out, path=path)
    methods = rf._methods(setting)
    attrs = {
        name: A.attribute(
            model, *sample.inputs, target=metric, method=m, at=setting.at, model_kwargs=kw
        )
        for name, m in methods.items()
    }
    ax = setting.unit_axes
    red = {name: rf.REDUCE[name] if ax else None for name in attrs}
    rankings = {name: F.ranking(a, unit_axes=ax, reduce=red[name]) for name, a in attrs.items()}
    n = next(iter(rankings.values())).n_units
    rec: dict[str, Any] = {"index": sample.index, "margin": margin, "n_units": n, "curves": {}}
    t0 = time.perf_counter()
    composed = 0
    for mode in ("remove", "retain"):
        pts = _points(n, mode == "retain")
        for name, ranking in rankings.items():
            c = F.curve(
                model,
                *sample.inputs,
                ranking=ranking,
                target=metric,
                mode=mode,
                points=pts,
                controls=F.controls(CURVE_CONTROLS, seed=10000 + sample.index),
                model_kwargs=kw,
                attributions=[attrs[name]],
            )
            m = c.protocol_result.measurements
            rec["curves"][f"{mode}/{name}"] = {
                "points": list(c.points),
                "drops": list(c.drops),
                "aopc_mean_drop": m["aopc_mean_drop"],
                "control_aopc_mean_drop": m["control_aopc_mean_drop"],
                "aopc_fraction_controls_worse": m["aopc_fraction_controls_worse"],
                "control_q05": m["control_q05"],
                "control_q95": m["control_q95"],
            }
            if verify:
                bnn.compose(
                    bnn.trace(model, *sample.inputs, model_kwargs=kw),
                    attributions=[attrs[name]],
                    faithfulness=[c],
                )
                composed += 1
        # ablation ranking under r1 = zero (rf: every r1 is zero)
        fam = iv.compare_family(
            model,
            [(sample.inputs, kw, [rf._single_unit(setting, u, F.zero()) for u in range(n)])],
            metric,
        )
        abl = [-e.effect for e in fam.effects[0]]
        order = list(stats.rank_order(abl, by="abs"))
        drops = _ablation_curve(model, setting, sample, metric, order, pts, mode == "retain")
        rec["curves"][f"{mode}/ablation"] = {
            "points": pts,
            "drops": drops,
            "aopc_mean_drop": sum(drops) / len(drops),
            "workaround": "compare_family (declared rankings refused by F.curve)",
        }
    rec["curve_seconds"] = time.perf_counter() - t0
    # library method diagnostics (all pairs) at k = 10%. A diagnostic takes ONE reduce for
    # both attributions (API review), so 'sum' is used here; RQ6 uses per-method
    # reductions from run_faithfulness.py.
    k10 = max(1, -(-n // 10))
    names = list(attrs)
    rec["method_agreement"] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            try:
                d = F.method_agreement(
                    model,
                    *sample.inputs[:1],
                    a=attrs[a],
                    b=attrs[b],
                    target=metric,
                    k=k10,
                    unit_axes=ax,
                    reduce="sum" if ax else None,
                    model_kwargs=kw,
                )
                rec["method_agreement"][f"{a}|{b}"] = {
                    "rank_correlation": d.measurements["rank_correlation"],
                    "topk_jaccard": d.measurements["topk_jaccard"],
                }
            except Exception as exc:  # recorded, not hidden
                rec["method_agreement"][f"{a}|{b}"] = {"error": f"{type(exc).__name__}: {exc}"}
    alt = (
        rf.AC.integrated_gradients(baseline=A.zero_baseline(), n_steps=64, rule="riemann_middle")
        if setting.captum_ig
        else A.integrated_gradients(baseline=A.zero_baseline(), n_steps=64, rule="riemann_middle")
    )
    try:
        a64 = A.attribute(
            model, *sample.inputs, target=metric, method=alt, at=setting.at, model_kwargs=kw
        )
        d = F.ig_step_sensitivity(
            model,
            *sample.inputs[:1],
            a=attrs["integrated_gradients"],
            b=a64,
            target=metric,
            k=k10,
            unit_axes=ax,
            reduce="sum" if ax else None,
            model_kwargs=kw,
        )
        rec["ig_step_sensitivity"] = {
            key: d.measurements[key]
            for key in (
                "max_abs_difference",
                "rank_correlation",
                "topk_jaccard",
                "completeness_deltas",
            )
        }
    except Exception as exc:  # recorded, not hidden
        rec["ig_step_sensitivity"] = {"error": f"{type(exc).__name__}: {exc}"}
    if setting.model_id == "B" and setting.site == "pixels":
        k = max(1, -(-n // 10))
        s = F.stability(
            model,
            sample.inputs[0],
            transformation=_roll(),
            method=methods["integrated_gradients"],
            target=metric,
            k=k,
            test=F.comprehensiveness(
                target=metric, min_drop=rf.PRIMARY_T * margin, statement="IG top-10% necessary"
            ),
            unit_axes=ax,
            reduce="sum",
        )
        rec["stability"] = {
            key: s.measurements[key]
            for key in (
                "prediction_x",
                "prediction_gx",
                "prediction_change",
                "rank_correlation",
                "topk_jaccard",
                "claim_outcomes",
            )
        } | {"aspects": {o.aspect: o.outcome.value for o in s.protocol_result.outcomes}}
        if verify:
            bnn.compose(bnn.trace(model, *sample.inputs), faithfulness=[s])
            composed += 1
    rec["composition_verified"] = composed
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["A", "B", "C"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    build = {"A": rf.settings_A, "B": rf.settings_B, "C": rf.settings_C}[args.model]
    model, samples, sets, _info, path = build()
    samples = samples[: args.limit] if args.limit else samples
    started = time.perf_counter()
    out: dict[str, Any] = {"environment": common.environment(), "settings": {}}
    for setting in sets:
        out["settings"][setting.site] = [
            run(model, setting, s, path, verify=i == 0) for i, s in enumerate(samples)
        ]
        print(f"{setting.model_id}/{setting.site} done", flush=True)
    out["runtime_seconds"] = time.perf_counter() - started
    suffix = f"_limit{args.limit}" if args.limit else ""
    target = common.RESULTS / f"diagnostics_{args.model}{suffix}.json.gz"
    with gzip.open(target, "wt") as fh:
        json.dump(out, fh, sort_keys=True, default=str)
    print(f"wrote {target} in {out['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
