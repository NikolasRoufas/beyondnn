"""Phase 5.5 central premise experiment (docs/PHASE_5_5_PLAN.md §5-§13).

For every declared held-out sample of one model, and each site:
- attributions (gradient, input x gradient, IG n=32 riemann_middle) of the margin;
- an ablation ranking (single-unit removal drops; INTERVENTIONAL) per replacement;
- a seeded random ranking (a control "method");
- comprehensiveness and sufficiency at every k and replacement with N=50
  count-matched controls, plus comprehensiveness with N=50 magnitude-matched
  controls under r1.
Raw drops are stored, so thresholds other than the primary one are derived later.

Usage: python run_faithfulness.py A|B|C [--limit N]
Only BeyondNN's public API is used (plan §15).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import common
import torch
from torch import nn

import beyondnn as bnn
from beyondnn.attribution import captum as AC
from beyondnn.faithfulness import stats

A, F, iv = bnn.attribution, bnn.faithfulness, bnn.interventions

PS = (0.01, 0.05, 0.10, 0.20)
N_CONTROLS = 50
PRIMARY_T = 0.5
IG_STEPS = 32
METHODS = ("gradient", "input_x_gradient", "integrated_gradients", "ablation", "random")
REDUCE = {"gradient": "l2", "input_x_gradient": "sum", "integrated_gradients": "sum"}
RANDOM_SEED_BASE = 30000  # plan §6 names a "seeded permutation"; the base is logged in §18


@dataclass
class Setting:
    model_id: str
    site: str
    at: Any  # attribution.input() or attribution.layer(...)
    module: str | None  # None for the input site
    unit_axes: tuple[int, ...] | None
    captum_ig: bool
    replacements: Callable[[int, tuple[Any, ...]], dict[str, Any]]


@dataclass
class Sample:
    index: int
    inputs: tuple[Any, ...]
    kwargs: dict[str, Any]
    label: int
    text: str | None = None


def _resample_row(pool: list[int], index: int) -> int:
    g = torch.Generator().manual_seed(20000 + index)
    return pool[int(torch.randint(len(pool), (1,), generator=g))]


# ------------------------------------------------------------------ settings


def settings_A() -> tuple[nn.Module, list[Sample], list[Setting], dict[str, Any], str]:
    model, data, info = common.train("A")
    chosen = common.select_samples(model, data.x, data.y, data.test, 60)
    samples = [Sample(i, (data.x[i : i + 1],), {}, int(data.y[i])) for i in chosen]
    xtr = data.x[data.train]
    with torch.no_grad():
        htr = model.net[:2](xtr)
    fmin = xtr.min(0).values.unsqueeze(0)

    def input_reps(index: int, _: tuple[Any, ...]) -> dict[str, Any]:
        j = _resample_row(data.train, index)
        return {
            "r1_zero(=train_mean)": F.zero(),
            "r2_resample_train_row": F.replacement(data.x[j : j + 1]),
            "r3_train_min": F.replacement(fmin),
        }

    def hidden_reps(index: int, _: tuple[Any, ...]) -> dict[str, Any]:
        j = data.train.index(_resample_row(data.train, index))
        return {
            "r1_zero": F.zero(),
            "r2_train_mean": F.replacement(htr.mean(0, keepdim=True)),
            "r3_resample_train_row": F.replacement(htr[j : j + 1]),
        }

    sets = [
        Setting("A", "input", A.input(), None, None, True, input_reps),
        Setting("A", "net.1", A.layer("net.1"), "net.1", None, False, hidden_reps),
    ]
    return model, samples, sets, info, ""


def settings_B() -> tuple[nn.Module, list[Sample], list[Setting], dict[str, Any], str]:
    model, data, info = common.train("B")
    chosen = common.select_samples(model, data.x, data.y, data.test, 60)
    samples = [Sample(i, (data.x[i : i + 1],), {}, int(data.y[i])) for i in chosen]
    xtr = data.x[data.train]
    with torch.no_grad():
        atr = model.relu2(model.conv2(model.relu1(model.conv1(xtr))))

    def pixel_reps(index: int, _: tuple[Any, ...]) -> dict[str, Any]:
        j = _resample_row(data.train, index)
        return {
            "r1_zero": F.zero(),
            "r2_train_mean_image": F.replacement(xtr.mean(0, keepdim=True)),
            "r3_resample_train_image": F.replacement(data.x[j : j + 1]),
        }

    def channel_reps(index: int, _: tuple[Any, ...]) -> dict[str, Any]:
        j = data.train.index(_resample_row(data.train, index))
        return {
            "r1_zero": F.zero(),
            "r2_train_mean_map": F.replacement(atr.mean(0, keepdim=True)),
            "r3_resample_train_map": F.replacement(atr[j : j + 1]),
        }

    sets = [
        Setting("B", "pixels", A.input(), None, (2, 3), True, pixel_reps),
        Setting("B", "relu2", A.layer("relu2"), "relu2", (1,), False, channel_reps),
    ]
    return model, samples, sets, info, ""


def c_samples(model: nn.Module, tok: Any, n: int) -> tuple[list[Sample], dict[str, Any]]:
    rows = common.sst2_validation()
    eligible = []
    for row in rows:
        enc = tok(row["sentence"], return_tensors="pt")
        t = enc["input_ids"].shape[1]
        if not 8 <= t <= 24:
            continue
        kwargs = {"attention_mask": enc["attention_mask"], "token_type_ids": enc["token_type_ids"]}
        with torch.no_grad():
            pred = int(model(enc["input_ids"], **kwargs).logits.argmax(-1))
        if pred == int(row["label"]):
            eligible.append(
                Sample(int(row["idx"]), (enc["input_ids"],), kwargs, pred, row["sentence"])
            )
    order = torch.randperm(
        len(eligible), generator=torch.Generator().manual_seed(common.SELECTION_SEED)
    )
    meta = {
        "validation_rows": len(rows),
        "eligible_correct_8_24_tokens": len(eligible),
        "file_sha256": rows[0]["_file_sha256"],
    }
    return [eligible[i] for i in order[:n].tolist()], meta


def settings_C() -> tuple[nn.Module, list[Sample], list[Setting], dict[str, Any], str]:
    model, tok, info = common.bert()
    samples, meta = c_samples(model, tok, 40)
    info["dataset"] = meta
    emb = model.bert.embeddings.word_embeddings.weight.detach()

    def token_reps(_: int, inputs: tuple[Any, ...]) -> dict[str, Any]:
        t = inputs[0].shape[1]
        return {
            "r1_zero": F.zero(),
            "r2_mask_embedding": F.replacement(emb[tok.mask_token_id].expand(1, t, -1).clone()),
            "r3_pad_embedding": F.replacement(emb[tok.pad_token_id].expand(1, t, -1).clone()),
        }

    site = "bert.embeddings.word_embeddings"
    sets = [Setting("C", "tokens", A.layer(site), site, (1,), False, token_reps)]
    return model, samples, sets, info, '["logits"]'


# ------------------------------------------------------------------ one sample


def _methods(setting: Setting) -> dict[str, Any]:
    ig = (
        AC.integrated_gradients(baseline=A.zero_baseline(), n_steps=IG_STEPS, rule="riemann_middle")
        if setting.captum_ig
        else A.integrated_gradients(
            baseline=A.zero_baseline(), n_steps=IG_STEPS, rule="riemann_middle"
        )
    )
    return {
        "gradient": A.gradient(),
        "input_x_gradient": A.input_x_gradient(),
        "integrated_gradients": ig,
    }


def _single_unit(setting: Setting, unit: int, rep: Any) -> Any:
    axes = setting.unit_axes
    if setting.module is None:
        if rep.tensor is None:
            return iv.zero_input(units=(unit,), unit_axes=axes)
        return iv.constant_input(rep.tensor, units=(unit,), unit_axes=axes)
    if rep.tensor is None:
        return iv.zero(setting.module, units=(unit,), unit_axes=axes)
    return iv.constant(setting.module, rep.tensor, units=(unit,), unit_axes=axes)


def ks_for(n: int) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for p in PS:
        out.setdefault(max(1, math.ceil(p * n)), []).append(p)
    return out


def run_sample(
    model: nn.Module, setting: Setting, sample: Sample, path: str, verify: bool
) -> dict[str, Any]:
    kw = sample.kwargs
    with torch.no_grad():
        out = model(*sample.inputs, **kw)
    logits = out.logits if path else out
    metric, pred, runner, margin = common.margin_target(logits, path=path)
    t0 = time.perf_counter()
    attrs = {
        name: A.attribute(
            model, *sample.inputs, target=metric, method=m, at=setting.at, model_kwargs=kw
        )
        for name, m in _methods(setting).items()
    }
    ax = setting.unit_axes
    rankings = {
        name: F.ranking(a, unit_axes=ax, reduce=REDUCE[name] if ax else None)
        for name, a in attrs.items()
    }
    n = next(iter(rankings.values())).n_units
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(RANDOM_SEED_BASE + sample.index)
    )
    orders: dict[str, dict[str, list[int]]] = {}
    scores: dict[str, dict[str, list[float]]] = {}
    for name, r in rankings.items():
        orders.setdefault("all", {})[name] = list(r.order)
        scores.setdefault("all", {})[name] = list(r.scores or ())
    orders["all"]["random"] = perm.tolist()
    completeness = {name: a.completeness_delta for name, a in attrs.items()}
    rows: list[list[Any]] = []
    verified = failures = 0
    reps = setting.replacements(sample.index, sample.inputs)
    for rname, rep in reps.items():
        fam = iv.compare_family(
            model,
            [(sample.inputs, kw, [_single_unit(setting, u, rep) for u in range(n)])],
            metric,
        )
        abl = [-e.effect for e in fam.effects[0]]
        orders[rname] = {"ablation": list(stats.rank_order(abl, by="abs"))}
        scores[rname] = {"ablation": abl}
        for k, ps in ks_for(n).items():
            for method in METHODS:
                attributions: list[Any] = []
                if method in attrs:
                    selection = F.top_k(
                        attrs[method], k=k, unit_axes=ax, reduce=REDUCE[method] if ax else None
                    )
                    attributions = [attrs[method]]
                else:
                    order = orders[rname]["ablation"] if method == "ablation" else perm.tolist()
                    selection = F.units(setting.at, tuple(order[:k]), n_units=n, unit_axes=ax)
                tests = [
                    ("comprehensiveness", "count"),
                    ("sufficiency", "count"),
                ]
                if rname.startswith("r1"):
                    tests.append(("comprehensiveness", "magnitude"))
                for test_name, match in tests:
                    controls = F.controls(
                        N_CONTROLS,
                        seed=10000 + sample.index,
                        match=match,
                    )
                    if test_name == "comprehensiveness":
                        template = F.comprehensiveness(
                            target=metric,
                            min_drop=PRIMARY_T * margin,
                            statement=f"the {method} top-{k} units are necessary for the margin",
                            controls=controls,
                            replacement=rep,
                        )
                    else:
                        template = F.sufficiency(
                            target=metric,
                            max_drop=PRIMARY_T * margin,
                            statement=f"the {method} top-{k} units suffice for the margin",
                            controls=controls,
                            replacement=rep,
                        )
                    res = F.run(
                        model,
                        *sample.inputs,
                        test=template,
                        selection=selection,
                        attributions=attributions,
                        model_kwargs=kw,
                    )
                    st = res.statistics
                    below = float(st.get("fraction_below", math.nan))  # type: ignore[arg-type]
                    tied = float(st.get("fraction_tied", math.nan))  # type: ignore[arg-type]
                    above = float(st.get("fraction_above", math.nan))  # type: ignore[arg-type]
                    sup = (below if test_name == "comprehensiveness" else above) + 0.5 * tied
                    cdrops = [float(d) for d in st.get("control_drops", ())]  # type: ignore[union-attr]
                    rows.append(
                        [
                            rname,
                            method,
                            k,
                            ps,
                            test_name,
                            match,
                            res.drop,
                            res.outcome.value,
                            bool(st.get("no_op")),
                            below,
                            tied,
                            above,
                            sup,
                            float(st.get("mc_p_value", math.nan)),  # type: ignore[arg-type]
                            sorted(cdrops)[len(cdrops) // 2] if cdrops else None,
                            list(res.selection.order[:k]) if res.selection.k else None,
                            res.result.id,
                            sorted({lim.code for lim in res.limitations}),
                        ]
                    )
                    if verify:
                        try:
                            bnn.compose(
                                bnn.trace(model, *sample.inputs, model_kwargs=kw),
                                attributions=attributions,
                                faithfulness=[res],
                            )
                            verified += 1
                        except Exception as exc:  # recorded, not hidden
                            failures += 1
                            rows[-1].append(f"{type(exc).__name__}: {exc}")
    return {
        "index": sample.index,
        "label": sample.label,
        "text": sample.text,
        "pred": pred,
        "runner_up": runner,
        "margin": margin,
        "n_units": n,
        "completeness_delta": completeness,
        "orders": orders,
        "scores": scores,
        "rows": rows,
        "seconds": time.perf_counter() - t0,
        "composition_verified": verified,
        "composition_failures": failures,
    }


ROW_FIELDS = [
    "replacement",
    "method",
    "k",
    "p",
    "test",
    "controls",
    "drop",
    "outcome",
    "no_op",
    "fraction_below",
    "fraction_tied",
    "fraction_above",
    "superiority",
    "mc_p_value",
    "median_control_drop",
    "selected",
    "result_id",
    "limitation_codes",
    "composition_error",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["A", "B", "C"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    build = {"A": settings_A, "B": settings_B, "C": settings_C}[args.model]
    model, samples, sets, info, path = build()
    samples = samples[: args.limit] if args.limit else samples
    started = time.perf_counter()
    out: dict[str, Any] = {
        "plan": "docs/PHASE_5_5_PLAN.md",
        "environment": common.environment(),
        "model": info,
        "samples": [s.index for s in samples],
        "row_fields": ROW_FIELDS,
        "settings": {},
    }
    for setting in sets:
        results = []
        for i, sample in enumerate(samples):
            results.append(run_sample(model, setting, sample, path, verify=i == 0))
            print(
                f"{setting.model_id}/{setting.site} {i + 1}/{len(samples)} "
                f"{results[-1]['seconds']:.1f}s",
                flush=True,
            )
        out["settings"][setting.site] = {
            "unit_axes": setting.unit_axes,
            "ig_implementation": "captum" if setting.captum_ig else "beyondnn",
            "samples": results,
        }
    out["runtime_seconds"] = time.perf_counter() - started
    suffix = f"_limit{args.limit}" if args.limit else ""
    target = common.RESULTS / f"faithfulness_{args.model}{suffix}.json.gz"
    with gzip.open(target, "wt") as fh:
        json.dump(out, fh, sort_keys=True, default=str)
    print(f"wrote {target} in {out['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
