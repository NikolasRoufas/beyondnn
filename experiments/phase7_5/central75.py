# ruff: noqa: E501, B007
"""Phase 7.5 central experiment under the frozen policy (docs/PHASE_7_5_FROZEN_POLICY.md §C).

Models/sites: A (MLP input, net.1), B (CNN pixels, relu2), C (BERT-tiny SST-2 tokens) with the
Phase-5.5 checkpoints, and D (BERT-base SST-2, textattack, held-out model, tokens).

Splits:
* ``dev``: the Phase-5.5 samples (A/B: 60, C: 40) -- these informed Phase 7;
* ``heldout``: the next samples in the same seeded order (A/B: up to 60, C: 40), and model D
  (40 samples), never inspected before the policy was frozen.

Evidence per sample and site, as in Phase 7 (D9): G / IG / R top-k at p in {5,10,20}%;
comprehensiveness (count controls, 0.95) and sufficiency (count controls, 0.95) under three
named replacements; comprehensiveness with magnitude controls and without controls under the
PRIMARY replacement; threshold 0.5 x margin; alternatives x0.5 and x1.5.

Frozen roles (per site): replacement PRIMARY = the mean / [MASK] replacement, ALTERNATIVE =
the resample / [PAD] replacement, STRESS_TEST = zero (A/input: zero == train mean is PRIMARY and
train-min is STRESS_TEST); k PRIMARY = the k of p = 10% (per sample), ALTERNATIVE = 5%, 20%;
null PRIMARY = count@0.95, ALTERNATIVE = magnitude@0.95, STRESS_TEST = none; threshold PRIMARY
= recorded, ALTERNATIVE = x0.5, STRESS_TEST = x1.5.

Usage (experiment env, from experiments/phase5_5):
    python ../phase7_5/central75.py A|B|C|D dev|heldout [--limit N]
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase5_5"))
import common  # noqa: E402
import run_faithfulness as RF  # noqa: E402
import torch  # noqa: E402

import beyondnn as bnn  # noqa: E402
from beyondnn.core.samples import sample_id  # noqa: E402
from beyondnn.schema import Site, SiteIO  # noqa: E402

A, F, AU, iv = bnn.attribution, bnn.faithfulness, bnn.audits, bnn.interventions
RESULTS = HERE / "results"
ARTIFACTS = HERE / "artifacts"
RESULTS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)

PS = (0.05, 0.10, 0.20)
PRIMARY_P = 0.10
N_CONTROLS = 50
PRIMARY_T = 0.5
FRACTION = 0.95
METHODS = ("gradient", "integrated_gradients", "random")
D_REPO = "textattack/bert-base-uncased-SST-2"
D_REVISION = "95f0f6f859b35c8ff0863ae3cd4e2dbc702c0ae2"
D_CALIBRATION_ROWS = 200
N = {"A": 60, "B": 60, "C": 40, "D": 40}
# replacement roles by site (name -> role), frozen
REPLACEMENT_ROLES = {
    ("A", "input"): {
        "r1_zero_train_mean": "primary",
        "r2_resample_train_row": "alternative",
        "r3_train_min": "stress_test",
    },
    ("A", "net.1"): {
        "r2_train_mean": "primary",
        "r3_resample_train_row": "alternative",
        "r1_zero": "stress_test",
    },
    ("B", "pixels"): {
        "r2_train_mean_image": "primary",
        "r3_resample_train_image": "alternative",
        "r1_zero": "stress_test",
    },
    ("B", "relu2"): {
        "r2_train_mean_map": "primary",
        "r3_resample_train_map": "alternative",
        "r1_zero": "stress_test",
    },
    ("C", "tokens"): {
        "r2_mask_embedding": "primary",
        "r3_pad_embedding": "alternative",
        "r1_zero": "stress_test",
    },
    ("D", "tokens"): {
        "r2_mask_embedding": "primary",
        "r3_pad_embedding": "alternative",
        "r1_zero": "stress_test",
    },
}


def ks_for(n: int) -> dict[float, int]:
    return {p: max(1, math.ceil(p * n)) for p in PS}


def named(reps: dict[str, Any]) -> dict[str, Any]:
    """Phase-5.5 replacements with declared names (ADR-048); zero stays unnamed ('zero')."""
    out = {}
    for name, rep in reps.items():
        if rep.tensor is None and name != "r1_zero(=train_mean)":
            out[name] = rep
        elif name == "r1_zero(=train_mean)":
            out["r1_zero_train_mean"] = F.replacement(torch.zeros(1, 30), name="r1_zero_train_mean")
        else:
            out[name] = F.replacement(rep.tensor, name=name)
    return out


def _slice(samples: list[Any], split: str, n: int) -> list[Any]:
    return samples[:n] if split == "dev" else samples[n : 2 * n]


def build(model_id: str, split: str) -> tuple[Any, list[Any], list[Any], dict[str, Any], str]:
    n = N[model_id]
    if model_id in ("A", "B"):
        model, data, info = common.train(model_id)
        chosen = common.select_samples(model, data.x, data.y, data.test, 10_000)
        chosen = _slice(chosen, split, n)
        samples = [RF.Sample(i, (data.x[i : i + 1],), {}, int(data.y[i])) for i in chosen]
        _, _, settings, _, path = getattr(RF, f"settings_{model_id}")()
        info["pool_size"] = len(common.select_samples(model, data.x, data.y, data.test, 10_000))
        return model, samples, settings, info, path
    if model_id == "C":
        model, tok, info = common.bert()
        pool, meta = RF.c_samples(model, tok, 10_000)
        info["dataset"] = meta
        _, _, settings, _, path = RF.settings_C()
        return model, _slice(pool, split, n), settings, info, path
    # D: held-out moderate model (BERT-base SST-2)
    if split != "heldout":
        raise SystemExit("model D exists only as held-out evidence")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(D_REPO, revision=D_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(D_REPO, revision=D_REVISION).eval()
    rows = common.sst2_validation()
    calib = rows[:D_CALIBRATION_ROWS]
    agree = 0
    for row in calib:
        enc = tok(row["sentence"], return_tensors="pt")
        with torch.no_grad():
            agree += int(model(**enc).logits.argmax(-1)) == int(row["label"])
    flip = agree < len(calib) / 2  # label mapping fixed on the calibration rows only
    eligible = []
    for row in rows[D_CALIBRATION_ROWS:]:
        enc = tok(row["sentence"], return_tensors="pt")
        t = enc["input_ids"].shape[1]
        if not 8 <= t <= 24:
            continue
        kwargs = {"attention_mask": enc["attention_mask"], "token_type_ids": enc["token_type_ids"]}
        with torch.no_grad():
            pred = int(model(enc["input_ids"], **kwargs).logits.argmax(-1))
        label = 1 - int(row["label"]) if flip else int(row["label"])
        if pred == label:
            eligible.append(
                RF.Sample(int(row["idx"]), (enc["input_ids"],), kwargs, pred, row["sentence"])
            )
    order = torch.randperm(
        len(eligible), generator=torch.Generator().manual_seed(common.SELECTION_SEED)
    )
    samples = [eligible[i] for i in order[:n].tolist()]
    emb = model.bert.embeddings.word_embeddings.weight.detach()

    def token_reps(_: int, inputs: tuple[Any, ...]) -> dict[str, Any]:
        t = inputs[0].shape[1]
        return {
            "r1_zero": F.zero(),
            "r2_mask_embedding": F.replacement(emb[tok.mask_token_id].expand(1, t, -1).clone()),
            "r3_pad_embedding": F.replacement(emb[tok.pad_token_id].expand(1, t, -1).clone()),
        }

    site = "bert.embeddings.word_embeddings"
    settings = [RF.Setting("D", "tokens", A.layer(site), site, (1,), False, token_reps)]
    info = {
        "repo": D_REPO,
        "revision": D_REVISION,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_state_digest": common.state_digest(model),
        "label_mapping": "flipped" if flip else "identity",
        "calibration_agreement": agree / len(calib),
        "eligible": len(eligible),
    }
    return model, samples, settings, info, '["logits"]'


def run_sample(
    model: Any, setting: Any, sample: Any, path: str
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    kw = sample.kwargs
    with torch.no_grad():
        out = model(*sample.inputs, **kw)
    logits = out.logits if path else out
    metric, pred, runner, margin = common.margin_target(logits, path=path)
    methods = RF._methods(setting)
    attrs = {
        name: A.attribute(
            model,
            *sample.inputs,
            target=metric,
            method=methods[name],
            at=setting.at,
            model_kwargs=kw,
        )
        for name in ("gradient", "integrated_gradients")
    }
    ax = setting.unit_axes
    n = F.ranking(
        attrs["gradient"], unit_axes=ax, reduce=RF.REDUCE["gradient"] if ax else None
    ).n_units
    perm = torch.randperm(
        n, generator=torch.Generator().manual_seed(RF.RANDOM_SEED_BASE + sample.index)
    ).tolist()
    ks = ks_for(n)
    roles = REPLACEMENT_ROLES[(setting.model_id, setting.site)]
    reps = named(setting.replacements(sample.index, sample.inputs))
    primary_rep = next(r for r, role in roles.items() if role == "primary")
    results: list[Any] = []
    single: dict[str, Any] = {}
    for rname, rep in reps.items():
        is_primary = rname == primary_rep
        for p, k in ks.items():
            for method in METHODS:
                if method == "random":
                    selection = F.units(setting.at, tuple(perm[:k]), n_units=n, unit_axes=ax)
                    attributions: list[Any] = []
                else:
                    selection = F.top_k(
                        attrs[method], k=k, unit_axes=ax, reduce=RF.REDUCE[method] if ax else None
                    )
                    attributions = [attrs[method]]
                tests = [("comprehensiveness", "count"), ("sufficiency", "count")]
                if is_primary:
                    tests += [("comprehensiveness", "magnitude"), ("comprehensiveness", "none")]
                for name, match in tests:
                    controls = (
                        None
                        if match == "none"
                        else F.controls(N_CONTROLS, seed=10000 + sample.index, match=match)
                    )
                    if name == "comprehensiveness":
                        extra = {} if controls is None else {"min_fraction_below": FRACTION}
                        template = F.comprehensiveness(
                            target=metric,
                            min_drop=PRIMARY_T * margin,
                            replacement=rep,
                            controls=controls,
                            statement=f"the {method} top-{k} units are necessary for the margin",
                            **extra,
                        )
                    else:
                        template = F.sufficiency(
                            target=metric,
                            max_drop=PRIMARY_T * margin,
                            replacement=rep,
                            controls=controls,
                            statement=f"the {method} top-{k} units suffice for the margin",
                            min_fraction_above=FRACTION,
                        )
                    res = F.run(
                        model,
                        *sample.inputs,
                        test=template,
                        selection=selection,
                        attributions=attributions,
                        model_kwargs=kw,
                    )
                    results.append(res)
                    if (
                        method == "integrated_gradients"
                        and is_primary
                        and p == PRIMARY_P
                        and name == "comprehensiveness"
                        and match == "count"
                    ):
                        single = {
                            "outcome": res.outcome.value,
                            "drop": res.drop,
                            "k": k,
                            "units": list(res.selection.selected),
                        }
    meta = {
        "index": sample.index,
        "sample_id": sample_id(*sample.inputs, model_kwargs=kw or None),
        "pred": pred,
        "runner_up": runner,
        "margin": margin,
        "n_units": n,
        "ks": {str(p): k for p, k in ks.items()},
        "primary_single_configuration": single,
        "text": sample.text,
    }
    return results, list(attrs.values()), meta


def roles_for(setting: Any, metas: list[dict[str, Any]]) -> list[Any]:
    out = []
    for name, role in REPLACEMENT_ROLES[(setting.model_id, setting.site)].items():
        out.append(
            AU.role("replacement", "zero" if name == "r1_zero" else f"tensor/{name}:*", role)
        )
    for m in metas:
        ks = m["ks"]
        primary_k = ks[str(PRIMARY_P)]
        out.append(AU.role("k", str(primary_k), "primary", sample=m["sample_id"]))
        for p, k in ks.items():
            if k != primary_k:
                out.append(AU.role("k", str(k), "alternative", sample=m["sample_id"]))
    out += [
        AU.role("null", "count@*", "primary"),
        AU.role("null", "magnitude@*", "alternative"),
        AU.role("null", "none", "stress_test"),
        AU.role("threshold", "{*}", "primary"),
        AU.role("threshold", "*|alt:*x0.5", "alternative"),
        AU.role("threshold", "*|alt:*x1.5", "stress_test"),
    ]
    return list({(r.axis, r.pattern, r.role, r.sample): r for r in out}.values())


def site_of(setting: Any) -> Site:
    if setting.module is None:
        return Site(module="", io=SiteIO.INPUT, output_path="args[0]")
    return Site(module=setting.module)


def plan_for(
    model: Any,
    setting: Any,
    targets: dict[str, Any],
    methods: dict[str, str],
    metas: list[dict[str, Any]],
) -> Any:
    roles = roles_for(setting, metas)
    claims = []
    for tag, method in methods.items():
        sel = AU.selection(site_of(setting), method=method, k=None)
        for rel, req, inv in (
            (
                "necessary_for",
                "necessity",
                [
                    AU.invariance("replacement", min_values=3),
                    AU.invariance("k", min_values=2),
                    AU.invariance("null", min_values=2),
                ],
            ),
            (
                "sufficient_for",
                "sufficiency",
                [AU.invariance("replacement", min_values=3), AU.invariance("k", min_values=2)],
            ),
        ):
            claims.append(
                AU.claim(
                    f"{tag}_{'necessary' if rel == 'necessary_for' else 'sufficient'}",
                    statement=f"the {tag} top-k units at {setting.site} {'are necessary' if rel == 'necessary_for' else 'suffice'} for the margin",
                    relation=rel,
                    target=None,
                    sample_targets=targets,
                    scope="instance",
                    requirement=req,
                    selection=sel,
                    invariant_over=inv,
                    roles=roles,
                )
            )
    return AU.plan(
        name=f"central75_{setting.model_id}_{setting.site.replace('.', '_')}".lower(),
        checkpoint=AU.checkpoint_of(model),
        declared_model=None,
        samples=list(targets),
        datasets=[],
        claims=claims,
        requirements=[
            AU.requirement(
                "necessity",
                policy=F.COMPREHENSIVENESS_POLICY,
                controls=True,
                alternatives=[
                    AU.alternative("comprehensiveness", "min_drop", factor=0.5),
                    AU.alternative("comprehensiveness", "min_drop", factor=1.5),
                ],
            ),
            AU.requirement(
                "sufficiency",
                policy=F.SUFFICIENCY_POLICY,
                controls=True,
                alternatives=[
                    AU.alternative("sufficiency", "max_drop", factor=0.5),
                    AU.alternative("sufficiency", "max_drop", factor=1.5),
                ],
            ),
        ],
        concepts=[],
        counterexamples=AU.counterexample_rule(
            max_counterexample_fraction=None,
            max_false_positive_rate=None,
            max_false_negative_rate=None,
        ),
        naive_auroc=None,
    )


def summarise(report: Any, metas: list[dict[str, Any]]) -> dict[str, Any]:
    by_sample = {m["sample_id"]: m for m in metas}
    out: dict[str, Any] = {}
    for c in report.claims:
        per_sample = []
        for g in c.groups:
            p = g.profile
            codes = sorted({f.code for f in g.findings})
            reversal_axes = sorted(
                {
                    f.axis
                    for f in g.findings
                    if f.code in ("alternative_reverses", "stress_test_reverses") and f.axis
                }
            )
            per_sample.append(
                {
                    "index": by_sample[g.sample]["index"],
                    "standing": g.standing.value,
                    "codes": codes,
                    "tested": p.tested if p else 0,
                    "outcomes": [list(o) for o in p.outcomes] if p else [],
                    "sensitive_axes": list(p.sensitive_axes) if p else [],
                    "reversal_axes": reversal_axes,
                }
            )
        out[c.name] = {
            "distribution": dict(c.distribution),
            "intervals": [i.describe() for i in c.intervals],
            "per_sample": per_sample,
            "claim_findings": [(f.kind.value, f.code, len(f.samples)) for f in c.findings],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["A", "B", "C", "D"])
    parser.add_argument("split", choices=["dev", "heldout"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--quiet", action="store_true", help="smoke test: print nothing about outcomes"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, samples, settings, info, path = build(args.model, args.split)
    if args.limit:
        samples = samples[: args.limit]
    payload: dict[str, Any] = {
        "environment": common.environment(),
        "model": info,
        "split": args.split,
        "policy": "docs/PHASE_7_5_FROZEN_POLICY.md §C",
        "settings": {},
    }
    for setting in settings:
        t0 = time.perf_counter()
        results, attributions, metas, targets = [], [], [], {}
        methods: dict[str, str] = {}
        for sample in samples:
            res, attrs, meta = run_sample(model, setting, sample, path)
            results += res
            attributions += attrs
            metas.append(meta)
            targets[meta["sample_id"]] = res[0].claim.target
            methods = {
                "g": attrs[0].record.method.name,
                "ig": attrs[1].record.method.name,
                "r": "declared",
            }
            if not args.quiet:
                print(
                    setting.site,
                    sample.index,
                    meta["primary_single_configuration"].get("outcome"),
                    flush=True,
                )
        run_seconds = time.perf_counter() - t0
        plan = plan_for(model, setting, targets, methods, metas)
        t1 = time.perf_counter()
        report = bnn.audit(results, plan=plan)
        audit_seconds = time.perf_counter() - t1
        AU.verify_report(report.to_dict(), results, plan)
        tag = f"{args.model}_{setting.site}_{args.split}" + (
            f"_limit{args.limit}" if args.limit else ""
        )
        path_r = ARTIFACTS / f"central75_{tag}_report.json"
        path_r.unlink(missing_ok=True)
        report.save(path_r)
        (ARTIFACTS / f"plan_central75_{tag}.json").write_text(bnn.schema.to_json(plan))
        payload["settings"][setting.site] = {
            "n_samples": len(samples),
            "plan_id": plan.id,
            "n_results": report.evidence.results,
            "results_excluded": report.evidence.results_excluded,
            "run_seconds": run_seconds,
            "audit_seconds": audit_seconds,
            "verify_report": "passed",
            "primary_single_configuration_supports": sum(
                1 for m in metas if m["primary_single_configuration"].get("outcome") == "supports"
            ),
            "audit": summarise(report, metas),
            "samples": metas,
        }
        del results, attributions, report
        gc.collect()
    tag = f"{args.model}_{args.split}" + (f"_limit{args.limit}" if args.limit else "")
    (RESULTS / f"central75_{tag}.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=str)
    )
    print("done", tag)


if __name__ == "__main__":
    main()
