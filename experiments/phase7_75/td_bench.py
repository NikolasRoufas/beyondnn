# ruff: noqa: B007
"""Phase 7.75 independent known-mechanism validation (docs/PHASE_7_75_PLAN.md §3, TD).

Ground truth is structural and programmatic (``td_programs.py``, ``td_structure.py``): a
component is known-true if it writes only variables the output function depends on, known-false
if it writes only decoy variables (read by the output function but ignored in the program
source), empty if it writes nothing. No intervention defines any label.

BeyondNN evidence per (component, sample), on the compiled (Tracr) model, exactly as the
Phase-7.5 E1 design: comprehensiveness with min_drop = clean margin (``metrics.margin``: the
argmax changes) under resample (PRIMARY; source = next sample), mean (ALTERNATIVE) and zero
(STRESS_TEST) replacements; count controls for heads (ALTERNATIVE null); thresholds x0.5
(ALTERNATIVE) and x1.5 (STRESS_TEST); IG top-1 head per layer with >= 2 heads. Separately,
an independent TransformerLens interchange of the same node (question A only).

Usage (InterpBench env, from experiments/phase7_5/artifacts):
    python ../../phase7_75/td_bench.py dev|heldout
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import td_programs as TD  # noqa: E402
import td_structure  # noqa: E402
import torch  # noqa: E402

import beyondnn as bnn  # noqa: E402
from beyondnn.core.samples import sample_id  # noqa: E402
from beyondnn.schema import Site, Subject  # noqa: E402

A, F, AU, iv = bnn.attribution, bnn.faithfulness, bnn.audits, bnn.interventions
RESULTS = HERE / "results"
ARTIFACTS = HERE / "artifacts"

# ------------------------------------------------------------------ frozen parameters
N_SAMPLES = 40
DATA_MAX = 400
DATA_SEED = 42
N_CONTROLS = 20
CONTROL_SEED = 777
FRACTION = 0.95
IG_STEPS = 32
ROLES = [
    ("replacement", "tensor/resample:*", "primary"),
    ("replacement", "tensor/mean:*", "alternative"),
    ("replacement", "zero", "stress_test"),
    ("null", "none", "primary"),
    ("null", "count@*", "alternative"),
    ("threshold", "{*}", "primary"),
    ("threshold", "*|alt:min_dropx0.5", "alternative"),
    ("threshold", "*|alt:min_dropx1.5", "stress_test"),
]


def environment() -> dict[str, Any]:
    from importlib.metadata import version

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE
    ).stdout.strip()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformer_lens": version("transformer-lens"),
        "beyondnn_commit": commit,
    }


def interchange(
    model: Any, name: str, index: Any, x: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    """TransformerLens interchange of ``name`` (restricted to ``index``) from ``source``;
    independent of BeyondNN (question A only)."""
    _, cache = model.run_with_cache(source, names_filter=lambda n: n == name)
    src = cache[name]

    def hook(act: torch.Tensor, hook: Any) -> torch.Tensor:
        out = act.clone()
        if index is None:
            out[:] = src
        else:
            out[index] = src[index]
        return out

    with torch.no_grad():
        return model.run_with_hooks(x, fwd_hooks=[(name, hook)])


def run_program(name: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    case = TD.TDCase(TD.PROGRAMS[name])
    model = case.get_hl_model(device="cpu").eval()
    struct = td_structure.structure(name)
    labels = {k: v["label"] for k, v in struct["components"].items()}
    cfg = model.cfg
    data = case.get_clean_data(max_samples=DATA_MAX, seed=DATA_SEED, unique_data=True)
    xs_all = torch.stack([data[i][0] for i in range(len(data))])
    P = xs_all.shape[1] - 1
    decode = {v: k for k, v in model.tracr_output_encoder.encoding_map.items()}
    with torch.no_grad():
        pred_all = model(xs_all)[:, P].argmax(-1)
    # the compiled model must reproduce the RASP interpreter (Tracr's guarantee); checked
    inv_in = {v: k for k, v in model.tracr_input_encoder.encoding_map.items()}
    agree = []
    for i in range(len(xs_all)):
        toks = [inv_in[int(t)] for t in xs_all[i]]
        want = case.get_correct_output_for_input(toks[1:])[P - 1]
        got = decode.get(int(pred_all[i]))
        agree.append(want is not None and want == got)
    seen: set[tuple[int, ...]] = set()
    chosen = []
    for i in range(len(xs_all)):
        key = tuple(int(v) for v in xs_all[i])
        if agree[i] and key not in seen:
            seen.add(key)
            chosen.append(i)
    chosen = chosen[:N_SAMPLES]
    xs = xs_all[chosen]
    n = len(xs)
    nodes: list[tuple[str, int, int | None, str]] = [
        (f"blocks.{L}.attn.hook_result", L, h, f"l{L}_h{h}")
        for L in range(cfg.n_layers)
        for h in range(cfg.n_heads)
    ] + [(f"blocks.{L}.hook_mlp_out", L, None, f"l{L}_mlp") for L in range(cfg.n_layers)]
    sites = sorted({s for s, _, _, _ in nodes})
    _, cache = model.run_with_cache(xs, names_filter=lambda nm: nm in sites)
    acts = {s: cache[s].detach() for s in sites}
    samples = [sample_id(xs[i : i + 1]) for i in range(n)]
    targets: dict[str, Any] = {}
    metrics, margins, preds = [], [], []
    with torch.no_grad():
        logits = model(xs)[:, P]
    for i in range(n):
        top = torch.topk(logits[i], 2)
        pred = int(top.indices[0])
        metric = iv.metrics.margin([0, P, pred])
        metrics.append(metric)
        margins.append(float(top.values[0] - top.values[1]))
        preds.append(pred)
        targets[samples[i]] = metric
    # question A: independent TransformerLens interchange (PRIMARY semantics), per node/sample
    tl_flip: dict[str, list[bool]] = {}
    for site, L, h, key in nodes:
        flips = []
        for i in range(n):
            j = (i + 1) % n
            idx = None if h is None else (slice(None), slice(None), h)
            out = interchange(model, site, idx, xs[i : i + 1], xs[j : j + 1])[0, P].argmax(-1)
            flips.append(int(out) != preds[i])
        tl_flip[key] = flips
    evidence: list[Any] = []
    attributions: list[Any] = []
    ig = A.integrated_gradients(baseline=A.zero_baseline(), n_steps=IG_STEPS)
    ig_selected: dict[tuple[int, int], int] = {}
    ig_layers = [L for L in range(cfg.n_layers) if cfg.n_heads >= 2]
    for i in range(n):
        x = xs[i : i + 1]
        j = (i + 1) % n
        for site, L, h, key in nodes:
            shape = acts[site].shape
            units = (h,) if h is not None else tuple(range(shape[1]))
            axes = (2,) if h is not None else (1,)
            n_units = shape[2] if h is not None else shape[1]
            reps = {
                "resample": F.replacement(acts[site][j : j + 1], name="resample"),
                "mean": F.replacement(acts[site].mean(0, keepdim=True), name="mean"),
                "zero": F.zero(),
            }
            sel = F.units(A.layer(site), units, n_units=n_units, unit_axes=axes)
            for rname, rep in reps.items():
                test = F.comprehensiveness(
                    target=metrics[i],
                    min_drop=margins[i],
                    statement=f"{key} necessary",
                    replacement=rep,
                )
                evidence.append(F.run(model, x, test=test, selection=sel))
                if h is not None and rname == "resample" and cfg.n_heads >= 2:
                    ctest = F.comprehensiveness(
                        target=metrics[i],
                        min_drop=margins[i],
                        statement=f"{key} necessary",
                        replacement=rep,
                        controls=F.controls(N_CONTROLS, seed=CONTROL_SEED + i),
                        min_fraction_below=FRACTION,
                    )
                    evidence.append(F.run(model, x, test=ctest, selection=sel))
        for L in ig_layers:
            site = f"blocks.{L}.attn.hook_result"
            attr = A.attribute(model, x, target=metrics[i], method=ig, at=A.layer(site))
            attributions.append(attr)
            sel = F.top_k(attr, k=1, unit_axes=(2,), reduce="sum")
            ig_selected[(i, L)] = sel.selected[0]
            for rname, rep in {
                "resample": F.replacement(acts[site][j : j + 1], name="resample"),
                "mean": F.replacement(acts[site].mean(0, keepdim=True), name="mean"),
                "zero": F.zero(),
            }.items():
                test = F.comprehensiveness(
                    target=metrics[i],
                    min_drop=margins[i],
                    statement=f"IG top-1 head at {site}",
                    replacement=rep,
                )
                evidence.append(F.run(model, x, test=test, selection=sel, attributions=[attr]))
    roles = [AU.role(a, p, r) for a, p, r in ROLES]
    claims = []
    for site, L, h, key in nodes:
        shape = acts[site].shape
        units = (h,) if h is not None else tuple(range(shape[1]))
        axes = (2,) if h is not None else (1,)
        claims.append(
            AU.claim(
                f"{key}_necessary",
                statement=f"{key} is necessary for the output at the last position",
                relation="necessary_for",
                target=None,
                sample_targets=targets,
                scope="instance",
                requirement="necessity",
                subject=Subject(site=Site(module=site), units=units, unit_axes=axes),
                roles=roles,
            )
        )
    for L in ig_layers:
        claims.append(
            AU.claim(
                f"l{L}_ig_top1_necessary",
                statement=f"the IG top-1 head of layer {L} is necessary for the output",
                relation="necessary_for",
                target=None,
                sample_targets=targets,
                scope="instance",
                requirement="necessity",
                selection=AU.selection(
                    f"blocks.{L}.attn.hook_result", method="integrated_gradients", k=1
                ),
                roles=roles,
            )
        )
    plan = AU.plan(
        name=f"td_{name}",
        checkpoint=AU.checkpoint_of(model),
        declared_model=None,
        samples=samples,
        datasets=[],
        claims=claims,
        requirements=[
            AU.requirement(
                "necessity",
                policy=F.COMPREHENSIVENESS_POLICY,
                controls=False,
                alternatives=[
                    AU.alternative("comprehensiveness", "min_drop", factor=0.5),
                    AU.alternative("comprehensiveness", "min_drop", factor=1.5),
                ],
            )
        ],
        concepts=[],
        counterexamples=AU.counterexample_rule(
            max_counterexample_fraction=None,
            max_false_positive_rate=None,
            max_false_negative_rate=None,
        ),
        naive_auroc=None,
    )
    ev = [*evidence, *attributions]
    report = bnn.audit(ev, plan=plan)
    AU.verify_report(report.to_dict(), ev, plan)
    attribution_only = bnn.audit(attributions, plan=plan) if attributions else None
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / f"td_{name}_report.json").unlink(missing_ok=True)
    report.save(ARTIFACTS / f"td_{name}_report.json")

    def truth(claim: Any, i: int) -> str:
        if claim.claim.selection is not None:
            L = int(claim.claim.selection.site.module.split(".")[1])
            return labels[f"l{L}_h{ig_selected[(i, L)]}"]
        s = claim.claim.subject
        L = int(s.site.module.split(".")[1])
        return labels[f"l{L}_h{s.units[0]}" if s.unit_axes == (2,) else f"l{L}_mlp"]

    rows = []
    primary_flip: dict[str, list[bool | None]] = {}
    for claim in report.claims:
        comp = claim.name.removesuffix("_necessary")
        for i, g in enumerate(sorted(claim.groups, key=lambda g: samples.index(g.sample))):
            per_config: dict[str, str] = {}
            if g.profile is not None:
                for cf in g.profile.configurations:
                    ax = dict(cf.axes)
                    per_config[
                        "|".join(
                            [
                                ax.get("replacement", "-").split(":")[0],
                                ax.get("null", "-").split("@")[0],
                                cf.alternative or "recorded",
                            ]
                        )
                    ] = cf.outcome
            rows.append(
                {
                    "program": name,
                    "claim": claim.name,
                    "component": comp if claim.claim.selection is None else f"ig:{comp}",
                    "kind": "selection"
                    if claim.claim.selection
                    else ("head" if claim.claim.subject.unit_axes == (2,) else "mlp"),
                    "sample": i,
                    "truth": truth(claim, i),
                    "standing": g.standing.value,
                    "codes": sorted({f.code for f in g.findings}),
                    "configs": per_config,
                    "tl_interchange_flips": tl_flip.get(comp, [None] * n)[i],
                }
            )
            if claim.claim.selection is None:
                primary_flip.setdefault(comp, []).append(
                    {"supports": True, "contradicts": False}.get(
                        per_config.get("tensor/resample|none|recorded", "")
                    )
                )
    return {
        "program": name,
        "description": TD.PROGRAMS[name].description,
        "roles_from_source": case.roles,
        "structure": struct,
        "n_candidates": len(xs_all),
        "compiled_matches_interpreter": sum(agree) / len(agree),
        "n_samples": n,
        "target_position": P,
        "plan_id": plan.id,
        "n_results": report.evidence.results,
        "results_excluded": report.evidence.results_excluded,
        "report_findings": [(f.kind.value, f.code, len(f.records)) for f in report.findings],
        "rows": rows,
        "attribution_only": None
        if attribution_only is None
        else {c.name: dict(c.distribution) for c in attribution_only.claims},
        "ig_selected": {f"{i}:{L}": h for (i, L), h in ig_selected.items()},
        "seconds": time.perf_counter() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["dev", "heldout"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    progs = TD.DEV if args.split == "dev" else TD.HELDOUT
    out: dict[str, Any] = {
        "environment": environment(),
        "split": args.split,
        "policy": "docs/PHASE_7_75_PLAN.md §3",
        "programs": {},
    }
    for p in progs:
        try:
            out["programs"][p] = run_program(p)
        except Exception as exc:  # reported, never hidden
            import traceback

            out["programs"][p] = {
                "program": p,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(),
            }
        print(p, out["programs"][p].get("error", "ok"), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"td_{args.split}.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
