# ruff: noqa: E501
"""Phase 7.75 concept validation on TD numeric Tracr programs (docs/PHASE_7_75_PLAN.md §4).

Each program: numeric indicators ``is_u`` (the used token) and ``is_d`` (the decoy token),
combined by ``numerical(LinearSequenceMap(is_u, is_d, a, 0))`` -- the decoy is read with
coefficient exactly 0 -- and averaged over the prefix (numerical Aggregate) into the output at the
last position. Truth from the program text: K+ (the ``is_u`` residual direction) is encoded and
used; K- (the ``is_d`` direction) is decodable and unused; K0 is K+ with permuted labels.

Also re-runs the Phase-7.5 held-out Tracr case 39 (``frac_prevs(is_x)``) under the Phase-7.75
criterion -- NOT blind (its Phase-7.5 use effect was known when the criterion was chosen).

Use criterion (ADR-055, frozen in the plan): min_change = C_REL x SD of the clean target over the
concept dataset's train split (C_REL = 0.2), computed before the use test; the Phase-7.5 absolute
0.1 is also run and reported for comparison.

Usage (InterpBench env, from experiments/phase7_5/artifacts): python ../../phase7_75/td_concepts.py dev|heldout|case39
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase7_5"))
import td_programs as TD  # noqa: E402,F401  (path setup for circuits-benchmark)
import torch  # noqa: E402
from circuits_benchmark.benchmark.common_programs import make_frac_prevs  # noqa: E402
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase  # noqa: E402
from tracr.rasp import rasp  # noqa: E402

import beyondnn as bnn  # noqa: E402

C, AU, iv = bnn.concepts, bnn.audits, bnn.interventions
RESULTS = HERE / "results"
DATA_MAX = 400
SPLIT_SEED = 1234
PERMUTATION_SEED = 4242
C_REL = 0.2
ABS = 0.1
ENC = C.encoding_criteria(min_fraction_below=0.95)
RULES = {"strict": (0.10, 0.10), "lenient": (0.30, 0.30)}


class NumericProgram(TracrBenchmarkCase):
    def __init__(
        self, name: str, used: str, decoy: str, coef: float, vocab: set[str], max_len: int
    ) -> None:
        super().__init__()
        self.name_, self.used, self.decoy, self.coef = name, used, decoy, coef
        self.vocab, self.max_len = vocab, max_len

    def get_name(self) -> str:
        return self.name_

    def get_program(self) -> rasp.SOp:
        is_u = rasp.numerical(
            rasp.Map(lambda t, u=self.used: 1 if t == u else 0, rasp.tokens)
        ).named("is_u")
        is_d = rasp.numerical(
            rasp.Map(lambda t, d=self.decoy: 1 if t == d else 0, rasp.tokens)
        ).named("is_d")
        comb = rasp.numerical(rasp.LinearSequenceMap(is_u, is_d, self.coef, 0.0)).named("comb")
        return make_frac_prevs(comb)

    def get_vocab(self) -> set[str]:
        return set(self.vocab)

    def get_max_seq_len(self) -> int:
        return self.max_len

    def get_task_description(self) -> str:
        return f"{self.coef} x fraction of '{self.used}' so far; '{self.decoy}' read with coefficient 0"

    def get_correct_output_for_input(self, inp: Any) -> Any:
        return [self.coef * list(inp[: i + 1]).count(self.used) / (i + 1) for i in range(len(inp))]


PROGRAMS = {
    "dev": {"cdev_a_b": ("a", "b", 1.0, set("abcde"), 10)},
    "heldout": {
        "cho_c_d": ("c", "d", 1.0, set("abcdef"), 14),
        "cho_x_a": ("x", "a", 1.0, set("abcdx"), 20),
        "cho_half_e_b": ("e", "b", 0.5, set("abcde"), 12),
        "cho_b_c": ("b", "c", 1.0, set("abcdefgh"), 16),
    },
}


def label_of(xs: torch.Tensor, token_id: int, splits: list[str]) -> tuple[list[int], float]:
    frac = (xs[:, 1:] == token_id).double().mean(1)
    train_median = float(frac[[i for i in range(len(xs)) if splits[i] == "train"]].median())
    return (frac > train_median).long().tolist(), train_median


def run_concept(
    model: Any,
    data: Any,
    feature: Any,
    name: str,
    target: Any,
    known: str,
    min_changes: dict[str, float],
) -> dict[str, Any]:
    concept = C.propose(feature, label=name, definition=name, label_source="dataset")
    encs = {
        d: C.encoding_test(
            model,
            concept,
            data,
            controls=[
                C.random_directions(200, seed=11, distribution=d),
                C.label_permutation(200, seed=12),
            ],
            criteria=ENC,
        )
        for d in ("covariance", "isotropic")
    }
    out: dict[str, Any] = {
        "known": known,
        "auroc": encs["covariance"].auroc,
        "encoding": {d: e.outcome.value for d, e in encs.items()},
        "rules": {},
    }
    validations = {}
    for rule, mc in min_changes.items():
        use = C.use_test(
            model,
            concept,
            data,
            target=target,
            relation="decreases",
            intervention=C.remove(C.zero()),
            controls=[C.random_directions(50, seed=13, distribution="covariance")],
            criteria=C.use_criteria(min_change=mc, min_fraction_beyond_controls=0.95),
        )
        vals = {d: C.validate(concept, encoding=encs[d], use=[use]) for d in encs}
        validations[rule] = vals
        out["rules"][rule] = {
            "min_change": mc,
            "use": use.outcome.value,
            "use_effect": use.mean_effect,
            "use_superiority": use.result.statistics.get("superiority"),
            "validation": {d: v.semantic_status.value for d, v in vals.items()},
        }
    out["_objects"] = (concept, encs, validations, data)
    return out


def run_model(
    model: Any,
    xs: torch.Tensor,
    used_id: int,
    decoy_id: int | None,
    f_used: Any,
    f_decoy: Any | None,
) -> dict[str, Any]:
    n = len(xs)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(SPLIT_SEED)).tolist()
    n_train, n_val = n // 2, n // 5
    splits = [""] * n
    for rank, i in enumerate(order):
        splits[i] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
    P = xs.shape[1] - 1
    target = iv.metrics.select([0, P, 0])
    with torch.no_grad():
        y_train = model(xs[[i for i in range(n) if splits[i] == "train"]])[:, P, 0]
    rel = C_REL * float(y_train.std())
    min_changes = {"rel": rel, "abs": ABS}
    rows = list(xs.split(1))
    lab_u, med_u = label_of(xs, used_id, splits)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(PERMUTATION_SEED)).tolist()
    concepts: dict[str, Any] = {}
    ds_u = C.dataset(rows, lab_u, splits, name="k_plus", label_source="program")
    concepts["K+"] = run_concept(
        model,
        ds_u,
        f_used(ds_u) if callable(f_used) else f_used,
        "k_plus",
        target,
        "positive",
        min_changes,
    )
    ds_0 = C.dataset(rows, [lab_u[i] for i in perm], splits, name="k_zero", label_source="permuted")
    concepts["K0"] = run_concept(
        model,
        ds_0,
        f_used(ds_0) if callable(f_used) else f_used,
        "k_zero",
        target,
        "negative",
        min_changes,
    )
    if decoy_id is not None and f_decoy is not None:
        lab_d, _ = label_of(xs, decoy_id, splits)
        ds_d = C.dataset(rows, lab_d, splits, name="k_minus", label_source="program")
        concepts["K-"] = run_concept(
            model, ds_d, f_decoy, "k_minus", target, "decodable_unused", min_changes
        )
    # audits under the strict / lenient caps with the Phase-7.5 concept roles, per use rule
    audits: dict[str, Any] = {}
    roles = [
        AU.role("null", "*covariance*", "primary"),
        AU.role("null", "*isotropic*", "alternative"),
        AU.role("replacement", "*", "primary"),
        AU.role("dataset", "*", "primary"),
    ]
    for rule in min_changes:
        ev = [
            v
            for r in concepts.values()
            for v in (*r["_objects"][2][rule].values(), *r["_objects"][1].values())
        ]
        for cap, (fp, fn) in RULES.items():
            plan = AU.plan(
                name=f"td_concepts_{rule}_{cap}",
                checkpoint=AU.checkpoint_of(model),
                declared_model=None,
                samples=[],
                datasets=sorted({r["_objects"][3].id for r in concepts.values()}),
                claims=[],
                requirements=[],
                concepts=[
                    AU.concept(r["_objects"][0].id, policy=C.POLICY_V1, roles=roles)
                    for r in concepts.values()
                ],
                counterexamples=AU.counterexample_rule(
                    max_counterexample_fraction=None,
                    max_false_positive_rate=fp,
                    max_false_negative_rate=fn,
                ),
                naive_auroc=0.6,
            )
            report = bnn.audit(ev, plan=plan)
            AU.verify_report(report.to_dict(), ev, plan)
            by_id = {r["_objects"][0].id: k for k, r in concepts.items()}
            audits[f"{rule}|{cap}"] = {
                by_id[k.concept.concept]: {
                    "standing": k.standing.value,
                    "findings": sorted({f.code for f in k.findings}),
                }
                for k in report.concepts
            }
    for r in concepts.values():
        r.pop("_objects")
    return {
        "n": n,
        "rel_min_change": rel,
        "train_median_fraction": med_u,
        "concepts": concepts,
        "audits": audits,
    }


def residual_direction(hl: Any, label: str) -> torch.Tensor:
    labels = list(hl.residual_stream_labels)
    idx = [
        i
        for i, lab in enumerate(labels)
        if lab.split(":")[0].rstrip("0123456789").rstrip("_") == label or lab == label
    ]
    assert len(idx) == 1, (label, [labels[i] for i in idx])
    v = torch.zeros(len(labels))
    v[idx[0]] = 1.0
    return v


def run_program(name: str, spec: tuple[Any, ...]) -> dict[str, Any]:
    used, decoy, coef, vocab, max_len = spec
    case = NumericProgram(name, used, decoy, coef, vocab, max_len)
    hl = case.get_hl_model(device="cpu").eval()
    data = case.get_clean_data(max_samples=DATA_MAX, seed=42, unique_data=True)
    xs = torch.stack([data[i][0] for i in range(len(data))])
    keep, seen = [], set()
    for i in range(len(xs)):
        key = tuple(int(v) for v in xs[i])
        if key not in seen:
            seen.add(key)
            keep.append(i)
    xs = xs[keep]
    enc = hl.tracr_input_encoder.encoding_map
    site = "blocks.0.hook_resid_post"
    f_u = C.direction(site, residual_direction(hl, "is_u"), axis=2, pooling="mean")
    f_d = C.direction(site, residual_direction(hl, "is_d"), axis=2, pooling="mean")
    out = run_model(hl, xs, enc[used], enc[decoy], f_u, f_d)
    return {
        "program": name,
        "task": case.get_task_description(),
        "residual_labels": list(hl.residual_stream_labels),
    } | out


def run_case39() -> dict[str, Any]:
    from external_interpbench import load_ll

    mod = importlib.import_module("circuits_benchmark.benchmark.cases.case_39")
    case = mod.Case39()
    hl = case.get_hl_model(device="cpu").eval()
    data = case.get_clean_data(max_samples=DATA_MAX, seed=42, unique_data=True)
    xs = torch.stack([data[i][0] for i in range(len(data))])
    x_id = hl.tracr_input_encoder.encoding_map["x"]
    site = "blocks.0.hook_resid_post"
    labels = list(hl.residual_stream_labels)
    v = torch.zeros(len(labels))
    v[labels.index("is_x")] = 1.0
    out = {"hl": run_model(hl, xs, x_id, None, C.direction(site, v, axis=2, pooling="mean"), None)}
    ll = load_ll("39")
    # LL K+ (as Phase 7.5): the direction fitted on each concept dataset's train split
    out["ll"] = run_model(
        ll,
        xs,
        x_id,
        None,
        lambda ds: C.fit_direction(ll, ds, site=site, axis=2, pooling="mean"),
        None,
    )
    return {
        "case": "39",
        "blind": False,
        "note": "Phase-7.5 held-out case; its use effect (about 0.06) was known when ADR-055 was chosen",
    } | out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["dev", "heldout", "case39"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.split == "case39":
        out: dict[str, Any] = run_case39()
    else:
        out = {"split": args.split, "programs": {}}
        for name, spec in PROGRAMS[args.split].items():
            try:
                out["programs"][name] = run_program(name, spec)
            except Exception as exc:
                import traceback

                out["programs"][name] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc(),
                }
            print(name, out["programs"][name].get("error", "ok"), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"td_concepts_{args.split}.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
