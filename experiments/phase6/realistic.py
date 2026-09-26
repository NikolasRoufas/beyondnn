"""Phase 6 realistic concept validation (docs/PHASE_6_PLAN.md §22, R1-R7).

Models are the Phase-5.5 models (checkpoints verified by state digest):
A = MLP / breast_cancer, B = CNN / digits, C = BERT-tiny / SST-2 validation.

Usage (experiment environment, see experiments/phase5_5/requirements.txt):
  python experiments/phase6/realistic.py A|B|C
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase5_5"))
import common  # noqa: E402

import beyondnn as bnn  # noqa: E402
from beyondnn.concepts.verify import verify_validation  # noqa: E402

C, iv = bnn.concepts, bnn.interventions
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 1234
NEGATION = ("not", "n't", "no", "never", "nothing", "nobody")


def splits_for(n: int) -> list[str]:
    """Seeded (1234) 50 / 20 / 30 train / val / test assignment over the pool (plan §22)."""
    order = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).tolist()
    out = [""] * n
    n_train, n_val = n // 2, n // 5
    for rank, i in enumerate(order):
        out[i] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
    return out


def threshold_labels(values: list[float], splits: list[str]) -> list[int]:
    """Concept = value > median of the TRAIN split (plan §22)."""
    train = sorted(v for v, s in zip(values, splits, strict=True) if s == "train")
    m = len(train)
    median = train[m // 2] if m % 2 else (train[m // 2 - 1] + train[m // 2]) / 2
    return [int(v > median) for v in values]


def log_softmax_metric(c: int, path: str) -> Any:
    def f(out: Any) -> float:
        logits = out.logits if path else out
        return float(torch.log_softmax(logits.double(), dim=-1)[0, c])

    return iv.metrics.custom(
        f"log_softmax_c{c}", f, implementation_revision="phase6-v1", config={"class": c}
    )


def lift_class(
    labels: list[int], classes: list[int], splits: list[str]
) -> tuple[int, dict[str, float]]:
    """The class with the highest train-split lift P(class | C = 1) / P(class) (ties: lower)."""
    train = [i for i, s in enumerate(splits) if s == "train"]
    pos = [i for i in train if labels[i] == 1]
    lifts = {}
    for c in sorted(set(classes)):
        p_c = sum(1 for i in train if classes[i] == c) / len(train)
        p_c_pos = sum(1 for i in pos if classes[i] == c) / len(pos)
        lifts[str(c)] = p_c_pos / p_c
    best = max(sorted(lifts), key=lambda c: (lifts[c], -int(c)))
    return int(best), lifts


def mean_reference(model: Any, data: Any, feature: Any) -> torch.Tensor:
    """Train-split mean of the feature's site leaf (the 'train_mean' reference)."""
    from beyondnn.concepts.features import record_site, site_tensors

    train = data.record.indices("train")
    site = feature.record.site
    trace = record_site(model, data, train, site)
    tensors = [t for _, t in site_tensors(trace, data, train, site, feature.record.call_index)]
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) != 1:
        raise ValueError("variable-shape site: a mean reference is undefined (use pooled sites)")
    return torch.stack(tensors).mean(dim=0).float()


def enc_controls(distribution: str) -> list[Any]:
    return [
        C.random_directions(200, seed=11, distribution=distribution),
        C.label_permutation(200, seed=12),
    ]


def run_concept(
    model: Any,
    data: Any,
    name: str,
    definition: str,
    site: str,
    axis: int,
    pooling: str,
    target: Any,
    target_desc: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {"concept": name, "definition": definition, "target": target_desc}
    feats = {
        "direction": C.fit_direction(model, data, site=site, axis=axis, pooling=pooling),
        "neuron": C.search_neurons(model, data, site=site, axis=axis, pooling=pooling),
    }
    ecrit = C.encoding_criteria(min_fraction_below=0.95)
    ucrit = C.use_criteria(min_change=0.1, min_fraction_beyond_controls=0.95)
    for kind, feature in feats.items():
        row: dict[str, Any] = {"feature": feature.id}
        if kind == "neuron":
            row["neuron_index"] = feature.record.index
            row["search"] = feature.record.source.params.to_plain()
        concept = C.propose(feature, label=name, definition=definition, label_source="dataset")
        t0 = time.perf_counter()
        enc = C.encoding_test(
            model, concept, data, controls=enc_controls("covariance"), criteria=ecrit
        )
        row["encoding_seconds"] = time.perf_counter() - t0
        iso = C.encoding_test(
            model, concept, data, controls=enc_controls("isotropic"), criteria=ecrit
        )
        stats = enc.result.statistics["controls"]
        row |= {
            "auroc": enc.auroc,
            "encoding": enc.outcome.value,
            "controls_covariance": [
                {k: c[k] for k in ("kind", "superiority", "median", "q95")} for c in stats
            ],  # type: ignore[union-attr]
            "encoding_isotropic": iso.outcome.value,
            "controls_isotropic": [
                {k: c[k] for k in ("kind", "superiority", "median", "q95")}
                for c in iso.result.statistics["controls"]
            ],  # type: ignore[union-attr]
            "false_positive_rate": enc.counterexamples.measurements["false_positive_rate"],
            "false_negative_rate": enc.counterexamples.measurements["false_negative_rate"],
            "n_false_positives": len(enc.counterexamples.measurements["false_positives"]),  # type: ignore[arg-type]
            "n_false_negatives": len(enc.counterexamples.measurements["false_negatives"]),  # type: ignore[arg-type]
        }
        refs = {
            "zero": C.zero(),
            "train_mean": C.reference(mean_reference(model, data, feature), name="train_mean"),
        }
        ctl = (
            [C.random_neurons(50, seed=13)]
            if kind == "neuron"
            else [C.random_directions(50, seed=13, distribution="covariance")]
        )
        uses = {}
        for rname, ref in refs.items():
            t0 = time.perf_counter()
            use = C.use_test(
                model,
                concept,
                data,
                target=target,
                relation="decreases",
                intervention=C.remove(ref),
                controls=ctl,
                criteria=ucrit,
            )
            uses[rname] = use
            row[f"use_{rname}"] = {
                "outcome": use.outcome.value,
                "mean_effect": use.mean_effect,
                "superiority": use.result.statistics["superiority"],
                "control_median": sorted(use.result.statistics["control_mean_effects"])[25],  # type: ignore[index,arg-type]
                "n": use.result.statistics["n"],
                "seconds": time.perf_counter() - t0,
            }
            val = C.validate(concept, encoding=enc, use=[use])
            t0 = time.perf_counter()
            verify_validation(val)
            row[f"validation_{rname}"] = {
                "status": val.semantic_status.value,
                "unmet": list(val.unmet),
                "verify_seconds": time.perf_counter() - t0,
                "scope": val.scope,
            }
        out[kind] = row
        print(
            name,
            kind,
            json.dumps({k: row[k] for k in ("auroc", "encoding", "encoding_isotropic")}),
            {
                r: (
                    row[f"use_{r}"]["outcome"],
                    round(row[f"use_{r}"]["mean_effect"], 4),
                    row[f"validation_{r}"]["status"],
                )
                for r in refs
            },
            flush=True,
        )
    return out


def random_direction_proposals(
    model: Any, data: Any, name: str, site: str, axis: int, pooling: str, d: int
) -> dict[str, Any]:
    """R6: 20 seeded random directions proposed as the concept, evaluated on a seeded
    40-sample subset of the test split (train/val unchanged)."""
    test = list(data.record.indices("test"))
    keep = set(
        torch.randperm(len(test), generator=torch.Generator().manual_seed(SEED + 1))[:40].tolist()
    )
    kept = [test[k] for k in sorted(keep)]
    splits = list(data.record.splits)
    idx = [i for i, s in enumerate(splits) if s != "test" or i in set(kept)]
    small = C.dataset(
        [data.inputs[i] for i in idx],
        [data.record.labels[i] for i in idx],
        [splits[i] for i in idx],
        name=f"{data.record.name}[test subsample 40]",
        label_source=data.record.label_source,
        model_kwargs=[data.kwargs[i] for i in idx],
    )
    naive = passed = 0
    aurocs = []
    for s in range(20):
        v = torch.randn(d, generator=torch.Generator().manual_seed(5000 + s))
        concept = C.propose(
            C.direction(site, v, axis=axis, pooling=pooling),
            label=f"{name} (random direction {s})",
            definition="random",
        )
        enc = C.encoding_test(
            model,
            concept,
            small,
            controls=enc_controls("covariance"),
            criteria=C.encoding_criteria(min_fraction_below=0.95),
        )
        aurocs.append(enc.auroc)
        naive += enc.auroc >= 0.6
        passed += enc.outcome.value == "supports"
    return {
        "concept": name,
        "n": 20,
        "naive_auroc_ge_0.6": naive,
        "controls_supports": passed,
        "aurocs": aurocs,
    }


# ------------------------------------------------------------------ models


def model_a() -> tuple[Any, list[Any], dict[str, Any]]:
    from sklearn.datasets import load_breast_cancer

    model, data, info = common.train("A")
    raw = load_breast_cancer()
    n = len(data.y)
    splits = splits_for(n)
    samples = [data.x[i : i + 1] for i in range(n)]
    classes = data.y.tolist()
    runs = []
    for name, col in (("K1 size", 0), ("K2 texture", 1), ("K3 smoothness", 4), ("K4 fractal", 9)):
        labels = threshold_labels([float(v) for v in raw.data[:, col]], splits)
        ds = C.dataset(
            samples,
            labels,
            splits,
            name=f"breast_cancer[all 569; seed {SEED} 50/20/30]",
            label_source=f"{raw.feature_names[col]} > train median",
        )
        c, lifts = lift_class(labels, classes, splits)
        runs.append(
            (
                ds,
                name,
                f"{raw.feature_names[col]} > train-split median",
                "net.1",
                1,
                "none",
                log_softmax_metric(c, ""),
                {"log_softmax_class": c, "train_lift": lifts},
            )
        )
    return model, runs, info


def model_b() -> tuple[Any, list[Any], dict[str, Any]]:
    model, data, info = common.train("B")
    n = len(data.y)
    splits = splits_for(n)
    samples = [data.x[i : i + 1] for i in range(n)]
    classes = data.y.tolist()
    ink = [float(data.x[i].sum()) for i in range(n)]
    centre = [float(data.x[i, 0, 2:6, 2:6].sum()) for i in range(n)]
    runs = []
    for name, vals, desc in (
        ("K5 ink", ink, "total ink"),
        ("K6 centre", centre, "ink in the central 4x4"),
    ):
        labels = threshold_labels(vals, splits)
        ds = C.dataset(
            samples,
            labels,
            splits,
            name=f"digits[all 1797; seed {SEED} 50/20/30]",
            label_source=f"{desc} > train median",
        )
        c, lifts = lift_class(labels, classes, splits)
        runs.append(
            (
                ds,
                name,
                f"{desc} > train-split median",
                "relu2",
                1,
                "mean",
                log_softmax_metric(c, ""),
                {"log_softmax_class": c, "train_lift": lifts},
            )
        )
    return model, runs, info


def model_c() -> tuple[Any, list[Any], dict[str, Any]]:
    model, tok, info = common.bert()
    rows = common.sst2_validation()
    n = len(rows)
    splits = splits_for(n)
    samples, kwargs, lengths, neg = [], [], [], []
    for row in rows:
        enc = tok(row["sentence"], return_tensors="pt")
        samples.append((enc["input_ids"],))
        kwargs.append(
            {"attention_mask": enc["attention_mask"], "token_type_ids": enc["token_type_ids"]}
        )
        lengths.append(float(enc["input_ids"].shape[1]))
        words = row["sentence"].lower().split()
        neg.append(int(any(w in NEGATION or w.endswith("n't") for w in words)))
    classes = [int(r["label"]) for r in rows]
    info["dataset"] = {"validation_rows": n, "file_sha256": rows[0]["_file_sha256"]}
    runs = []
    for name, labels, desc in (
        ("K7 positive", classes, "SST-2 label = positive (label-aligned positive control)"),
        ("K8 length", threshold_labels(lengths, splits), "token count > train median"),
        ("K9 negation", neg, f"contains one of {NEGATION} (whitespace tokens, or ending in n't)"),
    ):
        ds = C.dataset(
            samples,
            labels,
            splits,
            name=f"sst2-validation[872; seed {SEED} 50/20/30]",
            label_source=desc,
            model_kwargs=kwargs,
        )
        c, lifts = lift_class(labels, classes, splits)
        runs.append(
            (
                ds,
                name,
                desc,
                "bert.pooler",
                1,
                "none",
                log_softmax_metric(c, '["logits"]'),
                {"log_softmax_class": c, "train_lift": lifts},
            )
        )
    return model, runs, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["A", "B", "C"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, runs, info = {"A": model_a, "B": model_b, "C": model_c}[args.model]()
    started = time.perf_counter()
    results = []
    for ds, name, definition, site, axis, pooling, target, desc in runs:
        results.append(run_concept(model, ds, name, definition, site, axis, pooling, target, desc))
    ds0, name0, _, site0, axis0, pool0, _, _ = runs[0]
    from beyondnn.concepts.features import record_site, site_tensors

    one = record_site(model, ds0, [0], bnn.schema.Site(module=site0))
    ((_, t),) = site_tensors(one, ds0, [0], bnn.schema.Site(module=site0), 0)
    random_row = random_direction_proposals(
        model, ds0, name0, site0, axis0, pool0, int(t.shape[axis0])
    )
    out = {
        "environment": common.environment(),
        "model": info,
        "concepts": results,
        "random_direction_proposals": random_row,
        "runtime_seconds": time.perf_counter() - started,
    }
    (RESULTS / f"realistic_{args.model}.json").write_text(
        json.dumps(
            out,
            indent=1,
            default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o),
        )
    )
    print(
        "done",
        args.model,
        round(out["runtime_seconds"]),
        "s",
        random_row["naive_auroc_ge_0.6"],
        random_row["controls_supports"],
    )


if __name__ == "__main__":
    main()
