"""TNNLS revision protocol P1 (docs/tnnls_revision/protocols/P1_CENTRAL_RERUN_PROTOCOL.md).

Re-runs the frozen Phase-7.75 central design (``central75`` / ``central775`` functions, unchanged)
and additionally exports, per sample, every tested configuration of the necessity claims with
its axis values, role and outcome (from the audit report's sensitivity profile), plus a cost
breakdown. Optionally retrains models A/B with another seed (independent checkpoints).

Never writes into the repository's frozen results: everything goes to ``--out``.

Usage (from experiments/phase5_5, in the Phase-5.5 environment):
    python ../tnnls_revision/run_central_configs.py A heldout --out DIR [--seed S]
        [--shard i/n] [--save-evidence]
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for sub in ("phase5_5", "phase7_5", "phase7_75"):
    sys.path.insert(0, str(ROOT / "experiments" / sub))

import central75 as C75  # noqa: E402
import central775 as C775  # noqa: E402
import common  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import beyondnn as bnn  # noqa: E402

AU = bnn.audits


def train_seeded(kind: str, *, seed: int, ckpt_dir: Path) -> tuple[nn.Module, Any, dict[str, Any]]:
    """``common.train`` with the seed as a parameter (identical otherwise)."""
    data = common.breast_cancer() if kind == "A" else common.digits()
    torch.manual_seed(seed)
    model: nn.Module = common.MLP() if kind == "A" else common.CNN()
    path = ckpt_dir / f"model_{kind}_seed{seed}.pt"
    epochs, batch = (300, "full") if kind == "A" else (30, 64)
    config = {
        "optimizer": "Adam",
        "lr": 1e-3,
        "loss": "cross_entropy",
        "epochs": epochs,
        "batch": batch,
        "seed": seed,
    }
    if path.exists():
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        xtr, ytr = data.x[data.train], data.y[data.train]
        model.train()
        g = torch.Generator().manual_seed(seed)
        for _ in range(epochs):
            if kind == "A":
                opt.zero_grad()
                loss_fn(model(xtr), ytr).backward()
                opt.step()
            else:
                for b in torch.randperm(len(ytr), generator=g).split(64):
                    opt.zero_grad()
                    loss_fn(model(xtr[b]), ytr[b]).backward()
                    opt.step()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
    model.eval()
    info = {
        "architecture": repr(model),
        "training": config,
        "checkpoint_state_digest": common.state_digest(model),
        "accuracy": {
            s: common._accuracy(model, data.x[getattr(data, s)], data.y[getattr(data, s)])
            for s in ("train", "val", "test")
        },
        "dataset": data.meta,
    }
    return model, data, info


def _rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if sys.platform != "darwin" else r / (1024 * 1024)


def _du(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def config_rows(
    report: Any, metas: list[dict[str, Any]], tag: dict[str, Any]
) -> list[dict[str, Any]]:
    by_sid = {m["sample_id"]: m for m in metas}
    rows = []
    for c in report.claims:
        if not c.name.endswith("_necessary"):
            continue
        for g in c.groups:
            if g.sample is None or g.profile is None:
                continue
            m = by_sid[g.sample]
            for ce in g.profile.configurations:
                ax = dict(ce.axes)
                rows.append(
                    tag
                    | {
                        "sample_index": m["index"],
                        "primary_k": m["ks"][str(C75.PRIMARY_P)],
                        "claim": c.name,
                        "standing": g.standing.value,
                        "configuration": ce.configuration,
                        "role": ce.role,
                        "outcome": ce.outcome,
                        "alternative": ce.alternative or "",
                        "replacement": ax.get("replacement", ""),
                        "k": ax.get("k", ""),
                        "null": ax.get("null", ""),
                        "threshold": ax.get("threshold", ""),
                        "axes_json": json.dumps(ax, sort_keys=True),
                    }
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["A", "B", "C", "D"])
    ap.add_argument("split", choices=["dev", "heldout", "fresh"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None, help="retrain A/B with this seed")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--save-evidence", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    seed_tag = "frozen" if a.seed is None else f"seed{a.seed}"
    if a.seed is not None:
        if a.model not in ("A", "B"):
            raise SystemExit("--seed applies to the trained models A/B only")
        common.train = functools.partial(  # type: ignore[assignment]
            train_seeded, seed=a.seed, ckpt_dir=out.parent.parent / "checkpoints"
        )
    split = a.split
    if a.split == "fresh":  # protocol P3: positions 2n..3n-1 of the frozen seeded order
        if a.model not in ("B", "C"):
            raise SystemExit("fresh samples exist for B and C only (protocol P3)")
        C75._slice = lambda s, _split, n: s[2 * n : 3 * n]
        split = "heldout"
    t0 = time.perf_counter()
    model, samples, settings, info, path = C75.build(a.model, split)
    build_s = time.perf_counter() - t0
    if a.shard:
        i, n = (int(v) for v in a.shard.split("/"))
        samples = samples[i::n]
    payload: dict[str, Any] = {
        "environment": common.environment(),
        "model": info,
        "split": a.split,
        "eligibility": "all",
        "shard": a.shard,
        "seed": seed_tag,
        "protocol": "docs/tnnls_revision/protocols/P1_CENTRAL_RERUN_PROTOCOL.md",
        "settings": {},
    }
    timing: dict[str, Any] = {"build_seconds": build_s, "settings": {}}
    all_rows: list[dict[str, Any]] = []
    for setting in settings:
        t1 = time.perf_counter()
        results: list[Any] = []
        metas: list[dict[str, Any]] = []
        targets: dict[str, Any] = {}
        methods: dict[str, str] = {}
        for sample in samples:
            res, attrs, meta = C775.run_sample(model, setting, sample, path, None)
            results += res
            metas.append(meta)
            targets[meta["sample_id"]] = res[0].claim.target
            methods = {
                "g": attrs[0].record.method.name,
                "ig": attrs[1].record.method.name,
                "r": "declared",
            }
            print(setting.site, sample.index, flush=True)
        evidence_s = time.perf_counter() - t1
        plan = C775.plan_for(model, setting, targets, methods, metas, None)
        t2 = time.perf_counter()
        report = bnn.audit(results, plan=plan)
        audit_s = time.perf_counter() - t2
        t3 = time.perf_counter()
        AU.verify_report(report.to_dict(), results, plan)
        verify_s = time.perf_counter() - t3
        t4 = time.perf_counter()
        text = report.to_json()
        json_s = time.perf_counter() - t4
        site_t: dict[str, Any] = {
            "samples": len(samples),
            "results": report.evidence.results,
            "evidence_seconds": evidence_s,
            "audit_seconds": audit_s,
            "verify_seconds": verify_s,
            "report_to_json_seconds": json_s,
            "report_bytes": len(text.encode()),
        }
        if a.save_evidence:
            ev = out / f"evidence_{setting.site}"
            t5 = time.perf_counter()
            AU.save_evidence(results, ev)
            site_t["save_evidence_seconds"] = time.perf_counter() - t5
            site_t["evidence_bytes"] = _du(ev)
            (out / f"plan_{setting.site}.json").write_text(bnn.schema.to_json(plan))
            report.save(out / f"report_{setting.site}.json")
            t6 = time.perf_counter()
            paths = AU.load_evidence(ev)
            re = bnn.audit(paths, plan=plan)
            site_t["load_and_reaudit_seconds"] = time.perf_counter() - t6
            site_t["reaudit_from_disk_identical"] = re.to_json() == text
            t7 = time.perf_counter()
            AU.verify_report(AU.load_report(out / f"report_{setting.site}.json"), paths, plan)
            site_t["verify_saved_report_from_disk_seconds"] = time.perf_counter() - t7
        site_t["peak_rss_mb_so_far"] = _rss_mb()
        timing["settings"][setting.site] = site_t
        payload["settings"][setting.site] = {
            "n_samples": len(samples),
            "n_results": report.evidence.results,
            "results_excluded": report.evidence.results_excluded,
            "plan_id": plan.id,
            "run_seconds": evidence_s,
            "audit_seconds": audit_s,
            "verify_report": "passed",
            "audit": C75.summarise(report, metas),
            "samples": metas,
        }
        tag = {"model": a.model, "site": setting.site, "seed": seed_tag}
        all_rows += config_rows(report, metas, tag)
        del results, report
    (out / "summary.json").write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    (out / "timing.json").write_text(json.dumps(timing, indent=1, sort_keys=True))
    if all_rows:
        with (out / "configs.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
