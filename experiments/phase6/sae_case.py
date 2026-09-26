"""Phase 6 SAE case study (request §45; plan §16).

A small sparse autoencoder is trained locally (no downloads) on the MLP's hidden
activations (`net.1`, 64 -> 128 latents) using only the concept TRAIN split of the K1
(size) dataset. A latent is chosen by a recorded search (highest sign-free train AUROC
for K1 among all latents). A rule-based labeller (a stand-in for an LLM; its output is
recorded as GENERATED) proposes a label. The latent is then tested like any feature.

Usage (experiment env): python experiments/phase6/sae_case.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase5_5"))
sys.path.insert(0, str(HERE))
import realistic as R  # noqa: E402

import beyondnn as bnn  # noqa: E402
from beyondnn.concepts._core import auroc  # noqa: E402
from beyondnn.concepts.verify import verify_validation  # noqa: E402

C, iv = bnn.concepts, bnn.interventions
D_SAE, L1, STEPS, LR = 128, 3e-3, 3000, 1e-3


def train_sae(acts: torch.Tensor) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    d = acts.shape[1]
    w_enc = torch.nn.Parameter(torch.randn(d, D_SAE) * 0.1)
    w_dec = torch.nn.Parameter(torch.randn(D_SAE, d) * 0.1)
    b_enc = torch.nn.Parameter(torch.zeros(D_SAE))
    b_dec = torch.nn.Parameter(acts.mean(0).clone())
    opt = torch.optim.Adam([w_enc, w_dec, b_enc, b_dec], lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        z = torch.relu((acts - b_dec) @ w_enc + b_enc)
        recon = z @ w_dec + b_dec
        loss = ((recon - acts) ** 2).sum(1).mean() + L1 * (z.abs() * w_dec.norm(dim=1)).sum(
            1
        ).mean()
        loss.backward()
        opt.step()
    return {
        k: v.detach().clone()
        for k, v in (("encoder", w_enc), ("decoder", w_dec), ("b_enc", b_enc), ("b_dec", b_dec))
    }


def sae_stats(sae: dict[str, torch.Tensor], acts: torch.Tensor) -> dict[str, float]:
    z = torch.relu((acts - sae["b_dec"]) @ sae["encoder"] + sae["b_enc"])
    recon = z @ sae["decoder"] + sae["b_dec"]
    mse = float(((recon - acts) ** 2).sum(1).mean())
    var = float(((acts - acts.mean(0)) ** 2).sum(1).mean())
    return {
        "mse": mse,
        "explained_variance": 1 - mse / var,
        "l0": float((z > 0).float().sum(1).mean()),
        "dead_latents": int(((z > 0).sum(0) == 0).sum()),
    }


def main() -> None:
    torch.set_num_threads(1)
    model, runs, _info = R.model_a()
    ds, _name, definition, site, _axis, _pooling, target, _desc = runs[0]  # K1 size
    from sklearn.datasets import load_breast_cancer

    from beyondnn.concepts.features import record_site, site_tensors

    raw = load_breast_cancer()
    where = bnn.schema.Site(module=site)

    def activations(split: str) -> tuple[list[int], torch.Tensor]:
        idx = list(ds.record.indices(split))
        tr = record_site(model, ds, idx, where)
        return idx, torch.cat([t for _, t in site_tensors(tr, ds, idx, where, 0)]).float()

    train_idx, train_acts = activations("train")
    _, val_acts = activations("val")
    sae = train_sae(train_acts)
    checkpoint = (
        "sha256:"
        + __import__("hashlib")
        .sha256(
            b"".join(sae[k].numpy().tobytes() for k in ("encoder", "decoder", "b_enc", "b_dec"))
        )
        .hexdigest()
    )
    z_train = torch.relu((train_acts - sae["b_dec"]) @ sae["encoder"] + sae["b_enc"])
    labels = [ds.record.labels[i] for i in train_idx]
    scores = []
    for j in range(D_SAE):
        a = auroc(z_train[:, j].tolist(), labels)
        scores.append((max(a, 1 - a), -j, 1 if a >= 1 - a else -1))
    best, neg_j, sign = max(scores)
    latent = -neg_j
    stats = {"train": sae_stats(sae, train_acts), "val": sae_stats(sae, val_acts)}
    feature = C.sae_feature(
        model,
        site,
        latent=latent,
        checkpoint=checkpoint,
        reconstruction={
            "val_mse": stats["val"]["mse"],
            "val_explained_variance": stats["val"]["explained_variance"],
        },
        search={
            "method": "train_auroc",
            "candidate_count": D_SAE,
            "criterion": "max sign-free train AUROC",
            "train_auroc": best,
            "sign": sign,
        },
        data=ds,
        **sae,
    )
    # stand-in labeller (recorded as GENERATED): the raw input feature most correlated
    # with the latent on the train split
    xs = torch.tensor(raw.data[train_idx], dtype=torch.float64)
    zj = z_train[:, latent].double()
    corr = [
        float(torch.corrcoef(torch.stack([zj, xs[:, k]]))[0, 1]) if float(zj.std()) > 0 else 0.0
        for k in range(xs.shape[1])
    ]
    k_best = max(
        range(len(corr)), key=lambda k: (abs(corr[k]) if not math.isnan(corr[k]) else -1, -k)
    )
    text = (
        f"responds to high {raw.feature_names[k_best]}"
        if corr[k_best] > 0
        else f"responds to low {raw.feature_names[k_best]}"
    )
    gen = C.generated_label(model, feature, text, generator="rule-labeler", revision="phase6-v1")
    concept = C.propose(
        feature, definition=f"{definition} (tested against the K1 dataset)", generated=gen
    )
    enc = C.encoding_test(
        model,
        concept,
        ds,
        controls=R.enc_controls("covariance"),
        criteria=C.encoding_criteria(min_fraction_below=0.95),
    )
    rows: dict[str, Any] = {}
    for rname, ref in (
        ("zero", C.zero()),
        ("train_mean", C.reference(R.mean_reference(model, ds, feature), name="train_mean")),
    ):
        use = C.use_test(
            model,
            concept,
            ds,
            target=target,
            relation="decreases",
            intervention=C.remove(ref),
            controls=[C.random_directions(50, seed=13, distribution="covariance")],
            criteria=C.use_criteria(min_change=0.1, min_fraction_beyond_controls=0.95),
        )
        val = C.validate(concept, encoding=enc, use=[use])
        verify_validation(val)
        rows[rname] = {
            "use": use.outcome.value,
            "mean_effect": use.mean_effect,
            "superiority": use.result.statistics["superiority"],
            "status": val.semantic_status.value,
            "unmet": list(val.unmet),
        }
        print(val.describe())
    out = {
        "sae": {
            "d_sae": D_SAE,
            "l1": L1,
            "steps": STEPS,
            "lr": LR,
            "trained_on": "K1 concept train split activations of net.1",
            "checkpoint": checkpoint,
            "stats": stats,
        },
        "latent": latent,
        "search_train_auroc": best,
        "sign": sign,
        "generated_label": {"text": text, "status": gen.record.status.value, "corr": corr[k_best]},
        "encoding": {
            "outcome": enc.outcome.value,
            "auroc": enc.auroc,
            "controls": [
                {k: c[k] for k in ("kind", "superiority", "median", "q95")}
                for c in enc.result.statistics["controls"]
            ],  # type: ignore[union-attr]
            "fp_rate": enc.counterexamples.measurements["false_positive_rate"],
            "fn_rate": enc.counterexamples.measurements["false_negative_rate"],
        },
        "use": rows,
        "limitations": sorted({lim.code for lim in enc.trace.limitations}),
    }
    (R.RESULTS / "sae_case.json").write_text(json.dumps(out, indent=1, default=str))
    print(
        json.dumps(
            {k: out[k] for k in ("latent", "search_train_auroc", "generated_label")}, default=str
        )
    )


if __name__ == "__main__":
    main()
