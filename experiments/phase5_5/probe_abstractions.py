"""H8 (plan §2, §15): try each model with the unchanged Phase-5 public API and record
exactly what works and what is refused, before any framework change."""

from __future__ import annotations

import sys
import traceback
from typing import Any

import common
import torch

import beyondnn as bnn

A, F, iv = bnn.attribution, bnn.faithfulness, bnn.interventions


def attempt(label: str, fn: Any, log: list[dict[str, Any]]) -> Any:
    try:
        out = fn()
        log.append({"step": label, "ok": True})
        return out
    except Exception as exc:  # recorded, not hidden
        log.append(
            {
                "step": label,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "where": traceback.extract_tb(exc.__traceback__)[-1].name,
            }
        )
        return None


def main() -> None:
    log: list[dict[str, Any]] = []
    # A: vector input
    mA, dA, _ = common.train("A")
    x = dA.x[dA.test[:1]]
    metric, *_ = common.margin_target(mA(x))
    attr = attempt(
        "A input IG",
        lambda: A.attribute(
            mA,
            x,
            target=metric,
            method=A.integrated_gradients(baseline=A.zero_baseline(), n_steps=8),
        ),
        log,
    )
    attempt(
        "A input top_k + comprehensiveness",
        lambda: F.run(
            mA,
            x,
            test=F.comprehensiveness(target=metric, min_drop=0.1, statement="s"),
            selection=F.top_k(attr, k=3),
        ),
        log,
    )
    hid = attempt(
        "A hidden gradient",
        lambda: A.attribute(mA, x, target=metric, method=A.gradient(), at=A.layer("net.1")),
        log,
    )
    attempt(
        "A hidden top_k + comprehensiveness",
        lambda: F.run(
            mA,
            x,
            test=F.comprehensiveness(target=metric, min_drop=0.1, statement="s"),
            selection=F.top_k(hid, k=3),
        ),
        log,
    )
    # B: image input and channels
    mB, dB, _ = common.train("B")
    xb = dB.x[dB.test[:1]]
    metric_b, *_ = common.margin_target(mB(xb))
    ab = attempt(
        "B input IG",
        lambda: A.attribute(
            mB,
            xb,
            target=metric_b,
            method=A.integrated_gradients(baseline=A.zero_baseline(), n_steps=8),
        ),
        log,
    )
    attempt("B pixels top_k", lambda: F.top_k(ab, k=4), log)
    attempt(
        "B pixels declared units(n_units=64)",
        lambda: F.run(
            mB,
            xb,
            test=F.comprehensiveness(target=metric_b, min_drop=0.1, statement="s"),
            selection=F.units(A.input(), (0, 1), n_units=64),
        ),
        log,
    )
    ac = attempt(
        "B channel gradient",
        lambda: A.attribute(mB, xb, target=metric_b, method=A.gradient(), at=A.layer("relu2")),
        log,
    )
    attempt("B channels top_k", lambda: F.top_k(ac, k=2), log)
    attempt(
        "B channels declared units(n_units=16)",
        lambda: F.run(
            mB,
            xb,
            test=F.comprehensiveness(target=metric_b, min_drop=0.1, statement="s"),
            selection=F.units("relu2", (0,), n_units=16),
        ),
        log,
    )
    # C: transformer tokens
    mC, tok, _ = common.bert()
    enc = tok("a charming and often affecting journey .", return_tensors="pt")
    kwargs = {"attention_mask": enc["attention_mask"], "token_type_ids": enc["token_type_ids"]}
    ids = enc["input_ids"]
    attempt(
        "C trace",
        lambda: bnn.trace(mC, ids, model_kwargs=kwargs, sites=["bert.embeddings.word_embeddings"]),
        log,
    )
    out = mC(ids, **kwargs)
    metric_c, *_ = common.margin_target(out.logits, path='["logits"]')
    attempt(
        "C input gradient on ids",
        lambda: A.attribute(mC, ids, target=metric_c, method=A.gradient(), model_kwargs=kwargs),
        log,
    )
    ae = attempt(
        "C word-embedding IxG",
        lambda: A.attribute(
            mC,
            ids,
            target=metric_c,
            method=A.input_x_gradient(),
            at=A.layer("bert.embeddings.word_embeddings"),
            model_kwargs=kwargs,
        ),
        log,
    )
    attempt("C tokens top_k", lambda: F.top_k(ae, k=2), log)
    attempt(
        "C input-id masking (zero_input units = positions)",
        lambda: iv.intervene(
            mC,
            ids,
            intervention=iv.constant_input(torch.full_like(ids, tok.mask_token_id), units=(1,)),
            metric=metric_c,
            model_kwargs=kwargs,
        ),
        log,
    )
    attempt(
        "C token positions declared units(n=T) on word embeddings",
        lambda: F.run(
            mC,
            ids,
            test=F.comprehensiveness(target=metric_c, min_drop=0.1, statement="s"),
            selection=F.units("bert.embeddings.word_embeddings", (1,), n_units=ids.shape[1]),
            model_kwargs=kwargs,
        ),
        log,
    )
    for row in log:
        print(
            ("OK   " if row["ok"] else "FAIL ")
            + row["step"]
            + ("" if row["ok"] else f"  -> {row['error'][:160]}")
        )
    # The committed abstraction_probe.json was produced with the unchanged Phase-5 API
    # (plan §15); a rerun after ADR-034 is saved under another name.
    name = sys.argv[1] if len(sys.argv) > 1 else "abstraction_probe"
    common.save(name, {"environment": common.environment(), "log": log})


if __name__ == "__main__":
    main()
