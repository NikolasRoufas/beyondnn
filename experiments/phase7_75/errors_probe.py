# ruff: noqa: E501, E402, E731
"""Pre-Phase-8 error-message probe (docs/PRE_PHASE8_HARDENING_REPORT.md §7). Run from the repo root."""

import importlib.util
import pathlib
import tempfile

import torch

spec = importlib.util.spec_from_file_location("G", "tests/test_golden_workflow.py")
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
import beyondnn as bnn

A, F, C, AU, iv = bnn.attribution, bnn.faithfulness, bnn.concepts, bnn.audits, bnn.interventions
from beyondnn.schema import from_dict, to_dict

d = pathlib.Path(tempfile.mkdtemp())
G.build(d)
model = G.Golden().eval()
x = G._data()
s0, s1 = x[200:201], x[201:202]
m = iv.metrics.margin([0, 1])
attr0 = bnn.attribute(model, s0, target=m, method=A.gradient())


def show(label, fn):
    try:
        fn()
        print(f"[{label}] NO ERROR")
    except Exception as e:
        print(f"[{label}] {type(e).__name__}: {e}")


test = lambda **kw: F.comprehensiveness(target=m, min_drop=0.1, statement="s", **kw)
show(
    "eligibility: ineligible declared unit",
    lambda: F.units("hidden", (0,), n_units=24, eligible=(1, 2), eligibility="content"),
)
show("eligibility: mask without name", lambda: F.top_k(attr0, k=1, eligible=(1, 2)))
show(
    "wrong sample",
    lambda: F.run(
        model,
        s1,
        test=test(replacement=F.zero()),
        selection=F.top_k(attr0, k=1),
        attributions=[attr0],
    ),
)
plan = bnn.schema.from_json((d / "plan.json").read_text())
paths = AU.load_evidence(d / "evidence")
other = G.Golden()
with torch.no_grad():
    other.out.weight[1, 2] = 0.5
show("wrong checkpoint (audit model=)", lambda: bnn.audit(paths, plan=plan, model=other))
show("missing replacement", lambda: F.comprehensiveness(target=m, min_drop=0.1, statement="s"))
show("unsupported evidence type", lambda: bnn.audit([123], plan=plan))
show("evidence is a string", lambda: bnn.audit("evidence/", plan=plan))
env = to_dict(plan)
env["record_version"] = 99
show("newer serialized version", lambda: from_dict(env))
env2 = to_dict(plan)
env2["id"] = "audit_plan:" + "0" * 32
show("tampered record id", lambda: from_dict(env2))
# corrupt saved evidence: flip a byte in a record file of one saved trace
t0 = sorted(paths)[0]
files = sorted(p for p in t0.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl"))
f = files[0]
txt = f.read_text()
f.write_text(txt.replace('"', "'", 1) if '"' in txt else txt + "x")
show(
    f"corrupt saved evidence ({f.name})",
    lambda: bnn.audit(AU.load_evidence(d / "evidence"), plan=plan),
)
show(
    "load_validation from unrelated traces",
    lambda: C.load_validation(AU.load_evidence(d / "evidence")[:2]),
)
# scope: plan whose samples do not include the evidence's samples
plan2 = AU.plan(
    name="scope",
    checkpoint=plan.checkpoint,
    declared_model=None,
    samples=["sha256:" + "a" * 64],
    datasets=[],
    claims=[],
    requirements=[],
    concepts=[],
    counterexamples=plan.counterexamples,
    naive_auroc=None,
)
show(
    "claim targets not covering plan samples",
    lambda: AU.plan(
        name="bad",
        checkpoint=plan.checkpoint,
        declared_model=None,
        samples=["sha256:" + "a" * 64],
        datasets=[],
        claims=[plan.claims[0]],
        requirements=list(plan.requirements),
        concepts=[],
        counterexamples=plan.counterexamples,
        naive_auroc=None,
    ),
)
