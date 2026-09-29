"""Phase 7.75 analysis helpers and frozen-plan consistency (request §31/§32)."""

from __future__ import annotations

import importlib.util
import inspect
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import beyondnn as bnn

ROOT = Path(__file__).resolve().parents[1]
C, iv = bnn.concepts, bnn.interventions


def _load(name: str) -> Any:
    path = ROOT / "experiments" / "phase7_75" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AN = _load("analysis775")


def _row(truth: str, standing: str, kind: str = "head", **configs: str) -> dict[str, Any]:
    return {
        "program": "p",
        "component": "c",
        "kind": kind,
        "truth": truth,
        "standing": standing,
        "configs": configs,
    }


def test_same_operation_ground_truth_is_not_independent() -> None:
    assert AN.truth_is_independent("program_source", "resample") is True
    assert AN.truth_is_independent("resample", "Resample") is False
    with pytest.raises(ValueError, match="declare both"):
        AN.truth_is_independent("", "resample")


def test_truth_table_keeps_every_cell() -> None:
    rows = [
        _row("known_true", "supported"),
        _row("known_true", "contradicted"),
        _row("known_true", "inconclusive"),
        _row("known_false", "supported"),
        _row("known_false", "unsupported"),
        _row("known_false", "assumption_sensitive"),
        _row("empty", "inconclusive"),
        _row("known_true", "supported", kind="selection"),
    ]
    t = AN.truth_table(rows)
    assert t["known_true"] == {
        "supported": 1,
        "contradicted": 1,
        "unsupported": 0,
        "inconclusive": 1,
        "other": 0,
    }
    assert t["known_false"]["supported"] == 1  # a false positive is never hidden
    assert t["known_false"]["other"] == 1
    assert t["empty"]["inconclusive"] == 1
    assert AN.truth_table(rows, kinds=("selection",))["known_true"]["supported"] == 1
    with pytest.raises(ValueError, match="unknown truth"):
        AN.truth_table([_row("probably", "supported")])


def test_component_support_and_configuration_rates() -> None:
    rows = [
        _row("known_false", "contradicted", zero="supports"),
        _row("known_false", "contradicted", zero="contradicts"),
    ]
    assert AN.component_support(rows)["p/c"] == {"truth": "known_false", "supported": 0, "n": 2}
    assert AN.configuration_rates(rows, "zero", truth="known_false") == (1, 2)


def test_concept_threshold_uses_train_values_only() -> None:
    assert AN.rel_min_change([0.0, 2.0]) == pytest.approx(0.2 * 2**0.5)
    params = list(inspect.signature(AN.rel_min_change).parameters)
    assert params == ["train_target_values", "c"]  # no held-out outcome can enter it
    assert AN.C_REL == 0.2  # ADR-055, frozen


def test_shortcut_stratification_removes_no_sample() -> None:
    out = AN.stratify(
        ["supported", "contradicted", "supported"],
        [True, False, False],
        labels=("empty_premise_predictable", "needs_premise"),
    )
    assert sum(sum(v.values()) for v in out.values()) == 3
    with pytest.raises(ValueError, match="one stratum"):
        AN.stratify(["supported"], [], labels=("a", "b"))


class _Hidden(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


class _Cal(nn.Module):
    """y = w * h0 + h2 (h = x): 'x0 > 0' used; 'x1 > 0' decodable, never read."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = _Hidden()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        return (2.0 * h[:, 0] + h[:, 2]).unsqueeze(1)


@pytest.mark.parametrize(
    ("name", "neuron", "labels_of", "expected"),
    [
        ("known_positive", 0, lambda x: x[:, 0] > 0, "validated_concept"),
        ("decodable_unused", 1, lambda x: x[:, 1] > 0, "proposed_concept"),
        ("known_negative", 0, lambda x: x[:, 3] > 0, "proposed_concept"),
    ],
)
def test_known_concept_positive_unused_and_negative(
    name: str, neuron: int, labels_of: Any, expected: str
) -> None:
    n = 240
    splits = ["train"] * 120 + ["val"] * 40 + ["test"] * 80
    x = torch.randn(n, 16, generator=torch.Generator().manual_seed(7751))
    model = _Cal().eval()
    with torch.no_grad():
        train_y = model(x[:120])[:, 0].tolist()
    mc = AN.rel_min_change(train_y)
    data = C.dataset(
        list(x.split(1)),
        labels_of(x).long().tolist(),
        splits,
        name=name,
        label_source="constructed",
    )
    concept = C.propose(
        C.neuron("hidden", neuron), label=name, definition=name, label_source="dataset"
    )
    enc = C.encoding_test(
        model,
        concept,
        data,
        controls=[C.random_neurons(40, seed=1), C.label_permutation(100, seed=2)],
        criteria=C.encoding_criteria(min_fraction_below=0.95),
    )
    use = C.use_test(
        model,
        concept,
        data,
        target=iv.metrics.select([0, 0]),
        relation="decreases",
        intervention=C.remove(C.zero()),
        controls=[C.random_neurons(40, seed=3)],
        criteria=C.use_criteria(min_change=mc, min_fraction_beyond_controls=0.95),
    )
    assert C.validate(concept, encoding=enc, use=[use]).semantic_status.value == expected


def test_frozen_td_split_and_plan_agree() -> None:
    plan = (ROOT / "docs" / "PHASE_7_75_PLAN.md").read_text()
    src = (ROOT / "experiments" / "phase7_75" / "td_programs.py").read_text()
    dev = re.findall(r'"(dev_[a-z0-9_]+)": Program', src)
    held = re.findall(r'"(ho_[a-z0-9_]+)": Program', src)
    assert dev == ["dev_copy_1_2", "dev_map_decoy"]
    assert len(held) == 8
    for h in held:
        assert f"`{h}`" in plan  # every held-out program was declared in the frozen plan
    concepts = (ROOT / "experiments" / "phase7_75" / "td_concepts.py").read_text()
    assert '"dev": {"cdev_a_b"' in concepts
    assert "C_REL = 0.2" in concepts


def test_td_roles_are_the_frozen_e1_roles() -> None:
    """TD uses exactly the Phase-7.5 E1 roles: zero is a STRESS_TEST, resample is PRIMARY."""

    def roles(path: Path) -> str:
        src = path.read_text()
        m = re.search(r"ROLES = \[(.*?)\n\]", src, re.S)
        assert m is not None
        return re.sub(r"\s+", "", m.group(1))

    td = roles(ROOT / "experiments" / "phase7_75" / "td_bench.py")
    e1 = roles(ROOT / "experiments" / "phase7_5" / "external_interpbench.py")
    assert td == e1
    assert '("replacement","zero","stress_test")' in td
