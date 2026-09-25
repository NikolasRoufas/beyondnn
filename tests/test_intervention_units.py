"""Phase 5 (ADR-032): unit-level and model-input interventions and comparison families,
as extensions of the Phase-2 engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

import beyondnn as bnn
import beyondnn.interventions as iv
from beyondnn._testing.causal_models import Additive
from beyondnn._testing.faithfulness_models import ProbeReadable, Weighted8
from beyondnn.core.samples import sample_id
from beyondnn.schema import (
    EvidenceStatus,
    ExecutionMode,
    InterventionOperation,
    InterventionRecord,
    SchemaError,
    Site,
    SiteIO,
)
from beyondnn.schema.codec import _expected_id

SEL = iv.metrics.select([0, 0])
ONES = torch.ones(1, 8)


def run(model: Any, *inputs: Any, **kw: Any) -> iv.InterventionResult:
    kw.setdefault("metric", SEL)
    return iv.intervene(model.eval(), *inputs, **kw)


# ------------------------------------------------------------------ internal unit interventions


@pytest.mark.parametrize(
    ("units", "retain", "effect"),
    [((1,), False, 0.0), ((0,), False, -6.0), ((0,), True, 0.0), ((1,), True, -6.0)],
)
def test_internal_unit_removal_and_retention(
    units: tuple[int, ...], retain: bool, effect: float
) -> None:
    r = run(
        ProbeReadable(),
        torch.tensor([[3.0, 5.0]]),
        intervention=iv.zero("hidden", units=units, retain=retain),
    )
    assert r.value == effect
    assert r.intervention.units == units
    assert r.intervention.retain is retain
    assert r.effect.status is EvidenceStatus.INTERVENTIONAL


def test_the_intervened_activation_is_exactly_the_requested_perturbation() -> None:
    r = run(
        ProbeReadable(),
        torch.tensor([[3.0, 5.0]]),
        retention="cpu",
        intervention=iv.constant("hidden", torch.tensor([[7.0, 9.0]]), units=(1,)),
    )
    out = r.trace.activation("hidden", pass_index=r.intervention_output.pass_index)
    assert torch.equal(r.trace.tensor(out), torch.tensor([[3.0, 9.0]]))  # unit 0 untouched
    assert r.value == 0.0


# ------------------------------------------------------------------ model-input interventions


def test_input_removal_and_retention_on_weighted8() -> None:
    assert run(Weighted8(), ONES, intervention=iv.zero_input(units=(0, 1))).value == -12.0
    assert (
        run(Weighted8(), ONES, intervention=iv.zero_input(units=(0, 1), retain=True)).value == -3.0
    )
    assert run(Weighted8(), ONES, intervention=iv.zero_input()).value == -15.0
    values = torch.full((1, 8), 2.0)
    assert run(Weighted8(), ONES, intervention=iv.constant_input(values, units=(0,))).value == 8.0


def test_input_interventions_are_observed_exactly_and_recorded_as_interventions() -> None:
    r = run(Weighted8(), ONES, intervention=iv.zero_input(units=(0, 1)))
    expected = ONES.clone()
    expected[0, :2] = 0
    assert r.intervention.on_input
    assert r.intervention.site == Site(module="", io=SiteIO.INPUT, output_path="args[0]")
    perturbed = next(i for i in r.trace.inputs if i.pass_index == r.intervention_output.pass_index)
    assert perturbed.sample_id == sample_id(expected)  # the perturbation really happened
    assert r.effect.estimand.sample_id == sample_id(ONES)  # the effect is about x
    prov = r.trace.origin(perturbed)
    assert prov.execution.mode is ExecutionMode.INTERVENTION
    assert prov.execution.intervention_id == r.intervention.id
    assert "ZERO_ABLATION_MAY_BE_OOD" in {lim.code for lim in r.limitations}


def test_invalid_unit_interventions_are_refused() -> None:
    with pytest.raises(iv.InterventionError, match="out of range"):
        run(Weighted8(), ONES, intervention=iv.zero_input(units=(8,)))
    with pytest.raises(iv.InterventionError, match="out of range"):
        run(ProbeReadable(), torch.tensor([[3.0, 5.0]]), intervention=iv.zero("hidden", units=(2,)))
    with pytest.raises(ValueError, match="retain"):
        iv.zero("hidden", retain=True)
    with pytest.raises(ValueError, match="unique"):
        iv.zero_input(units=(1, 1))
    with pytest.raises(ValueError, match="non-empty"):
        iv.zero_input(units=())
    with pytest.raises(ValueError, match="zero_input"):
        iv.Intervention("", InterventionOperation.ZERO)
    with pytest.raises(iv.InterventionError, match="no broadcasting"):
        run(Weighted8(), ONES, intervention=iv.constant_input(torch.zeros(8), units=(0,)))


def test_intervention_record_schema_rules() -> None:
    root = Site(module="", io=SiteIO.INPUT, output_path="args[0]")
    InterventionRecord(site=root, operation=InterventionOperation.ZERO, units=(0,))
    with pytest.raises(SchemaError, match="sorted"):
        InterventionRecord(site=root, operation=InterventionOperation.ZERO, units=(1, 0))
    with pytest.raises(SchemaError, match="retain requires units"):
        InterventionRecord(site=root, operation=InterventionOperation.ZERO, retain=True)
    with pytest.raises(SchemaError, match="positional input"):
        InterventionRecord(
            site=Site(module="", io=SiteIO.INPUT, output_path='kwargs["m"]'),
            operation=InterventionOperation.ZERO,
        )
    with pytest.raises(SchemaError, match="ZERO or CONSTANT"):
        InterventionRecord(site=root, operation=InterventionOperation.PATCH)


def test_v1_intervention_payloads_migrate_without_invented_units(tmp_path: Path) -> None:
    r = run(Additive(), torch.tensor([[3.0, 5.0]]), intervention=iv.zero("a"))
    r.trace.save(tmp_path / "t")
    document = json.loads((tmp_path / "t" / "trace.json").read_text())
    renamed: dict[str, str] = {}
    for env in document["records"]:
        text = json.dumps(env["data"])
        for new, old in renamed.items():
            text = text.replace(new, old)
        env["data"] = json.loads(text)
        if env["kind"] == "intervention":
            env["record_version"] = 1
            env["data"].pop("units")
            env["data"].pop("retain")
            env["data"].pop("unit_axes")  # added in v3 (ADR-034)
        new_id = env["id"]
        env["id"] = _expected_id(env["kind"], env["record_version"], env["data"])
        renamed[new_id] = env["id"]
    (tmp_path / "t" / "trace.json").write_text(json.dumps(document))
    loaded = iv.InterventionResult.from_trace(bnn.load_trace(tmp_path / "t"))
    assert loaded.intervention.units is None
    assert loaded.intervention.retain is False
    assert loaded.value == -3.0


# ------------------------------------------------------------------ comparison families


def test_a_family_shares_one_baseline_pass_per_input() -> None:
    specs = [iv.zero_input(units=(u,)) for u in range(4)]
    family = iv.compare_family(Weighted8().eval(), [((ONES,), {}, specs)], SEL)
    assert family.trace.passes == 1 + 4
    assert [e.effect for e in family.effects[0]] == [-8.0, -4.0, -2.0, -1.0]
    assert family.baseline_values == (15.0,)
    assert all(e.estimand.sample_id == sample_id(ONES) for e in family.effects[0])
    assert [r.units for r in family.records[0]] == [(0,), (1,), (2,), (3,)]


def test_a_family_over_several_inputs_stays_per_sample() -> None:
    a, b = torch.tensor([[3.0, 5.0]]), torch.tensor([[1.0, 2.0]])
    family = iv.compare_family(
        Additive().eval(),
        [((a,), {}, [iv.zero("a")]), ((b,), {}, [iv.zero("a"), iv.zero("b")])],
        SEL,
    )
    assert family.samples == (sample_id(a), sample_id(b))
    assert [[e.effect for e in row] for row in family.effects] == [[-3.0], [-1.0, -2.0]]
    assert family.trace.passes == 2 + 3


def test_a_family_extension_adds_records_to_the_same_trace() -> None:
    seen: list[int] = []

    def extend(trace: Any, family: iv.ComparisonFamily) -> None:
        seen.append(len(family.effects[0]))
        from beyondnn.schema import TraceLimitation

        trace._add(TraceLimitation(code="NO_CLAIMS_TESTED", applies_to=(family.effects[0][0].id,)))

    family = iv.compare_family(
        Weighted8().eval(), [((ONES,), {}, [iv.zero_input()])], SEL, extend=extend
    )
    assert seen == [1]
    assert any(lim.code == "NO_CLAIMS_TESTED" for lim in family.trace.limitations)


def test_family_refusals_are_the_phase_2_refusals() -> None:
    with pytest.raises(iv.StatefulComparisonError, match="training"):
        iv.compare_family(Weighted8().train(), [((ONES,), {}, [iv.zero_input()])], SEL)
    with pytest.raises(iv.InterventionError, match="at least one group"):
        iv.compare_family(Weighted8().eval(), [], SEL)
    family = iv.compare_family(Weighted8().eval(), [((ONES,), {}, [])], SEL)
    assert family.effects == ((),)
    assert family.trace.passes == 1
