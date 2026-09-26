"""Pre-registered Phase-7 falsification scenarios A-N (docs/PHASE_7_PLAN.md §29.3).

Internal and unstable. Every scenario's evidence is produced by real BeyondNN runs on
small fixed-weight models whose ground truth is known by construction; nothing is
hand-written. Each builder returns the evidence, the plan, and the pre-registered
expectation (standing + finding codes). These are tests *of the audit*, not findings
about models.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

import beyondnn as bnn
from beyondnn.core.samples import sample_id
from beyondnn.schema import InterventionOperation, Relation, Site, Subject

__all__ = ["SCENARIOS", "Product", "Scenario", "WeightedSum", "build"]

A, F, iv = bnn.attribution, bnn.faithfulness, bnn.interventions


class _Pass(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 1.0


class WeightedSum(nn.Module):
    """``hidden = x`` (a named site with ``len(weights)`` units); ``y = hidden @ w``."""

    def __init__(self, weights: list[float]) -> None:
        super().__init__()
        self.hidden = _Pass()
        self.register_buffer("w", torch.tensor(weights))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w
        assert isinstance(w, torch.Tensor)
        out: torch.Tensor = (self.hidden(x) * w).sum(dim=1, keepdim=True)
        return out


class Product(nn.Module):
    """``y = h0 * h1`` (both needed; neither alone suffices)."""

    def __init__(self, n: int = 8) -> None:
        super().__init__()
        self.hidden = _Pass()
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.hidden(x)
        return (h[:, 0] * h[:, 1]).unsqueeze(1)


@dataclass
class Scenario:
    name: str
    evidence: list[Any]
    plan: Any
    claim: str | None  # the plan claim the expectation is about (None: concept)
    standing: str
    codes: tuple[str, ...]
    extra: dict[str, Any] = field(default_factory=dict)


TARGET = iv.metrics.select([0, 0])
NO_CAPS = None


def _rule(fp: float | None = None) -> Any:
    return bnn.audits.counterexample_rule(
        max_counterexample_fraction=None, max_false_positive_rate=fp, max_false_negative_rate=None
    )


def _plan(
    model: nn.Module, samples: list[str], claims: list[Any], requirements: list[Any], **kw: Any
) -> Any:
    AU = bnn.audits
    return AU.plan(
        name=kw.pop("name", "scenario"),
        checkpoint=kw.pop("checkpoint", AU.checkpoint_of(model)),
        declared_model=None,
        samples=samples,
        datasets=kw.pop("datasets", []),
        claims=claims,
        requirements=requirements,
        concepts=kw.pop("concepts", []),
        counterexamples=kw.pop("counterexamples", _rule()),
        naive_auroc=kw.pop("naive_auroc", None),
    )


def _comp_req(controls: bool) -> Any:
    return bnn.audits.requirement(
        "comprehensiveness", policy=F.COMPREHENSIVENESS_POLICY, controls=controls
    )


def _unit_claim(name: str, relation: str, requirement: str, **kw: Any) -> Any:
    return bnn.audits.claim(
        name,
        statement=f"unit 0 of hidden is {relation} for y",
        relation=relation,
        target=TARGET,
        scope="instance",
        requirement=requirement,
        subject=Subject(site=Site(module="hidden"), units=(0,)),
        **kw,
    )


def _comp(
    model: nn.Module,
    x: torch.Tensor,
    replacement: Any,
    *,
    controls: Any = None,
    min_drop: float = 5.0,
    min_fraction_below: float | None = None,
    n_units: int = 32,
) -> Any:
    kw: dict[str, Any] = {}
    if min_fraction_below is not None:
        kw["min_fraction_below"] = min_fraction_below
    test = F.comprehensiveness(
        target=TARGET,
        min_drop=min_drop,
        statement="unit 0 of hidden is necessary for y",
        controls=controls,
        replacement=replacement,
        **kw,
    )
    selection = F.units(A.layer("hidden"), (0,), n_units=n_units)
    return F.run(model, x, test=test, selection=selection)


def _one_unit(n: int = 32) -> tuple[nn.Module, torch.Tensor]:
    model = WeightedSum([5.0] + [0.01] * (n - 1)).eval()
    x = torch.ones(1, n)
    x[0, 0] = 2.0
    return model, x


# ------------------------------------------------------------------ scenarios


def scenario_a() -> Scenario:
    """A: fully supported (3 replacements with controls + a site-level intervention)."""
    model, x = _one_unit()
    reps = [
        F.zero(),
        F.replacement(torch.full((1, 32), 0.1)),
        F.replacement(torch.full((1, 32), -0.5)),
    ]
    ctl = F.controls(20, seed=3)
    evidence: list[Any] = [_comp(model, x, r, controls=ctl, min_fraction_below=0.9) for r in reps]
    site_claim = iv.make_claim(
        iv.zero("hidden"), TARGET, Relation.NECESSARY_FOR, x, statement="hidden is necessary"
    )
    evidence.append(
        bnn.intervene(
            model,
            x,
            intervention=iv.zero("hidden"),
            metric=TARGET,
            claims=[
                (
                    site_claim,
                    iv.threshold_spec(operation=InterventionOperation.ZERO, min_effect=5.0),
                )
            ],
        )
    )
    AU = bnn.audits
    plan = _plan(
        model,
        [sample_id(x)],
        [
            _unit_claim(
                "a_unit_necessary",
                "necessary_for",
                "comprehensiveness",
                invariant_over=[AU.invariance("replacement", min_values=3)],
            ),
            AU.claim(
                "a_site_necessary",
                statement="hidden is necessary for y",
                relation="necessary_for",
                target=TARGET,
                scope="instance",
                requirement="intervention",
                subject=Subject(site=Site(module="hidden")),
            ),
        ],
        [
            _comp_req(True),
            AU.requirement("intervention", policy=iv.INTERVENTION_POLICY, controls=False),
        ],
        name="scenario_a",
    )
    return Scenario(
        "A", evidence, plan, "a_unit_necessary", "supported", (), {"also": "a_site_necessary"}
    )


def scenario_b() -> Scenario:
    """B: attribution-only evidence for a causal claim."""
    from beyondnn._testing.causal_models import Redundant

    model = Redundant().eval()
    x = torch.tensor([[3.0, 5.0]])
    ig = A.integrated_gradients(baseline=A.zero_baseline(), n_steps=16)
    credit = A.make_claim(A.layer("p"), TARGET, x, statement="p receives attribution for y")
    attr = A.attribute(
        model,
        x,
        target=TARGET,
        method=ig,
        at=A.layer("p"),
        claims=[(credit, A.threshold_spec(ig, at=A.layer("p"), min_abs_attribution=2.0))],
    )
    plan = _plan(model, [sample_id(x)], [_p_claim()], [_int_req()], name="scenario_b")
    return Scenario(
        "B", [attr], plan, "p_necessary", "unsupported", ("attribution_is_not_intervention",)
    )


def _p_claim() -> Any:
    return bnn.audits.claim(
        "p_necessary",
        statement="p is necessary for y",
        relation="necessary_for",
        target=TARGET,
        scope="instance",
        requirement="intervention",
        subject=Subject(site=Site(module="p")),
    )


def _int_req() -> Any:
    return bnn.audits.requirement("intervention", policy=iv.INTERVENTION_POLICY, controls=False)


def scenario_c() -> Scenario:
    """C: attribution high, intervention on a redundant path shows no effect."""
    b = scenario_b()
    from beyondnn._testing.causal_models import Redundant

    model = Redundant().eval()
    x = torch.tensor([[3.0, 5.0]])
    claim = iv.make_claim(
        iv.zero("p"), TARGET, Relation.NECESSARY_FOR, x, statement="p is necessary for y"
    )
    effect = bnn.intervene(
        model,
        x,
        intervention=iv.zero("p"),
        metric=TARGET,
        claims=[(claim, iv.threshold_spec(operation=InterventionOperation.ZERO, min_effect=6.0))],
    )
    plan = _plan(model, [sample_id(x)], [_p_claim()], [_int_req()], name="scenario_c")
    return Scenario(
        "C",
        [*b.evidence, effect],
        plan,
        "p_necessary",
        "contradicted",
        ("attribution_intervention_disagree",),
    )


def scenario_d() -> Scenario:
    """D: necessary but not sufficient (y = h0 * h1)."""
    model = Product().eval()
    x = torch.tensor([[2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    selection = F.units(A.layer("hidden"), (0,), n_units=8)
    comp = F.run(
        model,
        x,
        test=F.comprehensiveness(
            target=TARGET, min_drop=3.0, statement="h0 necessary", replacement=F.zero()
        ),
        selection=selection,
    )
    suff = F.run(
        model,
        x,
        test=F.sufficiency(
            target=TARGET, max_drop=1.0, statement="h0 sufficient", replacement=F.zero()
        ),
        selection=selection,
    )
    AU = bnn.audits
    plan = _plan(
        model,
        [sample_id(x)],
        [
            _unit_claim("d_necessary", "necessary_for", "comprehensiveness"),
            _unit_claim("d_sufficient", "sufficient_for", "sufficiency"),
        ],
        [
            _comp_req(False),
            AU.requirement("sufficiency", policy=F.SUFFICIENCY_POLICY, controls=False),
        ],
        name="scenario_d",
    )
    return Scenario(
        "D",
        [comp, suff],
        plan,
        "d_necessary",
        "supported",
        ("necessary_not_sufficient",),
        {"other": ("d_sufficient", "contradicted")},
    )


def scenario_e() -> Scenario:
    """E: the effect reverses between replacements."""
    model, x = _one_unit()
    near = torch.ones(1, 32)
    near[0, 0] = 1.8
    evidence = [_comp(model, x, F.zero()), _comp(model, x, F.replacement(near))]
    plan = _plan(
        model,
        [sample_id(x)],
        [_unit_claim("e_necessary", "necessary_for", "comprehensiveness")],
        [_comp_req(False)],
        name="scenario_e",
    )
    return Scenario(
        "E",
        evidence,
        plan,
        "e_necessary",
        "assumption_sensitive",
        ("assumption_sensitive",),
        {"axis": "replacement"},
    )


def scenario_f() -> Scenario:
    """F: uncontrolled SUPPORTS; with matched random controls CONTRADICTS."""
    model = WeightedSum([1.0] * 32).eval()
    x = torch.ones(1, 32)
    evidence = [
        _comp(model, x, F.zero(), min_drop=0.5),
        _comp(
            model,
            x,
            F.zero(),
            min_drop=0.5,
            controls=F.controls(20, seed=5),
            min_fraction_below=0.95,
        ),
    ]
    plan = _plan(
        model,
        [sample_id(x)],
        [_unit_claim("f_necessary", "necessary_for", "comprehensiveness")],
        [_comp_req(False)],
        name="scenario_f",
    )
    return Scenario(
        "F",
        evidence,
        plan,
        "f_necessary",
        "assumption_sensitive",
        ("assumption_sensitive",),
        {"axis": "null"},
    )


def scenario_m() -> Scenario:
    """M: SUPPORTS without controls; the requirement demands controls."""
    model, x = _one_unit()
    plan = _plan(
        model,
        [sample_id(x)],
        [_unit_claim("m_necessary", "necessary_for", "comprehensiveness")],
        [_comp_req(True)],
        name="scenario_m",
    )
    return Scenario(
        "M",
        [_comp(model, x, F.zero())],
        plan,
        "m_necessary",
        "unsupported",
        ("missing_required_controls",),
    )


def scenario_n() -> Scenario:
    """N: a declared claim with no evidence at all."""
    model, x = _one_unit()
    plan = _plan(
        model,
        [sample_id(x)],
        [_unit_claim("n_necessary", "necessary_for", "comprehensiveness")],
        [_comp_req(True)],
        name="scenario_n",
    )
    return Scenario("N", [], plan, "n_necessary", "not_evaluated", ("no_evidence",))


def scenario_j() -> Scenario:
    """J: all evidence from another checkpoint."""
    model, x = _one_unit()
    other = WeightedSum([5.0] + [0.02] * 31).eval()
    plan = _plan(
        model,
        [sample_id(x)],
        [_unit_claim("j_necessary", "necessary_for", "comprehensiveness")],
        [_comp_req(False)],
        name="scenario_j",
    )
    return Scenario(
        "J",
        [_comp(other, x, F.zero())],
        plan,
        "j_necessary",
        "not_evaluated",
        ("other_checkpoint",),
    )


def scenario_k() -> Scenario:
    """K: evidence about a sample the plan does not declare."""
    model, x = _one_unit()
    x2 = x.clone()
    x2[0, 1] = 3.0
    plan = _plan(
        model,
        [sample_id(x)],
        [_unit_claim("k_necessary", "necessary_for", "comprehensiveness")],
        [_comp_req(False)],
        name="scenario_k",
    )
    return Scenario(
        "K",
        [_comp(model, x2, F.zero())],
        plan,
        "k_necessary",
        "not_evaluated",
        ("sample_out_of_scope",),
    )


# ------------------------------------------------------------------ concept scenarios


def _concept_setup() -> tuple[Any, Any, list[str]]:
    from beyondnn._testing.concept_models import ConceptToy, concept_inputs

    model = ConceptToy().eval()
    x = concept_inputs(600, seed=0)
    splits = ["train"] * 300 + ["val"] * 100 + ["test"] * 200
    return model, x, splits


def _concept_data(x: torch.Tensor, col: int, name: str, splits: list[str]) -> Any:
    C = bnn.concepts
    return C.dataset(
        list(torch.split(x, 1)),
        (x[:, col] > 0).long().tolist(),
        splits,
        name=name,
        label_source=name,
    )


def _enc(model: Any, concept: Any, data: Any, dist: str = "isotropic") -> Any:
    C = bnn.concepts
    return C.encoding_test(
        model,
        concept,
        data,
        controls=[
            C.random_directions(200, seed=1, distribution=dist),
            C.label_permutation(200, seed=2),
        ],
        criteria=C.encoding_criteria(min_fraction_below=0.95),
    )


def _use(model: Any, concept: Any, data: Any, output: int) -> Any:
    C = bnn.concepts
    return C.use_test(
        model,
        concept,
        data,
        target=iv.metrics.select([0, output]),
        relation="decreases",
        intervention=C.remove(C.zero()),
        controls=[C.random_neurons(50, seed=3)],
        criteria=C.use_criteria(min_change=0.25, min_fraction_beyond_controls=0.9),
    )


def _concept_plan(
    model: Any,
    data: Any,
    concept: Any,
    name: str,
    claims: tuple[Any, ...] = (),
    requirements: tuple[Any, ...] = (),
    **kw: Any,
) -> Any:
    C, AU = bnn.concepts, bnn.audits
    return _plan(
        model,
        [],
        list(claims),
        list(requirements),
        name=name,
        datasets=[data.id],
        concepts=[AU.concept(concept.id, policy=C.POLICY_V1)],
        **kw,
    )


def scenario_g() -> Scenario:
    """G: decodable but unused (h1 encodes x1 > 0 but feeds no output)."""
    from beyondnn.concepts._core import sample_set_id

    C, AU = bnn.concepts, bnn.audits
    model, x, splits = _concept_setup()
    data = _concept_data(x, 1, "gt-B", splits)
    feature = C.neuron("hidden", 1)
    concept = C.propose(feature, label="x1 > 0", definition="x1 > 0")
    enc = _enc(model, concept, data)
    use = _use(model, concept, data, 0)
    val = C.validate(concept, encoding=enc, use=[use])
    # the use test evaluates the positive test subset (use_test's declared default)
    test_ids = [data.record.samples[i] for i in data.record.indices("test", 1)]
    claim = AU.claim(
        "g_use",
        statement="h1 is used for y0",
        relation="decreases",
        target=iv.metrics.select([0, 0]),
        scope="finite_sample",
        sample_set=sample_set_id(test_ids),
        requirement="use",
        subject=Subject(site=feature.record.site, feature=feature.id),
    )
    plan = _concept_plan(
        model,
        data,
        concept,
        "scenario_g",
        (claim,),
        (AU.requirement("use", policy=C.USE_POLICY, controls=True),),
    )
    return Scenario(
        "G",
        [val],
        plan,
        None,
        "unsupported",
        ("decodable_not_used",),
        {"concept": concept.id, "claim": ("g_use", "contradicted")},
    )


def scenario_h() -> Scenario:
    """H: a GENERATED label asserted as validated, with no validation."""
    C = bnn.concepts
    model, x, splits = _concept_setup()
    data = _concept_data(x, 0, "gt-J", splits)
    wrong = C.neuron("hidden", 1)
    label = C.generated_label(model, wrong, "x0 > 0", generator="toy-labeler", revision="v1")
    concept = C.propose(wrong, definition="x0 > 0", generated=label)
    enc = _enc(model, concept, data)
    plan = _concept_plan(model, data, concept, "scenario_h")
    return Scenario(
        "H",
        [concept, enc],
        plan,
        None,
        "unsupported",
        ("generated_label_is_not_validation",),
        {"concept": concept.id},
    )


def scenario_i() -> Scenario:
    """I: a validated concept whose false-positive rate exceeds the declared cap."""
    C = bnn.concepts
    model, x, splits = _concept_setup()
    data = _concept_data(x, 7, "gt-H", splits)
    feature = C.neuron("hidden", 7)
    concept = C.propose(feature, label="x7 > 0", definition="x7 > 0")
    val = C.validate(
        concept, encoding=_enc(model, concept, data), use=[_use(model, concept, data, 4)]
    )
    plan = _concept_plan(model, data, concept, "scenario_i", counterexamples=_rule(0.2))
    return Scenario(
        "I",
        [val],
        plan,
        None,
        "unsupported",
        ("counterexample_heavy",),
        {"concept": concept.id, "status": val.semantic_status.value},
    )


def scenario_l() -> Scenario:
    """L: an encoding claim that holds under one null and fails under another (h2 on
    x2 > 0: isotropic random directions vs covariance-matched random directions)."""
    C, AU = bnn.concepts, bnn.audits
    model, x, splits = _concept_setup()
    data = _concept_data(x, 2, "gt-L", splits)
    feature = C.neuron("hidden", 2)
    concept = C.propose(feature, label="x2 > 0", definition="x2 > 0")
    iso = _enc(model, concept, data, "isotropic")
    cov = _enc(model, concept, data, "covariance")
    test_ids = [data.record.samples[i] for i in data.record.indices("test")]
    from beyondnn.concepts._core import sample_set_id

    claim = AU.claim(
        "l_encodes",
        statement="h2 encodes x2 > 0",
        relation="encodes",
        target=iso.claim.target,
        scope="finite_sample",
        sample_set=sample_set_id(test_ids),
        requirement="encoding",
        subject=Subject(site=feature.record.site, feature=feature.id),
        invariant_over=[AU.invariance("null", min_values=2)],
    )
    plan = _concept_plan(
        model,
        data,
        concept,
        "scenario_l",
        (claim,),
        (AU.requirement("encoding", policy=C.ENCODING_POLICY, controls=True),),
    )
    return Scenario(
        "L",
        [iso, cov],
        plan,
        "l_encodes",
        "assumption_sensitive",
        ("assumption_sensitive",),
        {"axis": "null"},
    )


SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "A": scenario_a,
    "B": scenario_b,
    "C": scenario_c,
    "D": scenario_d,
    "E": scenario_e,
    "F": scenario_f,
    "G": scenario_g,
    "H": scenario_h,
    "I": scenario_i,
    "J": scenario_j,
    "K": scenario_k,
    "L": scenario_l,
    "M": scenario_m,
    "N": scenario_n,
}


def build(name: str) -> Scenario:
    return SCENARIOS[name]()
