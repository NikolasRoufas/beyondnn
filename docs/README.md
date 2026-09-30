# BeyondNN documentation

New to BeyondNN? Read the [project README](../README.md) first (installation, a five-minute quickstart, the core concepts), then run the [examples](../examples/). The pages below go deeper.

## Getting started

- [README](../README.md): what BeyondNN is, installation, quickstart, the end-to-end example.
- [Examples](../examples/): numbered, runnable scripts (`01_quickstart.py` … `07_save_reload.py`).
- [Reproducibility](REPRODUCIBILITY.md): environments, tests, and how to re-run the research experiments.

## Core concepts

- [What "interpretable" means here](design/INTERPRETABILITY_DEFINITION.md): the vocabulary (observed, measured, attributed, interventional, …) and what each status can justify.
- [Architecture](design/ARCHITECTURE_PROPOSAL.md): the Measurement → Claim → Test → Evidence → Assessment → WHY pipeline.
- [Semantic status of concepts](concepts/semantic_status.md): UNLABELED_FEATURE → PROPOSED_CONCEPT → VALIDATED_CONCEPT; GENERATED labels.

## Tracing and evidence

- [Trace schema](design/TRACE_SCHEMA_PROPOSAL.md): record kinds, epistemic status, identity and versioning.
- README sections: [evidence statuses](../README.md#evidence-statuses), [provenance](../README.md#provenance).

## Attribution and interventions

- README: [five-minute quickstart](../README.md#five-minute-quickstart) (attribution vs intervention on a redundant path), [replacements](../README.md#replacements).
- Examples: [`02_attribution.py`](../examples/02_attribution.py), [`03_intervention.py`](../examples/03_intervention.py).

## Faithfulness protocols

- [Protocol reference](protocols/README.md): comprehensiveness, sufficiency, curves, stability, counterexamples, diagnostics. Each page says what the protocol can and cannot support.
- README: [controls](../README.md#controls), [unit eligibility](../README.md#unit-eligibility).

## Concepts

- [Concepts guide](concepts/README.md): features, datasets, proposals, encoding and use tests, validation.
- [Features](concepts/features.md), [controls](concepts/controls.md), [validation](concepts/validation.md).

## Auditing

- [Audits](audit/README.md): what an audit does, and never does.
- [Writing an audit plan](audit/plans.md): claims, requirements, configuration roles, eligibility, sensitivity profiles, uncertainty, persistence, reading reports.
- [Taxonomy](audit/taxonomy.md): standings, finding kinds, codes, severities.

## WHY

- README: [WHY](../README.md#why). `bnn.compose(...)` arranges recorded evidence and never generates a narrative.

## Persistence and provenance

- README: [save, reload, verify](../README.md#save-reload-verify).
- [Audit plans: persisting evidence](audit/plans.md#persisting-evidence).

## API and stability

- [API freeze and public API inventory](API_FREEZE.md): what is frozen, experimental or internal.
- [Scientific invariants](PRE_PHASE8_INVARIANTS.md): the invariants a change must keep, each mapped to the tests that protect it.
- [Architecture decisions (ADRs)](decisions/ARCHITECTURE_DECISIONS.md): what is settled, and why.

## Research validation

- [Paper evidence ledger](research/PAPER_EVIDENCE_LEDGER.md): every validated claim, with allowed and disallowed wording, scope and counterevidence.
- [Independent ground truth](research/PHASE_7_75_INDEPENDENT_GROUND_TRUTH.md): why compiled Tracr programs with program-defined truth were used.
- [Paper readiness](research/PHASE_7_75_PAPER_READINESS.md): the remaining gaps, stated plainly.
- [Ecosystem audit](research/ECOSYSTEM_AUDIT.md) and [differentiation](research/DIFFERENTIATION.md).

## Development history

BeyondNN was built in pre-registered phases; each has a plan and a go/no-go report. These are kept as the research record: negative results and deviations included, never rewritten.

| phase | subject | documents |
|---|---|---|
| 1 | tracing, schema, provenance | [report](PHASE_1_REPORT.md), [plan](roadmap/PHASE_1_PLAN.md) |
| 2 | controlled interventions | [plan](PHASE_2_PLAN.md), [report](PHASE_2_REPORT.md) |
| 3 | attribution | [plan](PHASE_3_PLAN.md), [report](PHASE_3_REPORT.md) |
| 4 | structured WHY | [plan](PHASE_4_PLAN.md), [report](PHASE_4_REPORT.md) |
| 5 | faithfulness protocols | [plan](PHASE_5_PLAN.md), [report](PHASE_5_REPORT.md) |
| 5.5 | faithfulness on trained models | [plan](PHASE_5_5_PLAN.md), [report](PHASE_5_5_REPORT.md), [API review](PHASE_5_5_API_REVIEW.md) |
| 6 | concepts | [plan](PHASE_6_PLAN.md), [report](PHASE_6_REPORT.md) |
| 7 | audits | [plan](PHASE_7_PLAN.md), [report](PHASE_7_REPORT.md) |
| 7.5 | external validation, uncertainty, API freeze | [plan](PHASE_7_5_PLAN.md), [frozen policy](PHASE_7_5_FROZEN_POLICY.md), [report](PHASE_7_5_REPORT.md), [external validation](PHASE_7_5_EXTERNAL_VALIDATION.md), [API review](PHASE_7_5_API_REVIEW.md) |
| 7.75 | scientific fixes, independent ground truth, evidence freeze | [plan (frozen)](PHASE_7_75_PLAN.md), [scientific fixes](PHASE_7_75_SCIENTIFIC_FIXES.md), [report](PHASE_7_75_REPORT.md) |
| pre-release | framework invariants and API hardening | [report](PRE_PHASE8_HARDENING_REPORT.md) |
| 8 | public release | [report](PHASE_8_REPORT.md) |

**Other records:**
- [Roadmap](roadmap/ROADMAP.md).
- [Experiment log](experiments/EXPERIMENT_LOG.md): append-only.
- Literature reviews and design notes in [`research/`](research/).
- The Trace 3B boundary note: [roadmap/TRACE3B_FUTURE.md](roadmap/TRACE3B_FUTURE.md).

## Documentation rules

| folder | contents | rule |
|---|---|---|
| `design/` | how BeyondNN works | living; significant changes need an ADR |
| `research/` | why it exists; validation evidence | dated updates |
| `decisions/` | ADRs | append-only; supersede, never rewrite |
| `experiments/` | what was run and what happened | append-only; negative results stay |
| `PHASE_*` reports | the development record | historical; not edited after their gate |
