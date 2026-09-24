# BeyondNN documentation

| Folder | What belongs here | Rule |
|---|---|---|
| [`design/`](design/) | How BeyondNN works: architecture, the trace/claim schema, definitions of terms | Living documents. Change them via PR. Significant changes need an ADR. |
| [`research/`](research/) | Why BeyondNN exists and what it must answer: ecosystem audit, differentiation, research questions | Update when the ecosystem or the evidence changes, and date the change. |
| [`roadmap/`](roadmap/) | What gets built, and when: the roadmap, per-phase plans, and `PHASE_<N>_REPORT.md` go/no-go reports | A phase starts only after the previous report's go decision. |
| [`experiments/`](experiments/) | What was run and what happened | **Append-only.** Negative results stay. |
| [`decisions/`](decisions/) | Architecture Decision Records | **Append-only.** Supersede, never rewrite. |

## Where to start

1. [`research/DIFFERENTIATION.md`](research/DIFFERENTIATION.md): why the project exists, and its limits.
2. [`design/INTERPRETABILITY_DEFINITION.md`](design/INTERPRETABILITY_DEFINITION.md): the vocabulary (observed, measured, attributed, interventional, …).
3. [`design/ARCHITECTURE_PROPOSAL.md`](design/ARCHITECTURE_PROPOSAL.md): the Measurement → Claim → Test → Evidence → Assessment → WHY pipeline, and the API.
4. [`design/TRACE_SCHEMA_PROPOSAL.md`](design/TRACE_SCHEMA_PROPOSAL.md): the exact record types.
5. [`decisions/ARCHITECTURE_DECISIONS.md`](decisions/ARCHITECTURE_DECISIONS.md): what is settled and why.
6. [`roadmap/PHASE_1_PLAN.md`](roadmap/PHASE_1_PLAN.md): what gets built first.

## Release blockers

Items marked `BLOCKS_PUBLIC_RELEASE` are tracked in [`roadmap/ROADMAP.md`](roadmap/ROADMAP.md#release-blockers-blocks_public_release).

## Future additions

Per-test and per-metric reference pages (definition, implementation, interpretation, limitations) will live
in `design/tests/` once Phase 2 introduces the first claim tests. They are not created until then.
