# Scientific audits (Phase 7)

`bnn.audit(evidence, plan=plan) -> AuditReport` classifies the claims and concepts that an `AuditPlan` declares, using the evidence that was recorded, re-derived, and is in scope. Decisions: ADR-044 to ADR-047. Pre-registration: `docs/PHASE_7_PLAN.md`.

- [Taxonomy: standings, finding kinds, codes, severities](taxonomy.md)
- [Writing a plan](plans.md)

## What an audit does

1. **Collects traces.** Evidence can be `TraceResult`s, paths to saved trace directories, or Phase 3–6 result objects. For objects, only their traces are used, including nested attribution, test, encoding and use traces.
2. **Checks integrity.** Record ids must match content, references must resolve, and one id may never carry two contents. A failing trace is excluded as a whole (INTEGRITY_FAILURE).
3. **Re-derives every claim-test result** from the records it cites, with its registered protocol:
   - intervention and attribution thresholds;
   - comprehensiveness and sufficiency (controls regenerated from their seeds; magnitudes checked);
   - concept encoding and use (from their traces alone: activations, effects, controls, counterexamples, assessments);
   - concept validations (summaries, rates and derived status) and fitted features (from any supplied recording of the train split).

   A result that does not re-derive is excluded and reported, never corrected.
4. **Applies the plan's scope.** Evidence from another checkpoint or model declaration is excluded (PROVENANCE_MISMATCH), and so is evidence about an undeclared sample or dataset (SCOPE_MISMATCH).
5. **Classifies each plan claim** by formal structure (relation, subject or selection, target, estimand), never by record id or statement text. For each claim it:
   - derives the verdict with `derive_verdict` under the claim's requirement policy;
   - explains disagreements by assumption axes;
   - re-evaluates declared alternative thresholds from the recorded evidence;
   - checks declared invariances, required controls, and type or scope overclaims;
   - applies counterexample caps.
6. **Classifies each plan concept** from its validations and tests: decodable-but-unused, null / replacement sensitivity, counterexample caps with identities, generated labels, the polysemanticity indicator, and SAE limitations.
7. **Reports coverage and limitations**: coverage, not confidence.

## What an audit never does

- run a model;
- compute a score or a confidence;
- resolve a contradiction;
- turn a per-sample distribution into a claim-level truth value;
- treat missing evidence as positive or negative;
- state that an explanation is trustworthy, reliable or correct.

## What an audit cannot see

- **Evidence that was not recorded or not supplied.** A favourable configuration chosen *before recording* is invisible. The defence is to declare invariances: a claim that asserts invariance over the replacement, k, threshold or null, but was tested under one value, is UNSUPPORTED (`invariance_untested`).
- **A dishonest plan.** A plan that declares no invariances is visible to every reader of the report, because the plan is embedded in full.
- **Bugs in BeyondNN's protocols.** Re-derivation uses the same protocol code.

## Persistence and verification

```python
report = bnn.audit(evidence, plan=plan)
report.save("audit.json")                      # atomic; never overwrites
doc = bnn.audits.load_report("audit.json")     # a dict
bnn.audits.verify_report(doc, evidence, plan)  # re-derives; AuditMismatchError on any difference
bnn.audit([saved_trace_dir, ...], plan=plan)   # the same report from reloaded traces
```

- **Plans** are records: `bnn.schema.to_json(plan)` and `bnn.schema.from_json(text)`.
- **Reports** are deterministic JSON: sorted keys, the full plan embedded, and a digest over the evidence record ids.

## WHY integration

`bnn.compose(trace, ..., audit=report)` adds an AUDIT section with:
- the per-sample standings for the reference input;
- the distribution across the declared samples;
- the finite-sample and population claims, and the concept audits, as context.

It refuses a report about another checkpoint, and one whose per-sample claims do not declare the reference sample.
