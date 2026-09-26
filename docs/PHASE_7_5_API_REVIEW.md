# Phase 7.5 API Review: external researcher, persisted workflows, external benchmarks

**Scope:**
1. **The simulated external-researcher workflow** (`experiments/phase7_5/researcher_workflow.py`):
   - it runs in a clean virtual environment holding only the built wheel and numpy, uses the public API only (no `beyondnn._testing`, no private imports), and uses a model new to BeyondNN;
   - steps: trace → attribution → intervention → faithfulness → concept evidence → audit → WHY → save → *fresh process* → load → audit → verify → WHY.
2. **Every public-API interaction** in the Phase-7.5 experiments: InterpBench / Tracr, central A–D, e-SNLI, token probes.

**Classes** (request §18): BLOCKING API DEFECT / SCIENTIFIC AMBIGUITY / ERGONOMIC ISSUE / DOCUMENTATION ISSUE / EXPECTED LIMITATION.

**Numbering:** the Phase-5.5 findings keep their `PHASE_5_5_API_REVIEW.md` numbers, prefixed "5.5-" below.

## 1. The external-researcher workflow

**First run (before fixes): failed.**
- The researcher could not write the audit plan from the documentation: the plan's `samples` and `sample_targets` need the recorded sample identity, and no public, documented helper computed it (F-1).
- After a private workaround, `save → reload → verify_report` failed, because the saved trace set differed from the set the audit had ingested (F-2).

**After the fixes** (`6b05859`, ADR-050):
- both processes pass end to end;
- the reloaded audit, the verified report, the rebuilt concept validation and the WHY are identical to the in-process ones (`reload ok … same as before: True`);
- the workflow is re-run in the Phase-7.5 regression matrix from the final wheel.

## 2. Findings

| # | Finding | Class | Status |
|---|---|---|---|
| F-1 | No public, documented way to compute the recorded sample identity needed by `audits.plan(samples=…)` / `claim(sample_targets=…)`. `interventions.sample_id` existed, but nothing in the audit docs pointed to it | BLOCKING API DEFECT | **fixed**: `audits.sample_id` (same function), documented in `docs/audit/plans.md` |
| F-2 | No public way to persist *exactly* the traces an audit ingests; `verify_report` after reload failed | BLOCKING API DEFECT | **fixed**: `audits.traces_of` / `save_evidence` / `load_evidence` |
| F-3 | `AU.selection("input", …)`: the input shorthand for input-space attributions was undocumented | DOCUMENTATION ISSUE | **fixed** (`docs/audit/plans.md`) |
| F-4 | A "margin" target (predicted vs best other class) needed a hand-written metric; the development cases showed that a runner-up margin fixed from the clean pass mis-states the claim when the runner-up changes (14 FNs on dev case 7) | ERGONOMIC ISSUE, with scientific consequences | **fixed**: builtin `interventions.metrics.margin` (ADR-051) |
| F-5 | Tensor replacements had only a content hash in their axis key, so roles could not be declared by name | ERGONOMIC ISSUE | **fixed**: `faithfulness.replacement(t, name=…)` → axis key `tensor/<name>:<hex>` (ADR-048) |
| F-6 | A WHY with concepts after restart needed the in-memory `ConceptValidationResult`, so the persisted loop could not produce WHY | BLOCKING API DEFECT | **fixed**: `concepts.load_validation(traces)` (ADR-050); tests on a reconstruction are refused (no inputs, no tensors) |
| F-7 | Two approaches that both use `method="declared"` at one site (human rationale vs random span, e-SNLI) cannot be told apart in one plan, because the selection identity is (site, method, k) | ERGONOMIC ISSUE | open, bounded: one audit per approach (as in `esnli.py`). A future `selection(…, source=<record>)` would remove it |
| F-8 | The axis-key grammar needed to write role patterns was undocumented | DOCUMENTATION ISSUE | **fixed**: table in `docs/audit/plans.md` |
| F-9 | `interventions.sample_id` raises `InterventionError`, `audits.sample_id` raises `SampleIdentityError`, for the same identity | DOCUMENTATION ISSUE | open, minor: both frozen; the values are identical |
| F-10 | Evidence on disk is large: 100 MB for 3 samples of model A (38k records, dominated by per-control records); 16 MB for 5 InterpBench samples | ERGONOMIC ISSUE | open: recorded in the performance section; retention options exist for traces, not for protocol results |
| F-11 | An over-restrictive null (EH4: count null on 4-head models) rejects known-correct claims. The audit cannot tell this from a true negative | SCIENTIFIC AMBIGUITY | open, inherent: only declared roles contain it (ALTERNATIVE → qualifying reversal). Documented in `PHASE_7_5_EXTERNAL_VALIDATION.md` §3.1 |
| F-12 | The frozen concept policy rejects a known-used Tracr variable (use effect 0.06 < 0.1) | SCIENTIFIC AMBIGUITY | open: not tuned (request §21); reported as a limitation |
| F-13 | A resample replacement can be a no-op on a given sample (the source's activation equals the clean one) | EXPECTED LIMITATION | correct behaviour: INCONCLUSIVE (`inconclusive_results`), never SUPPORTED or CONTRADICTED |
| F-14 | Plans refuse duplicate samples. The benchmark's own `unique_data` returned duplicates (case 124) | EXPECTED LIMITATION | correct refusal with a clear message; the benchmark issue is handled by DV-1 |
| 5.5-F-22 | `comprehensiveness` / `sufficiency` / `curve` silently defaulted to the zero replacement | BLOCKING for scientific use (an implicit OOD assumption) | **fixed**: `replacement` is required (ADR-052); the InterpBench zero-ablation FP rate (13%) is the evidence |

### Phase-5.5 findings still open (unchanged, bounded)

- 5.5-F-7 / F-8: `run_dataset` has one target and no `model_kwargs`.
- 5.5-F-9 / F-10: no provenance-bearing ablation rankings or declared-ranking curves.
- 5.5-F-12: one `reduce` for both diagnostics.
- 5.5-F-13: two sign conventions (documented).
- 5.5-F-15: `faithfulness.stats` not exported.
- 5.5-F-18: `JsonValue`-typed statistics.
- 5.5-F-21: SUPPORTS without `min_fraction_below` is not "better than random"; policies must opt in, as every Phase-7.5 policy does.

## 3. Changes made in this phase, and changes rejected

**Made:**
- **ADR-048:** configuration roles, sensitivity profiles, configuration-level disagreement, audit-plan v2 with v1 migration, report format v2, replacement names.
- **ADR-049:** `audits.wilson` / `bootstrap` / `paired_bootstrap`, and the `Interval` type.
- **ADR-050:** `concepts.load_validation` and protocol versions.
- **ADR-051:** `metrics.margin`.
- **ADR-052:** required replacements.
- **Evidence helpers:** `audits.sample_id` / `traces_of` / `save_evidence` / `load_evidence`.

**Rejected:**
- **A per-claim "robustness", "sensitivity" or "confidence" number in the profile:** it would become a score (ADR-007). The profile gives counts only.
- **Automatically downgrading a PRIMARY standing when most alternatives disagree:** roles would stop meaning anything, and standings would depend on how many alternatives were declared.
- **A default role for undeclared values:** undeclared values are listed and never counted, so an omission cannot silently change a standing.
- **Tuning the concept `min_change` / `min_fraction_below` after E2:** forbidden by the request (§21).
- **`selection(…, source=…)` for F-7:** feature creep in a freeze phase. Deferred.
- **A compressed evidence format for F-10:** a persistence-format change in a freeze phase. Deferred.
