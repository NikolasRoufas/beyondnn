# Pre-Phase-8 Scientific Invariants

- **Purpose:** the invariants that release engineering (Phase 8) must not break.
- **Protection:** each one is protected by the permanent test(s) named here.
- **If a change needs one of these tests edited,** the change is scientific: it needs an ADR, not a release-engineering commit.
- **First place to look:** `tests/test_golden_workflow.py`. If it breaks, BeyondNN's scientific workflow changed.

| # | invariant | protected by |
|---|---|---|
| 1 | **Attribution is not causal evidence.** Attribution-only evidence for a causal claim is UNSUPPORTED (`attribution_is_not_intervention`). | scenario B (`test_audit.py`); `test_golden_workflow.py`; Phase-7.5 mutations |
| 2 | **Evidence from another checkpoint never silently counts** (`other_checkpoint`; `audit(model=...)` refuses another checkpoint). | scenario J; `test_golden_workflow.py` (reload with `model=`) |
| 3 | **Evidence about an undeclared sample or dataset never silently counts** (`sample_out_of_scope`, dataset scope). | scenario K; `test_audit.py::test_dataset_scope_mutation_excludes_concept_evidence` |
| 4 | **PRIMARY standing is determined only by PRIMARY configurations.** ALTERNATIVE / STRESS_TEST results stay visible as findings (`alternative_reverses`, `stress_test_reverses`) and never redefine the primary claim. | `test_audit_75.py`; `test_golden_workflow.py`; mutations (7.5, 7.75) |
| 5 | **Eligibility is part of claim identity.** An all-units claim and an eligible-units claim never share evidence (`eligibility_mismatch`), and eligibility survives save / reload / migration. | `test_phase775.py`; `test_golden_workflow.py`; `test_migration_matrix.py` |
| 6 | **Special units are never filtered automatically.** An all-units claim includes [CLS] / [SEP]; exclusion is always a caller declaration. | `test_phase775.py::test_special_token_eligible_vs_ineligible`; mutation |
| 7 | **Replacement is always explicit, and its identity is recorded** (name + digest) through to the audit axis. | `test_faithfulness.py` (required keyword); `test_phase775.py`; `test_golden_workflow.py` |
| 8 | **Unattainable controls do not become contradictions** (`control_criterion_unattainable` → INCONCLUSIVE). A competitive-only failure is labelled `effect_without_competitive_advantage`. | `test_phase775.py`; mutations |
| 9 | **Decodability does not imply causal use** (`decodable_not_used`; the decodable-but-unused concept stays PROPOSED). | scenario G; `test_concepts.py::test_case_b_decodable_but_unused_stays_proposed`; `test_phase775_analysis.py` |
| 10 | **Generated labels do not become validated concepts** (`generated_label_is_not_validation`). | scenario H; `test_concepts.py` |
| 11 | **There is no global explanation score:** no robustness, trust, confidence, accuracy or interpretability number in any API or report. | API review (ADR-007); `docs/API_FREEZE.md` |
| 12 | **Saved evidence can be independently re-audited:** save → fresh process → load → verify → audit → WHY gives the same conclusions, and corrupt evidence is refused, naming the trace. | `test_golden_workflow.py`; `test_audit_75.py` |
| 13 | **Old records keep their meaning.** Every registered migration maps an old version to its documented old meaning (`None` = unknown / every unit / undeclared); nothing is invented. | `test_migration_matrix.py` |
| 14 | **Reports say who produced them** (BeyondNN version, audit semantics; ADR-056). Verification compares scientific content. | `test_golden_workflow.py` |
| 15 | **Phase-7.5 results are immutable evidence.** | `test_phase75_immutable.py` |
| 16 | **Claims are looked up by name.** `report.claims` order is by name and carries no meaning. | `test_golden_workflow.py::test_named_claim_lookup_is_independent_of_plan_order` |
| 17 | **Concept use thresholds are declared in the target's scale** (ADR-055). The threshold function takes train-split values only. | `test_phase775_analysis.py` |
