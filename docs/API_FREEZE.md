# API Freeze (Phase 7.5)

- **Status:** frozen at the end of Phase 7.5, for the Phase-8 framework paper and release preparation.
- **Rules from here on:**
  - **Changes within the frozen surface** need an ADR stating why, a record-version bump with a migration where records change, and a CHANGELOG entry.
  - **Additive changes** (a new optional keyword, a new function, a new finding code) are allowed with an ADR.
  - **Breaking changes** to frozen names or semantics are allowed only to fix a BLOCKING defect.
- **The version string** stays `0.0.0.dev0` until Phase-8 release preparation. The freeze concerns names, signatures, record formats and semantics, not the version number.

## 1. Frozen surface

| module | frozen names |
|---|---|
| `beyondnn` | `trace`, `recording`, `TraceResult`, `load_trace`, `instrument`, `compose`, `intervene`, `attribute`, `audit`, `EstimandScope`, `EvidenceStatus`, `Outcome`, `Relation`, `Verdict`, and the subpackages below |
| `beyondnn.schema` | the 104 names in `__all__` (record classes, enums, `to_json` / `from_json` / `to_dict` / `from_dict`, errors, `derive_verdict`, `derive_semantic_status`, `verify_ref`) |
| `beyondnn.attribution` | the 25 names in `__all__` |
| `beyondnn.interventions` | the 25 names in `__all__`, plus `metrics.select` / `difference` / `margin` / … |
| `beyondnn.faithfulness` | the 34 names in `__all__`. `replacement=` is **required** in `comprehensiveness`, `sufficiency` and `curve` (ADR-052) |
| `beyondnn.concepts` | the 39 names in `__all__`, including `load_validation` |
| `beyondnn.audits` | the 33 names in `__all__`: plan builders (`plan`, `claim`, `concept`, `requirement`, `alternative`, `invariance`, `selection`, `role`, `counterexample_rule`, `checkpoint_of`), `audit`, `verify_report`, `load_report`, the evidence helpers, the uncertainty functions, and the report types |
| `beyondnn.explain` | the 17 names in `__all__` |

## 2. Frozen formats and semantics

**Record kinds and versions:**
- `audit_plan` v2 (v1 migrates);
- `claim` v3, `input` v3, `intervention` v4;
- `causal_effect`, `evidence_selection`, `execution_occurrence`, `output`, `provenance`: v2;
- every other kind: v1.

**Versioned formats:**
- `SCHEMA_VERSION` "0.1";
- trace persistence `FORMAT_VERSION` 1;
- audit report `REPORT_FORMAT_VERSION` 2;
- protocol versions: all 1 (`protocols.PROTOCOL_VERSIONS`).

**Audit semantics:**
- **Standings:** the 7 values SUPPORTED, CONTRADICTED, MIXED, ASSUMPTION_SENSITIVE, INCONCLUSIVE, UNSUPPORTED, NOT_EVALUATED.
- **Findings:** the 13 finding kinds; the 3 severities; the finding codes emitted by `audits.engine`.
- **Roles:** with declared roles, standings use PRIMARY configurations only:
  - ALTERNATIVE reversals are QUALIFYING;
  - STRESS_TEST reversals are INFORMATIONAL;
  - undeclared values are listed and never counted.
- **The axis-key grammar** (`docs/audit/plans.md`).
- **No score:** there is no robustness, confidence, interpretability or trust number anywhere in the API (ADR-007). A profile's counts are descriptive.
- **Uncertainty:** `Interval` records the quantity, estimate, bounds, level, method, unit, n, k, draws and seed. Wilson bounds are exact 0 / 1 at k = 0 / n.

## 3. Not frozen

- `beyondnn._testing` (test fixtures and scenario builders).
- `beyondnn.faithfulness.stats` (5.5-F-15).
- The internal modules (`beyondnn.core.*` and the `engine` / `evidence` internals).
- The text of `render()` outputs and error messages (their information content is tested; their wording is not frozen).
- The experiment scripts and benchmarks.

## 4. Known open items carried into the freeze

These do not block the freeze, because each has a workaround that keeps records correct:
- F-7: one audit per `declared` approach at a site.
- F-9: two `sample_id` entry points.
- F-10: evidence size.
- The open Phase-5.5 items listed in `PHASE_7_5_API_REVIEW.md` §2.

## Phase 7.75 additions under the freeze (additive; ADRs 053–055)

- **Eligibility:** `faithfulness.ranking` / `top_k` / `units` take `eligible=` and `eligibility=`; `audits.selection(..., eligibility=)`.
- **Record versions:** `evidence_selection` v3 (v2 migrates); `audit_plan` v3 (v2 migrates).
- **New finding codes:** `eligibility_mismatch`, `control_criterion_unattainable`, `effect_without_competitive_advantage`.
- **Concept criteria:** policy only (ADR-055). `min_change` is declared in target units, derived from the train-split SD of the clean target; no API change.
- **Rejected:**
  - a control-role ontology (negative / competitive / stress);
  - a model-native / OOD field on replacements;
  - automatic special-token filtering.

  The reasons are in `docs/PHASE_7_75_SCIENTIFIC_FIXES.md`.
