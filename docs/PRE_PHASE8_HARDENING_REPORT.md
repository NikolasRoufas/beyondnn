# Pre-Phase-8 Framework Invariant and API Hardening Report

- **Date:** 2026-10-01. Local only: nothing pushed or published.
- **Not a research phase:** no model experiment was re-run, no feature was added, and no scientific number changed.

## 1. Starting HEAD

`32cd645` (Phase 7.75 gate: SCIENTIFICALLY READY FOR PHASE 8 WITH PAPER LIMITATIONS), with a clean tree.

**Baseline** (Phase 7.75 final check at this HEAD):
- 1038 passed + 1 skipped without Captum, and 1059 with Captum, on 3.10 / 3.12 / 3.14;
- ruff and mypy --strict clean;
- Phase-7.75 mutations 19/19.

## 2. Ending HEAD

The commit that adds this final text ("Pre-Phase-8 hardening report: final status"), directly after the commits listed in §18.

## 3. Golden workflow result

`tests/test_golden_workflow.py` (public API only; fixed-weight 24-unit model; 6 samples) runs:

trace → IG attribution → declared eligibility → selection → faithfulness interventions (named train-mean PRIMARY and an extreme STRESS_TEST replacement) with matched controls → site intervention → concept dataset, encoding and use tests → validation → plan with two claims (all units; units 1–23) → audit → WHY → save → **fresh process** → load → verify → re-audit → WHY.

**Pinned invariants:**

| invariant | pinned value |
|---|---|
| all-units claim | SUPPORTED 6/6 |
| eligible claim (h0 not eligible) | CONTRADICTED 6/6 |
| `eligibility_mismatch` on both claims | the other claim's evidence is never used |
| `stress_test_reverses` present | the standing is unchanged |
| roles | {primary, stress_test} |
| replacement axes | `tensor/extreme`, `tensor/train_mean` |
| concept | VALIDATED, audit SUPPORTED |
| WHY | AUDIT and CONCEPTS sections |
| reload | identical conclusions |
| provenance | every item of §7 recoverable |
| corrupt evidence | refused, naming the trace |
| named claim lookup | independent of plan order |

**Result:** 6 tests, all pass.

## 4. Public API inventory

In `docs/API_FREEZE.md` ("Public API inventory"):
- **FROZEN PUBLIC:** the top level, `schema`, `attribution`, `interventions`, the claim-deciding part of `faithfulness`, `concepts` (except `sae_feature`), `audits`, `explain`, `protocols` constants.
- **PUBLIC BUT EXPERIMENTAL:** `faithfulness.curve` / `run_dataset` (5.5-F-7 / 8 / 10), the four faithfulness diagnostics, `concepts.sae_feature`.
- **INTERNAL:** `_testing`, `core.*`, `faithfulness.stats`, audit engine / evidence internals, private schema helpers.

No API was expanded.

## 5. Migration matrix result

`tests/test_migration_matrix.py`: 26 tests, all pass.
- **Coverage:** every registered migration is in the matrix; a test asserts this, so a new migration cannot be added untested.
- **Method:** each old version is produced from a real current record (the golden workflow) by removing the fields that version lacked, then decoded.

| record | old versions → current | migration behaviour (scientific meaning preserved) |
|---|---|---|
| `evidence_selection` | 1, 2 → 3 | v1: `unit_axes` / `unit_reduction` = None (last-axis units); v2: `eligible` / `eligibility` = None (**every unit**) |
| `audit_plan` | 1, 2 → 3 | v1: `roles = []` (no roles: Phase-7 standings); v2: `selection.eligibility = None` (**every unit**) |
| `claim` | 1, 2 → 3 | v1: `subject.unit_axes = None`; v2: `subject.feature = None` |
| `intervention` | 1, 2, 3 → 4 | v1: `units = None`, `retain = False` (whole leaf); v2: `unit_axes = None`; v3: no direction |
| `input` | 1, 2 → 3 | v1: `pass_index = None` (unknown); v2: `sample_id = None` (**never invented**) |
| `output`, `execution_occurrence` | 1 → 2 | `pass_index = None` (unknown) |
| `provenance` | 1 → 2 | `declared_model = None` |
| `causal_effect` | 1 → 2 | `metric.declaration = None` |

**Also checked:**
- every current record round-trips through JSON;
- declared eligibility, roles, per-sample targets and replacement identity survive the round trip.

**Unsupported versions:**
- a version newer than current is refused (`UnsupportedVersionError`, existing tests);
- a missing migration is refused, never guessed;
- audit reports of format 1 (Phase 7) are not readable. Formats 2 and 3 are.

## 6. Terminology review

**Levels, all consistent:**

| level | terms |
|---|---|
| single test (`Outcome`) | supports / contradicts / inconclusive / not_applicable / errored |
| policy (`Verdict`) | supported / contradicted / mixed / inconclusive / untested |
| audit (`Standing`) | refines the verdict: + assumption_sensitive, unsupported, not_evaluated |
| evidence status | observed / measured / attributed / interventional / estimated_causal / validated_concept / generated |
| concept lifecycle (`SemanticStatus`) | unlabeled_feature → proposed_concept → validated_concept |
| configuration roles | primary / alternative / stress_test, plus the report-only `undeclared` for values no rule matches |

- **Shared words** (supported, contradicted, mixed, inconclusive; validated_concept) mean the same at each level they appear.
- **No term** is used with materially different meanings, so nothing was renamed.

## 7. Error-message review

Probed with `experiments/phase7_75/errors_probe.py` and the audit scenarios:

| failure | message (abridged) | verdict |
|---|---|---|
| ineligible declared unit | `declared units (0,) are not all eligible (content)` | clear |
| eligibility without name | "declare eligible units together with their eligibility name, e.g. …" | clear |
| wrong sample | "the selection is about sample X, not the tested input Y; evidence about another input is never used" | clear |
| wrong checkpoint | "the model (…) is not the plan's checkpoint (…); an audit never applies to another checkpoint" | clear |
| wrong sample / checkpoint in an audit | findings `sample_out_of_scope` / `other_checkpoint` name the result, the sample / checkpoint and the plan's | clear |
| missing replacement | Python: "missing 1 required keyword-only argument: 'replacement'" | acceptable: names the argument; not changed |
| unattainable control | finding: "could not meet their control criterion by construction … treated as inconclusive, not as contradicting" | clear |
| unsupported evidence type | "cannot audit int: evidence must be traces, saved trace paths, or BeyondNN result objects" | clear |
| newer serialized version | "audit_plan record_version 99 is newer than supported (3)" | clear |
| tampered record | "stored id … does not match content (expected …)" | clear |
| corrupt saved evidence | "trace.json is not valid JSON: …" | **fixed:** now names the trace directory |
| provenance mismatch at verification | "the stored report differs … at N path(s)" | **improved:** names an audit-semantics difference (ADR-056) |

No message recommends disabling a check.

## 8. Provenance review

For the golden workflow (`test_provenance_is_complete_for_the_golden_workflow`):

| item | recoverable from |
|---|---|
| BeyondNN version | evidence provenance (`environment.beyondnn_version`) and, since ADR-056, the report's `producer` |
| schema / record versions | trace envelope, record envelopes |
| checkpoint | provenance `model.state_digest`, equal to the plan checkpoint |
| sample / input identity | claim estimand, equal to the plan samples |
| dataset identity | the concept validation's dataset, in the plan datasets |
| target | the claim target, equal to the plan's per-sample target |
| protocol version | test spec, equal to `PROTOCOL_VERSIONS` |
| AuditPlan identity | report `plan_id` and the full plan |
| replacement identity | spec `replacement` (name + digest) |
| eligibility | selection record and spec |
| configuration role | report test entries |
| control configuration and seed | spec `controls` (n, seed, strategy) |

**Gap found and fixed:** a report did not record which BeyondNN / audit semantics produced it. ADR-056 adds the `producer` block, and report format 3 still reads format 2.

## 9. Scientific-safe-default review

All eight questions have the answer **NO**, each protected by an existing permanent test:

| question | protected by |
|---|---|
| Attribution-only support for a causal claim? | scenario B |
| Silent zero replacement? | `test_faithfulness` (required keyword) |
| Silent eligibility change? | `test_phase775`, migration matrix |
| STRESS_TEST decides the PRIMARY standing? | `test_audit_75`, golden |
| Evidence from the wrong sample / checkpoint counts? | scenarios J, K; `audit(model=)` |
| Generated label → validated? | scenario H |
| Unattainable control → contradiction? | `test_phase775` |
| Content-token claim uses special-token evidence? | `test_phase775`, golden |

The mapping is in `docs/PRE_PHASE8_INVARIANTS.md`. No duplicate tests were added.

## 10. Changes made

1. **`tests/test_golden_workflow.py`** (6 tests): golden workflow, eligibility / provenance of saved selections, named lookup, corrupt evidence, provenance completeness, report format 2 compatibility.
2. **`tests/test_migration_matrix.py`** (26 tests).
3. **Trace load errors** name the trace directory (`core/persistence.py`; same exception type).
4. **ADR-056:** report `producer` block (format 3; reads 2); `verify_report` compares scientific content and names a semantics difference.
5. **Phase-7.5 researcher workflow:** `report.claim(name)` instead of `claims[0]` (single-claim plan: behaviour identical).
6. **Docs:**
   - `PRE_PHASE8_INVARIANTS.md`;
   - API inventory in `API_FREEZE.md`;
   - `audit/plans.md` (claim order, producer);
   - README status (was "Phase 6"; now Phases 1–7.75 and the freeze);
   - roadmap Phase-8 scope (release engineering; the earlier "Benchmark" scope marked superseded);
   - CHANGELOG; ADR-056.
7. **`experiments/phase7_75/mutations_pre8.py`** (4 targeted mutations).

## 11. Issues inspected but intentionally not changed

- **`report.claims` ordering** (by name): documented as non-semantic, not changed.
- **Python's generic missing-`replacement` message:** names the argument; changing it would need a sentinel default in a frozen signature.
- **Phase-7 (format 1) reports** are not loadable: a historical limitation, stated.
- **Positional `.claims[0]` in tests** on single-claim plans: unambiguous; not changed.
- **Experimental surfaces** (`curve`, `run_dataset`, diagnostics, `sae_feature`) keep their known gaps. Classified, not expanded.
- **Deferred as capability, not correctness:** a sentinel-based replacement error, per-report hardware identity, compressed evidence (F-10).

## 12. Tests

`experiments/phase7_75/results/validation_matrix_pre8.txt`: **1070 passed + 1 skipped** without Captum, and **1091 passed** with Captum, on Python **3.10 / 3.12 / 3.14**.

The baseline was 1038 + 1 / 1059. The +32 tests are the golden workflow (6) and the migration matrix (26).

## 13. Mutations

`experiments/phase7_75/results/mutations_pre8.json`, all run at HEAD after the changes:

| suite | killed |
|---|---|
| Phase 7.75 | **19/19** (still holds) |
| Phase 7.5 | **24/24** |
| new targeted mutations | **4/4**: report producer dropped; `verify_report` comparing the producer; trace load error not naming the trace; migration inventing eligibility for old selections |

## 14. Lint / mypy / build / clean wheel

- ruff check and format: clean (169 files);
- mypy --strict: clean, with and without Captum (117 files);
- build: sdist and wheel.
- **Clean wheel** (Python 3.12, wheel + numpy only): 7 README examples OK; `examples/` 3/3 OK.
- **Researcher workflows:**
  - Phase 7.5: run and reload, `same as before: True`;
  - Phase 7.75 (with eligibility): run and reload, `same as before: True`.
- **Golden invariant workflow:** in the test suite, with a fresh-process reload.

## 15. Was scientific evidence invalidated?

**No.**
- The only behaviour changes are an error message and an additive report block.
- No audit outcome, record or number changed. Phase-7.5 results are pinned by hash (`test_phase75_immutable.py`), and Phase-7.75 results are untouched.

## 16. Must experiments be re-run?

**No.**

## 17. Gate

**PRE-PHASE-8 HARDENING COMPLETE.**
- The final framework invariants are listed in `docs/PRE_PHASE8_INVARIANTS.md` and protected by permanent tests and mutations.
- No framework correctness issue remains open.
- No scientific result needs regeneration.

## 18. Git status

Clean after the final commit.

9 local commits after `32cd645` (no attribution trailers):

- `3c0e7b0` Permanent golden scientific-workflow test (eligibility, roles, replacement identity, provenance, fresh-process reload, WHY); named claim lookup test; Phase-7.5 workflow uses named lookup (single claim: behaviour unchanged)
- `3bffb8a` Migration matrix test: every registered migration from each old version to current keeps its documented old meaning; current records round-trip
- `d8d1e21` Trace load errors name the trace directory (one corrupt trace among many must be identifiable); regression test
- `c7a7c5c` Audit reports record their producer (BeyondNN version, audit semantics; ADR-056); provenance-completeness and report-format tests; README status corrected
- `b4fb7c0` Pre-Phase-8 invariants (each mapped to its permanent tests); public API inventory in API_FREEZE; roadmap Phase 8 scope corrected
- `030d813` Pre-Phase-8 targeted mutation script
- `1ed537a` Error-message probe; pre-Phase-8 hardening report draft
- `b045f64` Pre-Phase-8 hardening: regression matrix (1070+1 / 1091 on 3.10/3.12/3.14), mutations (Phase 7.75 19/19, Phase 7.5 24/24, targeted 4/4), report
- (this commit) Pre-Phase-8 hardening report: final status

## 19. Pushed / published status

Not pushed (no remote), not published. No paper written. Phase 8 not started. Trace 3B not started.
