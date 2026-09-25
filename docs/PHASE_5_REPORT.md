# Phase 5 Gate Report: Faithfulness Tests

- **Date:** 2026-09-25
- **Plan:** `docs/PHASE_5_PLAN.md` (committed in `b7e1073` before any Phase-5 code, with every expected value).
- **Decisions:** ADR-032, ADR-033; ADR-030/031 finalized.
- **Repository:** local only, never pushed.

## 1. Starting commit

`39499da` (Phase 4 head). Verified before starting:
- clean tree;
- Python 3.10.16 and 3.12.13 (torch 2.14.0), and 3.14.3 (torch 2.12.0);
- without Captum: 808 passed + 1 reported skip; with Captum 0.9.0: 829 passed;
- ruff, format, and mypy `--strict` (with and without Captum) clean;
- sdist and wheel build.

There is no `beyondnn/causal/` package: the causal machinery is `beyondnn/interventions/`.

## 2. Ending commit

The commit that adds this report (the last one below).

## 3. Commits

| Commit | Content |
|---|---|
| `b7e1073` | Finalize ADR-030/ADR-031 and add the Phase 5 plan |
| `847d022` | Extend interventions to units, model inputs and comparison families (ADR-032) |
| `7044ce2` | Add faithfulness protocols, controls, curves and stability (ADR-033) |
| `ef379a0` | Compose faithfulness results into the structured WHY |
| `3030edc` | Add Phase 5 guard tests from the mutation audit |
| `59f96f9` | Document faithfulness protocols and add the README premise example |
| (this commit) | Add the Phase 5 faithfulness benchmark and gate report |

## 4. Files created

- **Code:**
  - `beyondnn/faithfulness/{__init__,claims,runner,spec,stability,stats,verify}.py`
  - `beyondnn/schema/faithfulness.py`
  - `beyondnn/_testing/faithfulness_models.py`
- **Tests:** `tests/test_faithfulness.py`, `tests/test_faithfulness_why.py`, `tests/test_intervention_units.py`
- **Benchmark:** `benchmarks/bench_faithfulness.py`
- **Docs:**
  - `docs/PHASE_5_PLAN.md`, `docs/PHASE_5_REPORT.md`
  - `docs/protocols/{README,comprehensiveness,sufficiency,curves,stability,counterexample,diagnostics}.md`

## 5. Files modified

- **Code:**
  - `beyondnn/__init__.py`, `beyondnn/protocols.py`
  - `beyondnn/interventions/{__init__,runner,spec}.py`
  - `beyondnn/schema/{__init__,interventions,limitations}.py`
  - `beyondnn/explain/{bundle,response,views}.py`
- **Tests:** `tests/{test_intervention_schema,test_limitations,test_package,test_phase1_audit,test_phase4,test_serialization,test_trace_schema,test_readme,test_benchmark_smoke}.py`
- **Docs and meta:**
  - `README.md`, `CHANGELOG.md`, `benchmarks/README.md`
  - `docs/README.md`
  - `docs/decisions/ARCHITECTURE_DECISIONS.md`
  - `docs/design/{ARCHITECTURE_PROPOSAL,TRACE_SCHEMA_PROPOSAL}.md`
  - `docs/roadmap/ROADMAP.md`, `docs/experiments/EXPERIMENT_LOG.md`

**Changed Phase 1–4 tests, all deliberate:**
- `test_intervention_schema`: one expected error message; root *outputs* are still refused.
- `test_phase4`: one type-ignore removed after `Coverage.faithfulness_evaluated` became `bool`. The assertion still holds, because it cannot be true without named protocols.
- Inventory tests: new kinds, codes, and the top-level name.

## 6. Files deleted

None.

## 7. Public API added

- **Top level:** `faithfulness`. The top level is now 18 names:
  - `EstimandScope`, `EvidenceStatus`, `Outcome`, `Relation`, `TraceResult`, `Verdict`, `__version__`;
  - `attribute`, `attribution`, `compose`, `faithfulness`, `instrument`, `intervene`, `interventions`, `load_trace`, `recording`, `schema`, `trace`.
- **`bnn.faithfulness`:**
  - tests and runners: `comprehensiveness`, `sufficiency`, `run`, `run_dataset`, `curve`;
  - stability: `stability`, `transformation`;
  - method diagnostics: `method_agreement`, `baseline_sensitivity`, `ig_step_sensitivity`;
  - selections and controls: `top_k`, `ranking`, `units`, `selector`, `fixed`, `zero`, `replacement`, `controls`;
  - policies: `COMPREHENSIVENESS_POLICY`, `SUFFICIENCY_POLICY`;
  - evaluator: `evaluate`;
  - result types: `FaithfulnessResult`, `DatasetResult`, `CurveResult`, `DiagnosticResult`;
  - declaration types: `Selection`, `SelectionRule`, `TestTemplate`, `Replacement`, `Controls`, `Transformation`;
  - errors: `FaithfulnessError`, `SelectionMismatchError`, `PerturbationNotAppliedError`.
- **`bnn.interventions`:** `zero_input`, `constant_input`, `compare_family`, `ComparisonFamily`; `units=` and `retain=` on `zero`/`constant`/`patch`.
- **`bnn.compose` / `ExplainResponse.from_evidence`:** `faithfulness=` parameter.
- **`Why`:** `faithfulness_tests`, `faithfulness_effects`, `protocol_results`, `faithfulness_protocols`.
- **`Coverage`:** `faithfulness_protocols`, `not_evaluated`.
- **`beyondnn.protocols`:** `COMPREHENSIVENESS`, `SUFFICIENCY`, `DIAGNOSTIC_PROTOCOLS`.

## 8. Schema changes

- `intervention` record version 2: `units`, `retain`, and positional model-input sites. v1 migrates with `units=None`, `retain=False`.
- New kinds (status `None`): `evidence_selection`, `protocol_result`, plus the `CheckOutcome` and `AspectOutcome` values.
- New limitation codes: `SITE_RELATIVE_SUFFICIENCY`, `SELECTION_TIE_AT_BOUNDARY`, `DECLARED_TRANSFORMATION_UNVERIFIED`.
- **No new `EvidenceStatus`.**

## 9. ADRs

- **ADR-030:** status finalized.
- **ADR-031:** status finalized, with the owner's approval of `InputRecord.sample_id` and the privacy limitation kept.
- **ADR-032 (new):** unit, input, and family interventions.
- **ADR-033 (new):** faithfulness records, protocols, controls, statistics, datasets, and coverage.

## 10. Protocol definitions

See `docs/protocols/` (one page per protocol).

- **`comprehensiveness` v1:**
  - `drop = F(x) − F(remove(S))`;
  - SUPPORTS iff `drop ≥ min_drop` (and the declared `fraction_below`);
  - INCONCLUSIVE for a no-op;
  - decides NECESSARY_FOR / DECREASES.
- **`sufficiency` v1:**
  - `drop = F(x) − F(retain(S))` within one site;
  - SUPPORTS iff `drop ≤ max_drop` (and the declared `fraction_above`);
  - decides SUFFICIENT_FOR, and only in that site-relative sense.
- **Curves:** `removal_curve` and `retention_curve` (the four curve names map to two modes); full curve, anchors, optional recorded AOPC normalisation, random-ranking controls.
- **`stability` v1:** four separate aspects under a caller-declared transformation.
- **`counterexample` v1 / `paired_control` v1:** dataset summaries linked to per-sample results.
- **Method diagnostics:** `method_agreement`, `baseline_sensitivity`, `ig_step_sensitivity`.

## 11. Ground-truth tasks

All tasks were defined before running (plan §9):
- `Weighted8`, `ProxyDistractor` (with a declared proxy dataset), `RedundantMax`, `EqualSum4`, `ProbeReadable`, `SaturatingPlus`;
- plus Phase-2 `Additive` and `Interaction`, and Phase-3 `Product` and `Saturating`.

They cover:
- causal vs distractor inputs;
- correlated but unused inputs;
- redundant causes and multiple sufficient mechanisms;
- insufficient single units;
- a readable-but-unused internal unit;
- saturation (misleading gradient);
- interaction (context-dependent necessity).

## 12. Pre-registered scenario results

All were met; see the experiment log for the full table.

| Scenario | Result |
|---|---|
| A | drop 12 exactly; `fraction_below` 0.945 (pre-registered 0.964 ± 0.04); P = 0.060 (pre-registered range [0.006, 0.066]) |
| B | IG selection drop 3 → SUPPORTS; correlation-ranked distractor drop 0 → CONTRADICTS |
| C | redundant x0 drop 0 → CONTRADICTS; both → 3 |
| D | retain {x0} drop 3 → sufficiency CONTRADICTS; remove {x0} drop 1 → SUPPORTS |
| E | prediction PASS; ranking ρ = −1 FAIL; top-1 Jaccard 0 FAIL, reported separately |
| F | held on 3 of 4; counterexample (3, 0) recorded |
| G | refused, at run time and at composition |
| H | 40 random methods: mean superiority within [0.40, 0.60]; IG ≥ 0.95 |
| I | gradient top-1 {x1} drop 0.2 → CONTRADICTS; IG {x0} drop 1.0 → SUPPORTS; agreement ρ = −1 FAIL. **Deviation:** the float32 gradient is exactly 0 (pre-registered ≈ 6e-10), so the ranking is the same. |
| J | measured-only works; faithfulness composition refused |
| K | comprehensiveness CONTRADICTED and sufficiency SUPPORTED for the same {x0}, both in WHY |
| L | readable h1 = x1; drops 6 / 0; site-relative sufficiency flagged |
| M | curves exact (AOPC 109/9 and 26/9) |
| N | baseline sensitivity ρ = −1 FAIL; step sensitivity PASS |
| RQ9 | zero replacement drops 8, 4, 2, 1, 0…; a mean replacement equal to x makes every test INCONCLUSIVE (no-op) |

**The premise test.** Does attaching protocols separate useful from convincing-but-non-causal explanations?
- In B and I, two rankings that look equally plausible (IG vs correlation saliency; gradient vs IG) are separated **only** by the declared comprehensiveness test.
- In K, one selection passes one protocol and fails the other, and BeyondNN shows both.
- **Caveat:** these are cases *designed* to exhibit the effect. They show that the framework *can* expose the difference and does not hide it. They do **not** show that it will on realistic models (see §23).

## 13. Random-baseline results

- **A** (`Weighted8`, top-2 of 8, N = 200, seed 0): 94.5% of matched random pairs dropped less and 5.5% tied (the same pair drawn), giving P = 0.060.
  - This is a real, reported limitation: with 8 units a perfect selection cannot reach P < 0.05, because 1/28 of random pairs coincide with it.
- **H:** random rankings were not better than matched controls (mean superiority within [0.40, 0.60]).
- **Paired controls:** paired differences are exact per sample (tested), and oriented for retention (tested).
- **Reproducibility:** controls reproduce from their seed; they never touch the global RNG (tested); composition re-derives them from the seed.

## 14. Mathematical validation

- **Exact values:** drops, curves, AOPC, paired differences, control fractions, Monte-Carlo p, Spearman, Jaccard, and quantiles, all on hand-computed cases.
- **Declared tolerances:** IG-based selections within `atol` 1e-5; IG step sensitivity within the midpoint bound 2.25/n² + 1e-5.

## 15. Mutation tests

20 mutations (plan §13) on scratch copies; all were caught, each by at least 2 tests:

```
   2  1 selection sample-id check skipped
  11  2 ranking direction reversed
   6  3 controls of a different size
   2  4a controls from the global RNG
   2  4b control seed ignored
  68  5 perturbation not applied
  17  6 retain replaces S instead of its complement
   2  7 faithfulness target not checked in composition
   2  8 site-relative sufficiency limitation dropped
   2  9 paired orientation ignored for retention
   2  10 paired with another sample's controls
   2  11 faithfulness_evaluated without results
   2  12 sufficiency decides NECESSARY_FOR
   2  13 counterexample dropped from the summary
   2  14 no-op reported as CONTRADICTS
   2  15 stability aspects collapsed
   2  16 forged faithfulness result accepted
   3  17 curve k=0 anchor dropped
   2  18 aopc normalisation changed
   2  19 selection re-derivation skipped in composition
   2  20 concepts_validated becomes true
ALL CAUGHT
```

- The first run had **two survivors**, both real test gaps, now fixed:
  - (7) a claim-free curve with another target was accepted, because composition's target check was masked by claim targets;
  - (9) the retention orientation of paired controls was untested.
- An earlier run exposed an invalid mutation anchor (13), which was corrected.

## 16. Performance

Median / p90 ms; evidence computed inside the timed call.

Environment: Python 3.14.3, torch 2.12.0, Darwin arm64, 4 threads; warm-up 1, 10 iterations; median / p90 ms.

| case | varied | value | passes | records | time |
|---|---|---|---|---|---|
| comprehensiveness, 50 controls | units d | 8 | 52 | 289 | 30.1 / 30.6 |
| comprehensiveness, 50 controls | units d | 32 | 52 | 343 | 31.0 / 31.1 |
| comprehensiveness, 50 controls | units d | 128 | 52 | 388 | 31.9 / 32.1 |
| comprehensiveness, d=16 | controls N | 0 | 2 | 18 | 1.3 / 1.3 |
| comprehensiveness, d=16 | controls N | 25 | 27 | 209 | 15.9 / 16.0 |
| comprehensiveness, d=16 | controls N | 100 | 102 | 716 | 67.0 / 67.2 |
| comprehensiveness, d=16 | controls N | 400 | 402 | 2369 | 360.5 / 369.7 |
| removal curve, 10 control rankings, d=16 | curve points | 5 | 45 | 324 | 27.8 / 30.3 |
| removal curve, 10 control rankings, d=16 | curve points | 9 | 89 | 673 | 59.4 / 60.6 |
| removal curve, 10 control rankings, d=16 | curve points | 17 | 177 | 1362 | 135.1 / 141.0 |
| dataset comprehensiveness, 20 controls, d=16 | samples | 1 | 22 | 151 | 13.8 / 13.9 |
| dataset comprehensiveness, 20 controls, d=16 | samples | 4 | 88 | 484 | 57.6 / 58.1 |
| dataset comprehensiveness, 20 controls, d=16 | samples | 16 | 352 | 1816 | 287.9 / 339.8 |
| internal comprehensiveness, 50 controls, d=16 | units d | 16 | 52 | 363 | 40.7 / 41.2 |

- Cost is linear in perturbation passes: about 0.6–0.9 ms and about 6 records per pass.
- It is nearly flat in the number of units, and slightly superlinear at 400 controls.
- No trace copies are made: one recording per test.
- No optimisation was done.

## 17. Backward compatibility

- Phase 1–4 APIs and meanings are unchanged.
- The measured-only render is unchanged (NOT EVALUATED keeps the Phase-4 wording when no faithfulness protocol ran).
- `InterventionRecord` v1 migrates (tested); pre-ADR-031 traces still load and give measured-only responses.
- Captum stays optional; faithfulness never imports it.

## 18. Test counts

| Environment | Result |
|---|---|
| Python 3.10 / 3.12 / 3.14, without Captum | **877 passed, 1 skipped** (the Captum cross-check module, reported) |
| Python 3.10 / 3.12 / 3.14, with Captum 0.9.0 | **898 passed**, 0 skipped |

## 19. Ruff / format / mypy

Ruff and ruff format are clean (92 files). mypy `--strict` is clean with and without Captum (86 source files).

## 20. Package build

- sdist and wheel build.
- **Fresh wheel install** (Python 3.12, torch 2.14), first without and then with Captum:
  - the Phase 1–5 smokes pass;
  - all 5 README examples pass;
  - a faithfulness composition reconstructed from saved traces is identical.

## 21. Optional dependencies

- Base: torch only; no numpy/scipy (statistics are pure Python/torch).
- Captum: optional, and unused by faithfulness.
- Quantus: not a dependency (heavy transitive dependencies; input maps only).

## 22. Known limitations

- **Selection:** vector-shaped sites only (every non-last dimension must be 1); images, sequences, and per-position selection are refused.
- **Scope:** one site per test; instance-level WHY only.
- **Replacement:**
  - zero or explicit replacement only (mean/resample is the caller's tensor);
  - removal/retention inputs are off-distribution and only flagged (no ROAR/ROAD);
  - RQ9 shows the replacement choice can decide the outcome.
- **Sufficiency:** internal sufficiency is site-relative.
- **Stability:** depends entirely on the caller's declared transformation and unit map.
- **Counterexamples:** only over declared sets, with no search. Universally quantified claims are protocol results, not claims.
- **Statistics:** Monte-Carlo p has resolution 1/(N+1) and is limited by coincident controls when there are few units.
- **Research questions:** RQ8 not attempted.
- **Hardware:** CPU-only.

## 23. Scientific weaknesses that remain

1. **All evidence is from tiny analytic models built to exhibit each effect.** The premise ("protocols separate useful from non-causal explanations") is demonstrated, not tested, on realistic models. It could still fail there: for example, if every plausible method passes comprehensiveness because the replacement itself is so destructive.
2. **Replacement dependence.** Comprehensiveness and sufficiency conclusions change with the replacement (RQ9), and there is no principled choice. BeyondNN records the choice but cannot validate it.
3. **Controls are matched only by count and site.** They are not matched by attribution mass, magnitude, or distribution position, so a "better than random" result can reflect selecting large-magnitude units.
4. **Claims from selections.** Claims instantiated from a method's selection are recorded with `ClaimSource.METHOD`. The claim is still "about" what the method chose, which invites selection effects if many methods and ks are tried and only passing ones are reported. Nothing in BeyondNN prevents that garden of forking paths.
5. **No calibration of criteria.** Thresholds (`min_drop`, `max_drop`, fractions) are the caller's; BeyondNN offers no guidance on sensible values.

## 24. Engineering weaknesses that remain

- **One full traced pass per perturbation.** Large N or many points is slow, and traces grow at about 6 records per pass.
- **Persistence:** results that rest on several traces (attributions, stability tests) must be saved and reloaded trace by trace; there is no manifest.
- **Declared internal selections** need `n_units` from the caller (checked against the recorded activation).
- **Cross-trace references** (selection → attribution, diagnostics → attributions) are verified only at composition time, not when a single trace is loaded.

## 25. Release blockers

These are unchanged:
- RB-1: the Code of Conduct contact placeholder.
- RB-2: the security contact placeholder.
- RB-3: GitHub CI has never run.

Nothing was pushed or published.

## 26. Gate decision

**GO WITH EXPLICIT LIMITATIONS**

## 27. Reasons

- **Criteria met:**
  - every pre-registered scenario was met (one explained numerical deviation, I, with the same conclusion);
  - every mutation is caught by at least 2 tests;
  - validation is green everywhere, including the clean wheel with and without Captum.
- **No silent misrepresentation is known:**
  - no new status and no global score;
  - raw effects are kept separate from protocol outcomes and assessments;
  - the site-relative, off-distribution, tie, and declared-transformation caveats are recorded on the results they affect;
  - forged or mismatched evidence is refused on composition.
- **Not a plain GO:**
  - §23.1: the premise is only demonstrated on constructed tasks;
  - §23.2–3: replacement dependence and count-only matched controls are real scientific limits of what these protocols can establish;
  - they are documented and flagged, not solved.

## 28. Before Phase 6

- **Required:**
  - owner review of this report and ADR-032/033;
  - a decision on whether §23.1 needs a realistic-model case study before concepts build on faithfulness results.
- **Recommended:**
  - magnitude-matched controls (§23.3);
  - a pre-registration convention for method/k choices (§23.4).
