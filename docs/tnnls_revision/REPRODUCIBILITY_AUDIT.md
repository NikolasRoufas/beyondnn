# Reproducibility audit (TNNLS revision, Stage 1)

- **Date:** 2026-11-03
- **Branch:** `tnnls-revision`
- **Base:** `c9e43ad`
- **Evidence:** `MANUSCRIPT_RESULT_VERIFICATION.csv` (row IDs R01–R36) and `experiments/tnnls_revision/stage1/results/recalculation.json`.

Five levels of reproducibility are kept separate below. Passing one level says nothing about the next.

| level | question | Stage 1 status |
|---|---|---|
| L1 software correctness | Does the library behave as specified on constructed cases? | **Re-executed.** All checks pass (§1). |
| L2 artifact recalculation | Do the reported aggregates follow from the committed per-sample / per-instance records? | **Done for 24 claims, 1 contradiction, 10 summary-only** (§2). |
| L3 deterministic re-audit | Re-derive standings from saved evidence (`verify_report`). | **Not repeatable from the repository.** Traces and reports are not committed; only the outcome "passed" is (§3). |
| L4 model-experiment reproduction | Re-run evidence generation and obtain the same per-sample standings. | **Model A (MLP, both sites) reproduced exactly from scratch on CPU** (§6). Models B, C, D, E, InterpBench and TD were not re-run. |
| L5 scientific validity | Do the results support the claims (independence, dependence, estimands)? | See `SCIENTIFIC_WEAKNESSES.md`. |

## 1. Software checks (L1)

Run locally on macOS arm64, Python 3.12.13 (`uv.lock`), CPU, 2026-11-03.

| command | outcome |
|---|---|
| `uv sync --locked` | ok |
| `uv run pytest -q -W error` | 1080 passed, 1 skipped (Captum optional), 116 s |
| `uv run --with "captum>=0.9,<0.10" pytest -q -W error tests/test_attribution_captum.py` | 21 passed |
| `uv run ruff check .` / `uv run ruff format --check .` | pass on the baseline tree |
| `uv run mypy --strict beyondnn tests` | no issues, 118 files |
| `examples/01_quickstart.py` … `07_save_reload.py` | all exit 0 |
| `tests/test_provenance.py test_serialization.py test_migration_matrix.py test_golden_workflow.py test_audit.py test_record_identity.py` | 184 passed |

Where the behaviours named in the research plan are tested (from reading the tests, not new tests):

| property | where tested |
|---|---|
| evidence admissibility (attribution cannot decide a causal claim) | `test_audit.py`, `test_golden_workflow.py` (`attribution_is_not_intervention`), `test_evidence_status.py` |
| provenance matching (checkpoint digest, sample id) | `test_provenance.py`, `test_fingerprint.py`, `test_declared_model.py`, `test_audit.py` scenarios J/K |
| selection scope / unit eligibility | `test_unit_axes.py`, `test_phase775.py` (ADR-053), `test_estimand.py` |
| intervention semantics (replacement required) | `test_interventions.py`, `test_intervention_units.py`, `test_intervention_schema.py` |
| serialization / version migration | `test_serialization.py`, `test_persistence.py`, `test_migration_matrix.py` |
| integrity checking (tampered result) | `test_audit.py` (tampered scenario A → INTEGRITY_FAILURE) |
| deterministic re-audit | `test_golden_workflow.py`, `test_audit.py`, `test_audit_75.py` (`verify_report`) |
| structured WHY | `test_explain.py`, `test_faithfulness_why.py`, `test_audit_75.py` |

**Limits:**
- Every test case was written by the authors.
- Mutation-testing results (25/25, 24/24, 19/19, 4/4) are recorded logs and were **not re-run** in Stage 1.
- CI covers Linux x86-64, Python 3.10, 3.12 and 3.14.
- Floating-point exact ties can differ across platforms. The repository documents this (`docs/REPRODUCIBILITY.md`, "Cross-platform numerics").

## 2. Artifact recalculation (L2)

Script: `experiments/tnnls_revision/stage1/recalculate_stage1.py`.
- It reads the 24 frozen files listed in `recalculation.json → source_sha256`.
- It uses its own aggregation code, not `evaluate775.py`, `analysis775.py` or `analysis75.py`.
- Run: `python3 experiments/tnnls_revision/stage1/recalculate_stage1.py` (standard library only; under 5 s).

### Results that reproduce exactly

| claim | recalculated |
|---|---|
| central reversal headline (7.75) | **47/52** (per site 7/8, 2/4, 13/14, 1/1, 13/14, 11/11) |
| TD decoys PRIMARY-supported | 0/400, Wilson [0, 0.0095] |
| TD used components PRIMARY-supported | 591/760 = 0.7776 [0.7467, 0.8058]; per component 28–38 of 40 |
| TD zero ablation on decoys | 400/400 |
| TD correlated-program decoys | 0/160 (but see R08b) |
| TD IG-selected decoys | 0/57 (57 of 240 selections) |
| TD attribution-only selection claims | 240/240 UNSUPPORTED |
| TD PRIMARY vs TransformerLens interchange | 1160/1160 decisive instances agree |
| InterpBench (18 cases) | TP 942/942, FP 0/8840, 18 ambiguous; identical in the 7.75 re-audit |
| InterpBench IG-selected unnecessary heads contradicted | 1571/1571; attribution-only 1960/1960 UNSUPPORTED |
| InterpBench per configuration | zero FP 1150/8840 (13.0%); mean FP 116/8840 (1.3%); count-null contradicts 425/431 necessary heads (98.6%) |
| BERT-base D1 / D2 | 11/40 / 6/40; 10/11 D1 supports contain [SEP] (0 [CLS]); 3/11 survive in D2; D2 6/6 and C2 10/11 reversed |
| TD concepts | 4/4 known positives validated; 0/8 negatives validated under any rule |
| e-SNLI | IG F1 0.3425; IG − random 0.1486; own 98.75% bootstrap [0.0307, 0.2698] vs recorded [0.0308, 0.2701]; inter-annotator IoU 0.559 |
| Phase 7 manifest | 11/11 SHA-256 match |

### Discrepancies

1. **Two headline values for the same 52 cases.**
   - Phase 7.5 held-out: **49/52** (`central75_*_heldout.json`; ledger E16).
   - Phase 7.75 re-audit of the same samples: **47/52** (`central775_*_heldout_all.json`; ledger Claim A).
   - The difference comes from audit changes ADR-053/054, made after the Phase 7.5 outcome had been seen (A/input 8/8 → 7/8, C 14/14 → 13/14).
   - The manuscript must say which audit version it reports. It must also say that the 7.75 numbers are a **re-audit of already-seen held-out samples**, not a fresh held-out estimate.
2. **"0/160 exact-copy decoys" (R08b, CONTRADICTED_BY_ARTIFACTS).**
   - In `ho_corr_1/2`, only `y_decoy` (l1_h0; 80 instances) is an exact copy of `x_used`.
   - `tok_copy` (l0_mlp; 80 instances) is a copy of the current token, which is the upstream source of `y_decoy`.
   - The count is right; the description is overstated.
3. **The 47 reversals cannot be attributed to configuration axes from the artifacts.**
   - The per-sample `reversal_axes` field (`central75.py`, line 418) merges ALTERNATIVE and STRESS_TEST reversal findings.
   - Labels such as `threshold` most likely come from the ×1.5 STRESS_TEST threshold. A ×0.5 ALTERNATIVE cannot turn a supported result into a contradicted one if the decision rule is monotone; that is an inference that must be checked in Stage 2.
   - Descriptively, 12/47 reversed samples have only k-involving recorded explanations, and 35/47 have at least one explanation without k.
   - This is not an isolated effect. It is the input that Stage 2 needs.

### Summary-only results (no recomputation possible from per-unit records)

- e-SNLI necessity counts (10/7/3 of 40) and the paired N1 interval.
- Token-OOD and shortcut probes.
- Phase 7 84/280 vs 1/280.
- Phase 7.5 concept K+ rejection.
- Mutation logs.
- `verify_report` outcomes.
- Researcher-workflow log.
- Performance timings.

## 3. Deterministic re-audit (L3)

- Every `central775_*`, `td_heldout` and InterpBench file records `verify_report: passed` and `results_excluded: 0`.
- The traces and audit reports themselves sit in git-ignored `artifacts/` and were **not available**: no local repository existed before this clone, and none was found on the server.
- Re-auditing from saved evidence was **re-executed only for software fixtures** (golden workflow, examples). It was not re-executed for any research experiment.
- **Consequence for the paper:** the claim that audits were re-derived from saved traces with identical output (ledger Claim D) currently rests on recorded logs.
- **Fix:** publish a small, versioned evidence bundle (traces + plan + report) for at least one site per model class, for example as a Zenodo archive. Then a reviewer can run `verify_report` without re-running any model.

## 4. Model-experiment reproduction (L4)

- **Run:** CPU re-run of `experiments/phase7_75/central775.py A heldout --eligibility all`.
- **Isolation:** run in an isolated `git archive` copy of `c9e43ad` (outside the repository), in the pinned Phase 5.5 environment, so no frozen file could be overwritten.
- **Comparison:** each held-out sample's standing and finding codes against `central775_A_heldout_all.json`.
- **Result:** exact reproduction (§6).

Every other model reproduction needs either downloads (BERT-tiny/base, e-SNLI, InterpBench) or substantial CPU or GPU time; BERT-base D is about 19 min per sample on CPU (R32). These belong to Stage 4.

## 5. Reproducibility documentation gaps

1. Phase 7.75 results are not hash-pinned.
2. The manuscript is not versioned in the repository. There is no `paper/`, so the plan's `paper/` versus `paper_tnnls/` separation cannot be set up yet.
3. No archived evidence bundle (§3).
4. `docs/REPRODUCIBILITY.md` gives approximate costs from single runs on one Apple machine; the server environment is not documented.
5. The checkpoints for trained models A and B are re-created by deterministic training. Their digests are recorded, but the checkpoints are not distributed.

## 6. Addendum: model-A CPU reproduction (S1-REPRO-A)

- **Command** (in an isolated `git archive c9e43ad` copy, from `experiments/phase5_5`):
  `uv run --no-project --python 3.12 --with-requirements requirements.txt --with-editable <copy> python ../phase7_75/central775.py A heldout --eligibility all`
- **Environment:** Python 3.12.13, torch 2.14.0, captum 0.9.0, numpy 2.5.3, sklearn 1.9.1, transformers 5.17.0, one torch thread, macOS arm64. This matches the recorded environment.
- **Cost:** 10 min 42 s wall clock (423 s user CPU), including model training.
- **Comparison:** `experiments/tnnls_revision/stage1/compare_reproduction.py`; output in `experiments/tnnls_revision/stage1/results/repro_A_comparison.json`.

| check | frozen | re-run |
|---|---|---|
| checkpoint state digest | `sha256:4c19a8da…1c24` | identical |
| accuracy (train / val / test) | 0.9971 / 0.9561 / 0.9737 | identical |
| results per site (input, net.1) | 3672, 3672 | 3672, 3672 |
| `verify_report` | passed, passed | passed, passed |
| per-sample differences (51 samples × 6 claims × 2 sites) in standing, codes, outcomes, reversal axes, tested | — | **0** |

**What this does establish:** for model A, the deterministic CPU training, the evidence generation and the audit reproduce the frozen per-sample outputs exactly on the same platform class (macOS arm64). That includes the A/input 7/8 and A/net.1 2/4 contributions to 47/52.

**What it does not establish:**
- cross-platform reproduction (Linux x86-64 can differ on exact ties; `docs/REPRODUCIBILITY.md`);
- reproduction of B, C, D, E, TD or InterpBench;
- any scientific validity beyond reproducibility.
