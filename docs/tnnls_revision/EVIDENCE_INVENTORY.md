# Evidence inventory (TNNLS revision, Stage 1)

- **Date:** 2026-11-03
- **Scope:** what evidence exists in the repository, where it came from, and what is missing.
- **Method:** files were read directly. Headline counts were recomputed by `experiments/tnnls_revision/stage1/recalculate_stage1.py`. That script uses the standard library only and none of the repository's aggregation helpers. Its output is in `experiments/tnnls_revision/stage1/results/recalculation.json`, which records the SHA-256 of each of its 24 source files.
- **Not done:** no model was run, and no frozen file was changed.

## 1. Baseline

| item | value |
|---|---|
| repository | `https://github.com/NikolasRoufas/beyondnn`, cloned 2026-11-03 into `~/beyondnn` |
| local state before cloning | **`~/beyondnn` did not exist on this Mac.** It was not found anywhere else on disk or on the research server. The public clone is therefore the baseline. Any unpushed local work, and git-ignored `artifacts/` (checkpoints, traces, downloads), are not available. |
| starting commit | `c9e43ad7588ad6fe3119adc249e7ca739e978542` (`main`, "Finalize BeyondNN v0.1.0 post-release documentation") |
| release tag | `v0.1.0` → `885e3ec`. Between the tag and HEAD, only README and release docs changed (5 files). |
| PyPI | `beyondnn 0.1.0`, uploaded 2026-10-01 |
| revision branch | `tnnls-revision`, created from `c9e43ad` |
| package version | `0.1.0` (`beyondnn/__init__.py`, hatch dynamic version) |
| local verification env | Python 3.12.13 (uv-managed); torch 2.14.1, numpy 2.5.3, pytest 9.1.1, ruff 0.16.9, mypy 2.3.1, uv 0.11.23; macOS arm64 |
| recorded experiment envs | Phase-5.5 env: Python 3.12.13, torch 2.14.0, captum 0.9.0, transformers 5.17.0, sklearn 1.9.1, numpy 2.5.3, one torch thread, macOS arm64. TD/InterpBench env: Python 3.11.15, torch 2.3.1, transformer_lens 1.19.0. |

### Frozen research commits

| commit | date | what was frozen |
|---|---|---|
| `ee3f91e` | 2026-09-26 | Phase 7.5 plan and frozen evaluation policy (`docs/PHASE_7_5_FROZEN_POLICY.md`), before any held-out evaluation |
| `b355129` | 2026-09-29 | Phase 7.75 frozen plan (`docs/PHASE_7_75_PLAN.md`), before any held-out 7.75 run; Phase 7.5 results hash-pinned |
| `a5fd344` | 2026-09-29 | TD held-out results |
| `bb303eb` | 2026-09-30 | BERT-base D1/D2 held-out (10+10 shards merged) |
| `a156d7e` | 2026-09-30 | deviation DV-1 (crash; resharding; outcome-blind) |
| `885e3ec` | 2026-10-01 | v0.1.0 tag |

### Integrity of frozen results

- Phase 7: all 11 files in `experiments/phase7/results/RESULT_MANIFEST.json` match their recorded SHA-256 (checked 2026-11-03).
- Phase 7.5: hash-pinned by `tests/test_phase75_immutable.py`, which passed.
- **Phase 7.75: no hash pin exists.** The Stage 1 recalculation records the current hashes. Pinning these files is recommended (P0 in `REVISION_EXPERIMENT_PLAN.md`).

## 2. What exists

### 2.1 Library (software)

- **Package and tests:** `beyondnn/` and `tests/`; mypy `--strict` checks 118 files across both.
- **Decisions:** 56 ADR sections in `docs/decisions/ARCHITECTURE_DECISIONS.md`.
- **Protocols:** `docs/protocols/`.
- **Audit taxonomy:** `docs/audit/`.
- **API freeze:** `docs/API_FREEZE.md`.
- **Invariants:** `docs/PRE_PHASE8_INVARIANTS.md`.

### 2.2 Experiments and committed results

| phase | scripts | committed results | models | split discipline |
|---|---|---|---|---|
| 5.5 | `experiments/phase5_5/*.py` | `faithfulness_*.json.gz`, `diagnostics_*.json.gz` | A MLP/breast_cancer, B CNN/digits (both trained, seed 0), C BERT-tiny/SST-2 (HF pinned) | pre-registered (`PHASE_5_5_PLAN.md`) |
| 6 | `experiments/phase6/*.py` | `ground_truth.json`, `realistic_*.json` | hand-built + A/B/C | pre-registered |
| 7 | `experiments/phase7/*.py` | `scenarios.json`, `central_{A,B,C}.json`, `concepts_*.json`, `mutations.json`, `nlp.json`, `performance.json`, `analysis.json`, `RESULT_MANIFEST.json` | A, B, C (60/60/40 samples) | pre-registered; deviations D1–D13 |
| 7.5 | `experiments/phase7_5/*.py` | `central75_{A,B,C,D}_{dev,heldout}.json`, `external_interpbench_*.json`, `external_tracr_concepts_*.json`, `esnli.json`, `nlp_probes_{C,D,E}.json`, `mutations75.json`, `performance75.json`, `hypotheses75.json`, `figure_data/fd1–fd9`, `reruns/` | A, B, C, D BERT-base SST-2, E BERT-base SNLI; 18 InterpBench SIIT models | dev/held-out; frozen policy `ee3f91e`; DV-1 |
| 7.75 | `experiments/phase7_75/*.py` | `td_{dev,heldout}.json`, `td_concepts_*.json`, `concept_calibration.json`, `central775_*_heldout_{all,content_tokens}.json` (+20 D shards), `interpbench_*.json`, `nli_strata.json`, `mutations775.json`, `mutations_pre8.json`, `hypotheses775.json`, `figure_data/fd10–fd16`, `performance/bench_audit_*.json` | 8 held-out compiled Tracr TD programs; 4 TD concept programs; A–D re-audited | frozen plan `b355129`; DV-1 |

### 2.3 Prior research documentation

- `docs/research/PAPER_EVIDENCE_LEDGER.md`: allowed and disallowed wording per claim.
- `PHASE_7_75_PAPER_READINESS.md`: NeurIPS/ACL readiness, written before the TNNLS target.
- `ECOSYSTEM_AUDIT.md`, `DIFFERENTIATION.md`, `PHASE_7_DIFFERENTIATION.md`, `PHASE_7*_LITERATURE.md`: related work as of September 2026.
- `PHASE_7_75_INDEPENDENT_GROUND_TRUTH.md`.

## 3. What the per-result records allow

Each audit result file stores per-sample (or per-instance) **audit outputs**: standing, finding codes, per-configuration outcomes and reversal-axis labels. They do **not** store the underlying traces or reports. Consequences:

- **Can be recomputed without a model:** every aggregate over per-sample or per-instance standings and configuration outcomes. That covers 47/52, all TD counts, the InterpBench confusion and per-configuration rates, the special-token counts and the concept outcomes.
- **Can be re-derived from evidence only by regenerating traces:** the standings themselves, via `verify_report` and deterministic re-audit. The committed files contain the *outcome* of `verify_report` ("passed"), not the material needed to repeat it.
- **Requires re-running models:** timings, memory, evidence generation, any new configuration, and any check of whether a recorded axis label isolates an effect.

## 4. What is missing

| missing item | impact |
|---|---|
| **The manuscript LaTeX source and Supplementary Appendices A–D** | The compiled PDF (16 pages, TNNLS format, SHA-256 `0471417f…a1fa60`) was supplied later on 2026-11-03 and has been checked (`MANUSCRIPT_CONSISTENCY.md`). It is not committed, because the repository is public. The LaTeX source and the supplementary appendices are still missing. There is no `paper/` directory. |
| Saved traces and audit reports for research runs | Deterministic re-audit of the research experiments cannot be repeated from the repository. |
| Trained checkpoints A/B | Retraining is deterministic on CPU (`common.train`, seed 0, digest-verified), but this was not executed in Stage 1. |
| Seed or checkpoint replicates | None exist: one checkpoint per model. |
| Baseline-workflow comparison | None exists: no Captum/Quantus-only workflow was run on the same evidence. |
| With/without-audit overhead comparison | `fd9_performance.csv` times the BeyondNN workflow only; one run per setting. |
| Hash pin for Phase 7.75 results | See §1. |

## 5. Software checks run on 2026-11-03 (local, CPU)

| check | result |
|---|---|
| `uv sync --locked` | ok |
| `uv run pytest -q -W error` | **1080 passed, 1 skipped** (Captum absent), 116 s |
| Captum tests (`--with captum<0.10`) | 21 passed |
| `uv run ruff check .` / `ruff format --check .` | pass on the baseline tree (the first failure was in the new Stage 1 script only, since fixed) |
| `uv run mypy --strict beyondnn tests` | no issues (118 files) |
| `examples/01–07` | all run |

Passing tests establish software behaviour on constructed cases. They are not independent scientific validation.
