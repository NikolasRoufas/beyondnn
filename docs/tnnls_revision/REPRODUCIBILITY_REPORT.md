# Reproducibility report (Stage 4)

- **Scope:** save → reload → integrity → provenance → re-audit → WHY, plus cross-platform re-execution and controlled evidence mutations.
- **Sources:**
  - `experiments/tnnls_revision/results/aggregate.json` (S4-XPLAT, S4-COST);
  - `results/S3_CMP_v2/` (tamper F7, provenance faults F2/F3);
  - Stage 1 `REPRODUCIBILITY_AUDIT.md`.

## 1. Re-execution of the frozen central design (S2-CFG / S4-XPLAT)

| model | checkpoint used | platform | per-sample differences vs frozen macOS results (standing, codes, outcome profile; all 6 claims) |
|---|---|---|---|
| MLP (both sites, 51 + 51) | frozen digest `4c19a8da…`, rebuilt on macOS | Linux x86-64 | **0** |
| CNN (both sites, 60 + 60) | frozen digest `d799d1e8…`, rebuilt on macOS | Linux x86-64 | **0** |
| BERT-tiny (40) | HF `M-FAC/bert-tiny-finetuned-sst2` @ `41ad670` | Linux x86-64 | **0** |
| BERT-base (40) | HF `textattack/bert-base-uncased-SST-2` @ `95f0f6f` | Linux x86-64 | see §5 (added when the 20 shards completed) |
| MLP, earlier Stage 1 run | frozen, macOS | macOS arm64 | 0 (S1-REPRO-A) |

**Training is not bit-reproducible across platforms.**
- `common.train` with seed 0 on Linux x86-64 gave different checkpoint digests than on macOS arm64, for both the MLP and the CNN.
- Re-executing *with the frozen checkpoint* is exact. Retraining is a different checkpoint.
- On the Linux seed-0 checkpoints the central counts happened to be identical to the frozen ones (`ROBUSTNESS_REPORT.md`).

## 2. Save → reload → re-audit (MLP, frozen checkpoint, both sites)

| step | input site | hidden site |
|---|---|---|
| `save_evidence` | 136 s, 1.69 GB | 120 s, 1.97 GB |
| `load_evidence` + re-audit | 235 s | 259 s |
| re-audit from disk identical to the in-memory report | **yes** | **yes** |
| `verify_report(load_report(saved), loaded evidence, plan)` | passed (243 s) | passed (276 s) |

Every load fully re-validates every record (content hashes, references, schema versions). That is why reloading costs more than auditing in memory.

## 3. Controlled mutations

| mutation | detection |
|---|---|
| edited outcome field in 5 saved traces (S3-CMP F7) | `TracePersistenceError` on load: stored id does not match content; the evidence set is refused |
| results from another checkpoint (F2) | excluded (`other_checkpoint`); 5/5 samples NOT_EVALUATED |
| results from another input filed under a declared sample (F3) | excluded as undeclared; 5/5 NOT_EVALUATED |
| PRIMARY results removed (F4) | `primary_untested`; 5/5 NOT_EVALUATED |
| controls removed (F5) | `primary_untested`, `required_protocol_missing`; 20/20 UNSUPPORTED |
| code mutations of the audit (recorded logs, not re-run) | 25/25, 24/24, 19/19, 4/4 killed |

## 4. Software path

- **Suite:** 1080 tests pass on the branch, 1 skipped (Captum optional; the 21 Captum tests pass with Captum).
- **Lint and typing:** ruff, format and `mypy --strict` are clean.
- **Golden workflow:** the golden-workflow test covers trace → attribution → intervention → faithfulness → concept → audit → WHY → save → reload → verify.

## 5. BERT-base re-execution

Filled in after completion; see the final report and `aggregate.json`.

## 6. Remaining gaps

- **No published evidence bundle.** The MLP evidence (3.7 GB for both sites) is too large for the repository. A compressed archive of one or two samples per site, with plan and report, would let readers run `verify_report`. This is not done yet and is a P0 item for the authors.
- **Cross-platform agreement covers audit outputs on fixed checkpoints, not retraining.**
- **Concepts, TD, InterpBench and e-SNLI** were not re-executed in this revision; they remain verified at the artifact level only (Stage 1).
