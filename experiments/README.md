# Experiments

These are research experiments behind BeyondNN's validation. They are **not needed to use the library.**
- **Layout:** each folder corresponds to one pre-registered stage of the project's development history (see [`docs/README.md`](../docs/README.md#development-history)).
- **Contents:** each folder holds the scripts with their frozen parameters, and `results/` with the committed outputs.
- **Artifacts:** downloads, checkpoints and raw traces go to `artifacts/` (git-ignored).

| folder | subject | key results |
|---|---|---|
| `phase5_5/` | faithfulness protocols on trained MLP, CNN and BERT-tiny | `results/faithfulness_*.json.gz`, `analysis.md` |
| `phase6/` | concept validation on hand-built models with known truth; realistic concepts; SAE case | `results/ground_truth.json`, `realistic_*.json` |
| `phase7/` | audit scenarios (14 pre-registered cases); central audits; concept audits | `results/scenarios.json`, `central_*.json` |
| `phase7_5/` | external validation (InterpBench, Tracr concepts), central held-out, e-SNLI rationales, token probes | `results/hypotheses75.json` and the per-experiment files |
| `phase7_75/` | independent ground truth (TD: compiled Tracr programs with decoys), concept calibration, content-token eligibility (BERT-base), re-audits | `results/hypotheses775.json`, `td_heldout.json`, `central775_*.json` |

**Frozen parameters:** hypotheses and parameters were frozen before held-out runs; the plans are in `docs/PHASE_*_PLAN.md`, and deviations are appended there.

**Running an experiment:** see [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md), which covers environments, downloads and cost.
