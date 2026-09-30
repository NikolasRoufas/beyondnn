# Reproducibility

This page separates two things:
- **Library verification:** minutes, no downloads. Run this first.
- **Research experiment reproduction:** hours to a day, with pinned environments and downloaded models and datasets.

## Supported environment

- **Python:** 3.10, 3.12 and 3.14 (tested). 3.11 and 3.13 are expected to work.
- **PyTorch:** ≥ 2.3, CPU. GPU and other devices are not verified.
- **Captum:** 0.9.x, optional (`beyondnn[captum]`).
- **Package manager:** [uv](https://docs.astral.sh/uv/). `uv.lock` pins the development environment.

## Library verification

```bash
git clone https://github.com/NikolasRoufas/beyondnn.git
cd beyondnn
uv sync                                              # locked dev environment in .venv
uv run pytest -q                                     # the full test suite (about 2 minutes)
uv run pytest -q tests/test_golden_workflow.py       # the permanent scientific-workflow test
uv run pytest -q tests/test_migration_matrix.py      # record migrations
uv run ruff check . && uv run ruff format --check .
uv run mypy                                          # --strict (configured in pyproject.toml)
for f in examples/*.py; do uv run python "$f" > /dev/null && echo "ok $f"; done
```

**With Captum:** `uv sync --extra captum`, then `uv run pytest -q`. The Captum-specific tests are skipped when Captum is absent.

**Other Python versions:** `uv run --python 3.10 pytest -q` (uv downloads that interpreter).

**Benchmarks:** `uv run python benchmarks/bench_audit.py --quick`; see [`benchmarks/README.md`](../benchmarks/README.md). The other `bench_*.py` scripts work the same way.

## How seeds and configurations are recorded

- **Every random choice is seeded and recorded in the evidence itself:** random controls (seed, draws, strategy), bootstrap intervals (seed, draws, level) and sample selection. Nothing uses the global RNG.
- **Every record** carries the model checkpoint digest, the sample identity, the target, the protocol and its version, the replacement identity, the declared eligibility, and the BeyondNN, torch and Python versions.
- **Reports** record the plan's identity and the BeyondNN version and audit rules that produced them.
- **`bnn.audits.verify_report`** re-derives a saved report from saved evidence.

## Research experiments

Scripts, frozen parameters and committed results are in [`experiments/`](../experiments/README.md). Results are committed, so none of this needs re-running to read the evidence.

| experiment | what it validates | environment | downloads | approximate cost (CPU) |
|---|---|---|---|---|
| `phase5_5/run_faithfulness.py` | faithfulness on trained MLP / CNN / BERT-tiny | `experiments/phase5_5/requirements.txt` | scikit-learn datasets; BERT-tiny SST-2 (Hugging Face); SST-2 (GLUE) | minutes (MLP / CNN) to about 30 min (BERT-tiny) |
| `phase6/ground_truth.py` | concept validation on hand-built models | same | none | about 2 min |
| `phase7/scenarios.py` | audit rules on 14 pre-registered scenarios | library only | none | under a minute |
| `phase7_5/external_interpbench.py` | implementation consistency on InterpBench | `experiments/phase7_5/requirements_interpbench.txt` (Python 3.11; Tracr, InterpBench, circuits-benchmark) | InterpBench weights (HF, pinned revision); circuits-benchmark (git, pinned commit) | about 30 min |
| `phase7_75/td_bench.py`, `td_concepts.py` | independent ground truth: compiled Tracr programs | same as InterpBench | none (programs compiled locally) | minutes |
| `phase7_5/esnli.py`, `nlp_probes.py` | human rationales; token OOD; shortcuts | Phase-5.5 env | BERT-base SNLI and SST-2 (HF, pinned); e-SNLI CSVs (pinned commit) | about 3 h |
| `phase7_75/central775.py D ...` | BERT-base SST-2, all tokens vs content tokens | Phase-5.5 env | BERT-base SST-2 (HF, pinned) | **about 10–19 min per sample**; the 40-sample run takes hours (run it in shards) |

**Environments:**
- **Phase-5.5 environment**, from the repository root:

  ```bash
  uv run --no-project --python 3.12 --with-requirements experiments/phase5_5/requirements.txt --with-editable . python experiments/phase7/scenarios.py
  ```

- **InterpBench / Tracr environment** (Python 3.11):

  ```bash
  uv venv --python 3.11 .ibenv
  uv pip install --python .ibenv/bin/python -r experiments/phase7_5/requirements_interpbench.txt -e .
  ```

**Before re-running an experiment:**
- The InterpBench scripts expect the circuits-benchmark checkout and the InterpBench download under `experiments/phase7_5/artifacts/` (git-ignored); the pinned revisions are in the scripts.
- Experiment outputs go to the experiment's `results/` directory. The Phase-7.5 results are pinned by hash (`tests/test_phase75_immutable.py`), so re-running a Phase-7.5 experiment in place changes a pinned file. Write to a copy.
- Plans, frozen policies and deviations for each experiment are in the [development history](README.md#development-history).
