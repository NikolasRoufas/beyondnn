# Benchmarks

`bench_trace_overhead.py` characterises Phase-1 tracing cost on CPU. It uses only the internal reference models and a synthetic ~1M-parameter dense model: no datasets and no downloads.

```bash
python benchmarks/bench_trace_overhead.py              # 3 warm-up + 30 measured iterations
python benchmarks/bench_trace_overhead.py --quick      # smoke run
python benchmarks/bench_trace_overhead.py --json out.json
```

`bench_interventions.py` characterises Phase-2 comparison cost. For TinyMLP and TinyTransformer it times a plain forward, a baseline trace, a zero-ablation comparison, and a patch comparison. It uses the same flags (`--quick`, `--json`).

`bench_attribution.py` characterises Phase-3 attribution cost: forward, trace, native gradient, input×gradient, and IG (16/64 steps), plus Captum IG when Captum is installed. It covers an analytic model, TinyMLP, TinyCNN, and TinyTransformer (embedding-layer attribution). It uses the same flags.

- **Timing:** everything runs under `torch.no_grad()`. Times are the median and p90 of wall-clock milliseconds.
- **Sizes:** exact byte counts (retained tensor bytes, `trace.json`, `tensors.pt`).
- **Memory:** no `tracemalloc` figures are reported, because it misses native PyTorch allocations.

Results are logged, with methodology, in `docs/experiments/EXPERIMENT_LOG.md`. This is characterisation, **not** a pass/fail gate. Numbers depend on the machine; rerun rather than reuse them.

`bench_synthesis.py` characterises Phase-4 composition and rendering (no model runs in the timed region; the evidence is computed beforehand): measured-only, plus attribution, plus intervention, and everything plus claims. It uses the same flags.

`bench_faithfulness.py` characterises Phase-5 faithfulness tests: time, passes, and records as the number of units, random controls, curve points, and dataset samples grows (input level, plus one internal-site case). It uses the same flags.

`bench_audit.py` characterises Phase-7 audits: `bnn.audit` (integrity, re-derivation, classification), `to_json`, and `verify_report` (a full re-audit), for comprehensiveness evidence on 1, 4 and 16 samples plus one concept body. It uses the same flags. The evidence is computed beforehand and not timed. The realistic-scale audit measurements (MLP, CNN, BERT-tiny) are in `experiments/phase7/results/performance.json`.
