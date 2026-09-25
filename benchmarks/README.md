# Benchmarks

`bench_trace_overhead.py` characterises Phase-1 tracing cost on CPU. It uses only the internal reference models and a synthetic ~1M-parameter dense model: no datasets and no downloads.

```bash
python benchmarks/bench_trace_overhead.py              # 3 warm-up + 30 measured iterations
python benchmarks/bench_trace_overhead.py --quick      # smoke run
python benchmarks/bench_trace_overhead.py --json out.json
```

- **Timing:** everything runs under `torch.no_grad()`. Times are the median and p90 of wall-clock milliseconds.
- **Sizes:** exact byte counts (retained tensor bytes, `trace.json`, `tensors.pt`).
- **Memory:** no `tracemalloc` figures are reported, because it misses native PyTorch allocations.

Results are logged, with methodology, in `docs/experiments/EXPERIMENT_LOG.md`. This is characterisation, **not** a pass/fail gate. Numbers depend on the machine; rerun rather than reuse them.
