# Server readiness (TNNLS revision, Stage 1)

- **Checked:** 2026-11-03, about 14:18 and 14:46 UTC.
- **Commands:** read-only diagnostics only. No file was written, no process was signalled, no sudo was used, and nothing ran on a GPU.

## Connection

| item | value |
|---|---|
| SSH succeeded | **YES**, via a cloudflared ProxyCommand to the trycloudflare hostname given in the research plan. The key was supplied explicitly; no credentials are recorded here. |
| local SSH configuration | the existing aliases point to older temporary hostnames and were not used. Temporary tunnel hostnames change when the tunnel restarts. |
| actual host | `interstellar` (`hostname`), user `roufas` |
| OS / kernel | Linux 6.8.0-138-generic, Ubuntu, x86_64 |
| system Python | 3.12.3 |
| CPUs / load | 48 cores; load average 0.20 |
| RAM | 188 GiB total, 177 GiB available; swap 8 GiB (121 MiB used) |

## Filesystems

| mount | size | used | avail | use% | inodes used |
|---|---|---|---|---|---|
| `/` | 48G | 46G | **0** | **100%** | 13% |
| `/data` | 1.6T | 499G | 996G | 34% | 1% |

- `/data/roufas/research` is writable.
- `/data/roufas/hf_cache` exists.
- `/data/roufas/research/beyondnn_tnnls_revision/` does **not** exist yet. It was not created, because no server work is authorised in Stage 1.
- **No BeyondNN checkout, traces or manuscript exist on the server.**
- **Risk:** the root filesystem is still full. Any tool that writes to `~`, `/tmp` or a default cache can fail. Every server job must set `HOME`-adjacent caches, `TMPDIR`, `HF_HOME`, `UV_CACHE_DIR`, `TRITON_HOME` and similar to `/data`.

## GPUs

| GPU | model | memory used | util | compute mode | processes |
|---|---|---|---|---|---|
| 0 | RTX 5000 Ada (32 GB) | 2 MiB | 0% | Default | none |
| 1 | RTX 5000 Ada (32 GB) | 26.9 GB | 100% | Exclusive_Process (MPS) | `nvidia-cuda-mps-server`; `VLLM::EngineCore` (user `roufas`, running about 2.5 h; in tmux sessions `rwa_vllm`, `rwa_e1`, `rwa_pairB5`, from another study) |

- The NVIDIA driver reports 610.57.04, CUDA UMD 13.3.
- **GPU 1** is occupied by an unrelated running job. It also has a history of MPS hangs. **Not available.** The job was not inspected further or touched.
- **GPU 0** is idle. Earlier direct use of GPU 0 (bypassing MPS) was approved only for a different project. **No approval exists for BeyondNN.** An idle GPU at one moment is not evidence that it is safe.
- **PyTorch:** an existing research venv imports torch 2.15.0+cu132 when run CPU-only (`CUDA_VISIBLE_DEVICES=""`). **No CUDA smoke test was run**, because neither GPU is cleared for this project. No BeyondNN environment exists on the server.

## Verdict

**SERVER COMPUTE BLOCKED** for GPU experiments, because:
1. `/` is at 100%;
2. GPU 1 is occupied, and the MPS history is unresolved;
3. GPU 0 has no authorisation for this project;
4. no CUDA smoke test has been run in a BeyondNN environment;
5. no frozen protocol or compute budget exists yet (all Stage 2–4 protocols are unwritten).

**CPU work is feasible** on the server's 48 cores with `/data` storage, once a protocol is approved. Most ESSENTIAL experiments in `REVISION_EXPERIMENT_PLAN.md` (E2.2, E3.2, E4.1–E4.4) are CPU-scale.

### Before any server run

1. Re-check `df -h /`.
2. Set all caches and `TMPDIR` under `/data/roufas/research/beyondnn_tnnls_revision/`.
3. Get an explicit GPU decision from the authors, if a GPU is needed at all.
4. Run a CUDA smoke test in a pinned BeyondNN environment.
5. Commit the protocol and budget.
6. Run jobs in tmux with logs and resumable shards. The tunnel can drop mid-command.

## Update after Stages 2–4 (2026-11-03, about 15:30–16:40 UTC)

- **State at the start:**
  - The vLLM job on GPU 1 had ended (MPS server only, 47 MiB); GPU 0 was idle.
  - `/` was still 100% full; `/data` had 996 GB free.
- **What ran:**
  - All work ran on CPU, in tmux sessions with `TMUX_TMPDIR`, caches and `TMPDIR` under `/data`, at most 41 concurrent single-thread processes (load average ≤ 37).
  - **No GPU was used.**
- **Totals:** 22.7 timed CPU-hours; 6.5 GB written under `/data/roufas/research/beyondnn_tnnls_revision/`.
- **State at the end:** all sessions ended, `/` unchanged at 100%, no other user's process touched.
