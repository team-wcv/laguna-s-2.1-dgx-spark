# laguna-s-2.1-dgx-spark

Serve [`poolside/Laguna-S-2.1-NVFP4`](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4)
(117.6B MoE / 8.5B active, native 1M context, NVFP4, DFlash speculative decoding) with **vLLM on a
single NVIDIA DGX Spark (GB10)** — bare-metal, no sudo, with a systemd user service, an
inference watchdog, and a measured tuning recipe.

**Tested on 1x NVIDIA DGX Spark (GB10); deployed and operated from a Mac over SSH.**
Scope is deliberately single-node: no multi-node/TP=2 content, no multi-model sidecars.

Poolside replaced the NVFP4 weights in August 2026. This fork pins the replacement target
and current DFlash draft by immutable revision instead of downloading mutable `main`. The
replacement target is materially larger than the July checkpoint: vLLM accounted for a
92.85 GiB checkpoint and loaded target + draft at 95.61 GiB on the measured Spark. The
older 256K / 32-sequence profile no longer fits safely on one 128 GB box.

## Deployment specs

| Component | Value |
|---|---|
| Model | `poolside/Laguna-S-2.1-NVFP4` — 117.6B total / 8.5B active MoE, 256 routed experts top-10, 48 layers (36 sliding-window + 12 global) |
| Target revision | `826aacdf6d8b2699d4e367def6f17c83b06044c2` (August 25, 2026 replacement) |
| Draft revision | `b3b5921a900b9e0a1e27e50bdaeb480692a6d19b` |
| Measured load | 92.85 GiB target checkpoint; 95.61 GiB target + draft in vLLM |
| Context | 96,000 served; 1,048,576 native model configuration does not fit with this target/draft on one 128 GB Spark |
| Engine | vLLM **0.25.1** (`--torch-backend=cu130`, aarch64 PyPI wheels) |
| Kernels | FlashInfer nightly trio **0.6.15.dev20260712** (`flashinfer-python/-cubin/-jit-cache`) |
| Python | uv-managed CPython **3.12** (bundles `Python.h` — no sudo, no `apt install python3.12-dev`) |
| Deployment | Bare-metal uv venv on the Spark; systemd **user** service + 5-min watchdog timer |
| Spec decode | DFlash, 7 speculative tokens; 15 was AB-tested and rejected on the August weights |
| Hardware | 1x DGX Spark: GB10 Grace Blackwell, aarch64, 121 GiB unified memory, CUDA 13 |

## Measured performance (August replacement checkpoint)

| Metric | Value |
|---|---|
| Decode, single stream, 512-token code probes | **22.82 · 21.19 tok/s** (22.0 weighted average) |
| DFlash K=7 acceptance | 3.52 accepted tokens/step; 50.3% drafted-token acceptance |
| TTFT | 1.02–1.51 s on the final two code probes |
| KV pool | 111,449 tokens at util 0.852; 1.16× headroom for a 96K request |
| Functional gates | exact content, `poolside_v1` tool calls, separated reasoning, Tailnet API |
| Cold start | about 16–18 min to read/load the 95.61 GiB target + draft |

Full current and historical tables, rejected profiles, and method:
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Quickstart

Prerequisites: a DGX Spark on DGX OS / Ubuntu 24.04 with CUDA 13, ~150 GB free disk, SSH
access from your Mac, and HF access to the (possibly gated) model repos.

```bash
# Mac → Spark: put this repo at ~/laguna-s-2.1 on the Spark
rsync -av laguna-s-2.1-dgx-spark/ your-spark-host:~/laguna-s-2.1/

# On the Spark (interactive: apt-free; downloads immutable target/draft revisions)
ssh -t your-spark-host 'bash ~/laguna-s-2.1/deploy/install.sh'     # ~15 min + download
ssh -t your-spark-host 'bash ~/laguna-s-2.1/deploy/serve.sh'       # foreground first run, ~15 min cold start
ssh your-spark-host 'bash ~/laguna-s-2.1/deploy/smoke-test.sh'     # 7-check acceptance gate

# Then run it as a systemd user service (+ watchdog timer)
ssh your-spark-host 'mkdir -p ~/.config/systemd/user &&
  cp ~/laguna-s-2.1/deploy/vllm-laguna.service ~/laguna-s-2.1/deploy/vllm-laguna-watchdog.* ~/.config/systemd/user/ &&
  systemctl --user daemon-reload &&
  systemctl --user start vllm-laguna.service vllm-laguna-watchdog.timer'
```

Full walkthrough incl. boot auto-start (linger), day-2 ops and rollback:
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Repo map

- `deploy/install.sh` — one-time setup: uv + managed CPython 3.12, vLLM 0.25.1 cu130,
  pinned FlashInfer nightly trio and immutable August target/draft pulls. Idempotent, no sudo.
- `deploy/preflight.sh` — 8 boot guards (venv, JIT headers, GPU, weights, memory budget,
  sysctl drift, port, co-tenant). `FAIL` refuses to start; `FORCE=1` overrides the two marked guards.
- `deploy/serve.sh` — the hardened serve: card recipe flags + `MAX_NUM_BATCHED_TOKENS=8192`
  default, persistent JIT caches, MemAvailable wait loop, all knobs env-overridable.
- `deploy/smoke-test.sh` — 7-check gate: canary, chat, `poolside_v1` tool-call parser,
  thinking parser, DFlash acceptance, concurrency-3, ~6K-prefill probe.
- `deploy/warmup.sh` — post-start primer (ExecStartPost): chat + tool-call + >4K prefill.
- `deploy/watchdog.sh` + `vllm-laguna-watchdog.{service,timer}` — 5-min inference watchdog:
  tagged 1-token canary, KV-saturation triage (a busy engine is not a wedged engine),
  single-unit restart.
- `deploy/vllm-laguna.service` — systemd user unit (preflight gate, warmup post-start,
  30-min start timeout for the cold boot).
- `bench/` — `bench.py`, the stdlib-only on-node bench client used to produce every
  number in the docs, plus a short guide to measuring your own deployment.
- `docs/` — [DEPLOYMENT](docs/DEPLOYMENT.md) · [TUNING](docs/TUNING.md) ·
  [PERFORMANCE](docs/PERFORMANCE.md).

## License

- **This repo** (scripts + docs): [MIT](LICENSE).
- **The model** is separate: `poolside/Laguna-S-2.1-NVFP4` is **OpenMDW-1.1** (commercial
  use allowed) + the Poolside Acceptable Use Policy — accept it on Hugging Face before pulling.
