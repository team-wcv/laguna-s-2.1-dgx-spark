#!/usr/bin/env bash
# serve.sh — hardened single-node serving of poolside/Laguna-S-2.1-NVFP4 on a
# NVIDIA DGX Spark (GB10). Defaults are the August 2026 replacement checkpoint
# profile measured on a 128 GB Spark while existing Exo control processes stayed up.
#
# Hardening on top of the card recipe:
#   * preflight gate (memory budget, sysctl drift, weights, port, co-tenant guard)
#   * MemAvailable wait loop before launch (avoids the request_memory() race — a thin-margin
#     launch is what crash-loops GB10 boxes; also softens upstream vllm#46307 profile_run overrun)
#   * persistent TRITON_CACHE_DIR / FLASHINFER_WORKSPACE_BASE on disk (ephemeral cache =
#     ~15-min cold recompile on every restart)
#   * conservative JIT fan-out (uncapped nvcc can exhaust unified memory)
#
# Deliberately NOT set (each a documented footgun — see docs/TUNING.md):
#   * --max-num-seqs stays 1: the August weights leave too little graph/KV memory
#     for the older 32-sequence profile
#   * no min_p / logit_bias: vLLM 400s them under speculation
#   * no --moe-backend / --linear-backend: auto FlashInferCutlass is correct on sm_121 (0.25.1);
#     flashinfer_b12x is a broken, slower opt-in
#   * no --max-cudagraph-capture-size: the one-sequence default produces the small
#     capture set that leaves enough memory for a 96K KV cache
#
# --default-chat-template-kwargs enable_thinking:true: server-wide thinking default per
# the base card's agentic recipe. Per-request chat_template_kwargs
# {"enable_thinking": false} still wins. Cost: thinking burns output budget, so tight
# max_tokens caps can truncate to empty content (measured: a 512-token thinking
# request returned content=None at finish=length).
set -euo pipefail

# --- config (env-overridable) --------------------------------------------------------------
LAGUNA_HOME="${LAGUNA_HOME:-$HOME/laguna-s-2.1}"
VENV="${VENV:-$HOME/venvs/vllm025}"
MODEL_ID="${MODEL_ID:-poolside/Laguna-S-2.1-NVFP4}"
DFLASH_MODEL_ID="${DFLASH_MODEL_ID:-poolside/Laguna-S-2.1-DFlash-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-826aacdf6d8b2699d4e367def6f17c83b06044c2}"
DFLASH_REVISION="${DFLASH_REVISION:-b3b5921a900b9e0a1e27e50bdaeb480692a6d19b}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-7}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-96000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.852}"
LAGUNA_HOST="${LAGUNA_HOST:-0.0.0.0}"             # no auth: expose only on a trusted LAN/Tailnet
LAGUNA_PORT="${LAGUNA_PORT:-8888}"
# MAX_NUM_BATCHED_TOKENS: default 8192 — ADOPTED from our AB matrix (see
#   docs/TUNING.md + docs/PERFORMANCE.md): TTFT −23% @8K, −13% @32K, decode
#   unchanged, KV pool −5.8%. Unset/empty/`none` reverts to the vLLM default,
#   which on GB10 falls into the small-GPU heuristic branch
#   (get_device_total_memory < 70 GiB on unified memory) = 2048, and DFlash
#   the engine's small-GPU heuristic still harms long prefill. 8192 is the
#   verified production value; decode is unchanged by this knob.
# ATTENTION_BACKEND: auto already picks FlashInferBackend on sm_121 — only
#   set to force something else.
# GEN_CONFIG_OVERRIDES: default mirrors the card; note the checkpoint's own
#   generation_config.json already contributes top_k=20 (verified in logs).
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHINFER}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
GEN_CONFIG_OVERRIDES="${GEN_CONFIG_OVERRIDES:-{\"temperature\":0.7,\"top_p\":0.95,\"top_k\":20}}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"             # weights pre-pulled by install.sh; 0 = allow hub lookups
MEM_MARGIN_GIB="${MEM_MARGIN_GIB:-7}"
MEM_WAIT_MAX_ATTEMPTS="${MEM_WAIT_MAX_ATTEMPTS:-120}"
MEM_WAIT_POLL_SECONDS="${MEM_WAIT_POLL_SECONDS:-5}"

# --- environment ----------------------------------------------------------------------------
export HF_HOME HF_HUB_OFFLINE
export CUTE_DSL_ARCH=sm_121a                       # FP4 kernel JIT arch string (card-required)
export MAX_JOBS="${MAX_JOBS:-2}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"
export VLLM_USE_DEEP_GEMM=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/usr/local/cuda/bin:$PATH"            # nvcc for JIT
export TRITON_CACHE_DIR="$LAGUNA_HOME/cache/triton"
export FLASHINFER_WORKSPACE_BASE="$LAGUNA_HOME/cache/flashinfer"
mkdir -p "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

# Resolve immutable HF snapshots instead of trusting whichever revision a mutable
# `main` ref happened to point at during download.
hub_snapshot() {
  local repo="$1" revision="$2"
  printf '%s/hub/models--%s/snapshots/%s' "$HF_HOME" "${repo//\//--}" "$revision"
}
MODEL_PATH="${MODEL_PATH:-$(hub_snapshot "$MODEL_ID" "$MODEL_REVISION")}"
DFLASH_MODEL_PATH="${DFLASH_MODEL_PATH:-$(hub_snapshot "$DFLASH_MODEL_ID" "$DFLASH_REVISION")}"
test -f "$MODEL_PATH/model.safetensors.index.json"
test -f "$DFLASH_MODEL_PATH/model.safetensors"

# --- gate + memory wait ----------------------------------------------------------------------
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  bash "$LAGUNA_HOME/deploy/preflight.sh"
fi

MEMTOTAL_KB="$(awk '/^MemTotal:/{print $2}' /proc/meminfo)"
NEEDED_KB="$(awk -v t="$MEMTOTAL_KB" -v u="$GPU_MEMORY_UTILIZATION" -v m="$MEM_MARGIN_GIB" \
  'BEGIN{printf "%d", t*u + m*1024*1024}')"
echo "== waiting for MemAvailable >= $((NEEDED_KB/1024/1024)) GiB (util=$GPU_MEMORY_UTILIZATION + ${MEM_MARGIN_GIB}GiB margin)"
for i in $(seq 1 "$MEM_WAIT_MAX_ATTEMPTS"); do
  AVAIL_KB="$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)"
  if [ "$AVAIL_KB" -ge "$NEEDED_KB" ]; then
    echo "== memory margin cleared: $((AVAIL_KB/1024/1024)) GiB available (attempt $i/$MEM_WAIT_MAX_ATTEMPTS)"
    break
  fi
  if [ "$i" -eq "$MEM_WAIT_MAX_ATTEMPTS" ]; then
    echo "== WARNING: margin never cleared after $MEM_WAIT_MAX_ATTEMPTS attempts; proceeding — vLLM's request_memory() will likely fail cleanly"
  fi
  sleep "$MEM_WAIT_POLL_SECONDS"
done

# --- serve (model card's DGX Spark recipe) ---------------------------------------
echo "== starting vllm serve $MODEL_ID@$MODEL_REVISION on $LAGUNA_HOST:$LAGUNA_PORT"
echo "   draft: $DFLASH_MODEL_ID@$DFLASH_REVISION"
echo "   first start reads a 92.85 GiB checkpoint and can take 15–20 min"
EXTRA_ARGS=()
[ -n "$MAX_NUM_BATCHED_TOKENS" ] && [ "$MAX_NUM_BATCHED_TOKENS" != "none" ] \
  && EXTRA_ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
[ -n "$ATTENTION_BACKEND" ] && EXTRA_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
exec "$VENV/bin/vllm" serve "$MODEL_PATH" \
  --served-model-name "$MODEL_ID" \
  --trust-remote-code \
  --speculative-config "{\"model\":\"$DFLASH_MODEL_PATH\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS,\"method\":\"dflash\"}" \
  --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --override-generation-config "$GEN_CONFIG_OVERRIDES" \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --enable-prefix-caching \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --host "$LAGUNA_HOST" --port "$LAGUNA_PORT" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
