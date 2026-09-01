#!/usr/bin/env bash
# preflight.sh — bounded boot guards for single-node Laguna serving on GB10.
#
# FAIL = refuse to start (exit 1). WARN = start anyway.
# FORCE=1 overrides the two guards marked [FORCE-able].
set -uo pipefail

LAGUNA_HOME="${LAGUNA_HOME:-$HOME/laguna-s-2.1}"
VENV="${VENV:-$HOME/venvs/vllm025}"
MODEL_ID="${MODEL_ID:-poolside/Laguna-S-2.1-NVFP4}"
DFLASH_MODEL_ID="${DFLASH_MODEL_ID:-poolside/Laguna-S-2.1-DFlash-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-826aacdf6d8b2699d4e367def6f17c83b06044c2}"
DFLASH_REVISION="${DFLASH_REVISION:-b3b5921a900b9e0a1e27e50bdaeb480692a6d19b}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.852}"
MEM_MARGIN_GIB="${MEM_MARGIN_GIB:-7}"
LAGUNA_PORT="${LAGUNA_PORT:-8888}"
SYSCTL_CONF="${SYSCTL_CONF:-/etc/sysctl.d/90-laguna-oom.conf}"

fail() { echo "preflight FAIL: $*" >&2; exit 1; }
warn() { echo "preflight WARN: $*" >&2; }
ok()   { echo "preflight ok: $*"; }

# 1. venv installed
[ -x "$VENV/bin/vllm" ] || fail "vllm not found at $VENV/bin/vllm — run deploy/install.sh first"
ok "venv present ($VENV)"

# 2. Python headers for JIT (non-fatal: only matters on a cold kernel cache).
#    install.sh uses a uv-managed CPython that bundles its own headers — check there.
if ! "$VENV/bin/python" -c 'import os,sysconfig,sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()["include"],"Python.h")) else 1)' 2>/dev/null; then
  warn "Python.h missing for $VENV/bin/python — Triton/FlashInfer JIT fails on a cold cache (re-run deploy/install.sh, or sudo apt-get install -y python3.12-dev)"
fi

# 3. GPU visible
if nvidia-smi -L >/dev/null 2>&1; then
  ok "GPU visible: $(nvidia-smi -L | head -n1)"
else
  warn "nvidia-smi not answering"
fi

# 4. Exact immutable target and draft snapshots are present.
hub_snapshot() {
  local repo="$1" revision="$2"
  printf '%s/hub/models--%s/snapshots/%s' "$HF_HOME" "${repo//\//--}" "$revision"
}
MODEL_PATH="${MODEL_PATH:-$(hub_snapshot "$MODEL_ID" "$MODEL_REVISION")}"
DFLASH_MODEL_PATH="${DFLASH_MODEL_PATH:-$(hub_snapshot "$DFLASH_MODEL_ID" "$DFLASH_REVISION")}"
[ -f "$MODEL_PATH/config.json" ] && [ -f "$MODEL_PATH/model.safetensors.index.json" ] \
  || fail "target snapshot missing or incomplete: $MODEL_ID@$MODEL_REVISION ($MODEL_PATH)"
[ -f "$DFLASH_MODEL_PATH/config.json" ] && [ -f "$DFLASH_MODEL_PATH/model.safetensors" ] \
  || fail "draft snapshot missing or incomplete: $DFLASH_MODEL_ID@$DFLASH_REVISION ($DFLASH_MODEL_PATH)"
ok "target snapshot present: $MODEL_ID@$MODEL_REVISION"
ok "draft snapshot present: $DFLASH_MODEL_ID@$DFLASH_REVISION"

# 5. [FORCE-able] Memory budget: MemAvailable >= util*MemTotal + margin.
#    Unified memory is exclusive — a thin-margin start is what crash-loops GB10 boxes.
MEMTOTAL_KB="$(awk '/^MemTotal:/{print $2}' /proc/meminfo)"
AVAIL_KB="$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)"
NEEDED_KB="$(awk -v t="$MEMTOTAL_KB" -v u="$GPU_MEMORY_UTILIZATION" -v m="$MEM_MARGIN_GIB" \
  'BEGIN{printf "%d", t*u + m*1024*1024}')"
if [ "$AVAIL_KB" -lt "$NEEDED_KB" ]; then
  msg="MemAvailable $((AVAIL_KB/1024/1024)) GiB < required $((NEEDED_KB/1024/1024)) GiB (util $GPU_MEMORY_UTILIZATION + ${MEM_MARGIN_GIB} GiB margin) — is another model server still holding memory?"
  if [ "${FORCE:-0}" = "1" ]; then warn "$msg — FORCE=1, continuing"; else fail "$msg (FORCE=1 to override)"; fi
else
  ok "memory budget: $((AVAIL_KB/1024/1024)) GiB available >= $((NEEDED_KB/1024/1024)) GiB required"
fi

# 6. vm.min_free_kbytes runtime vs persisted — drift changes memory behavior after reboot
#    (WARN only, a wrong-but-known value is the admin's call). See docs/TUNING.md for
#    why 2097152 (2 GiB) is the value we run.
runtime="$(sysctl -n vm.min_free_kbytes 2>/dev/null || true)"
persisted="$(grep -E '^[[:space:]]*vm\.min_free_kbytes[[:space:]]*=' "$SYSCTL_CONF" 2>/dev/null \
  | tail -n1 | sed -E 's/^[[:space:]]*vm\.min_free_kbytes[[:space:]]*=[[:space:]]*([0-9]+)[[:space:]]*$/\1/' || true)"
if ! printf '%s' "$runtime" | grep -qE '^[0-9]+$' || ! printf '%s' "$persisted" | grep -qE '^[0-9]+$'; then
  warn "vm.min_free_kbytes runtime (${runtime:-<unreadable>}) vs persisted $SYSCTL_CONF (${persisted:-<missing>}) — could not verify"
elif [ "$runtime" != "$persisted" ]; then
  warn "vm.min_free_kbytes runtime ($runtime) != persisted ($persisted) — next reboot changes memory behavior"
else
  ok "vm.min_free_kbytes runtime matches persisted ($runtime)"
fi

# 7. API port free
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${LAGUNA_PORT}\$"; then
  fail "port $LAGUNA_PORT is already listening — another server is up (LAGUNA_PORT to relocate)"
fi
ok "port $LAGUNA_PORT free"

# 8. [FORCE-able] Co-tenant guard: unified memory cannot host two LLM servers.
#    Docker container check is best-effort (the serving user may lack docker access):
#    any running container whose name or image looks like a vLLM server blocks start.
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  cotenant="$(docker ps --format '{{.Names}} {{.Image}}' 2>/dev/null | grep -i 'vllm' || true)"
  if [ -n "$cotenant" ]; then
    if [ "${FORCE:-0}" = "1" ]; then
      warn "another vLLM server appears to be running in docker ($cotenant) — FORCE=1, continuing"
    else
      fail "another vLLM server appears to be running in docker ($cotenant) — stop it first or FORCE=1"
    fi
  fi
fi
if pgrep -f 'vllm serve' >/dev/null 2>&1; then
  if [ "${FORCE:-0}" = "1" ]; then
    warn "another 'vllm serve' process is running — FORCE=1, continuing"
  else
    fail "another 'vllm serve' process is running ($(pgrep -f 'vllm serve' | tr '\n' ' ')) — stop it first or FORCE=1"
  fi
fi
if pgrep -x llama-server >/dev/null 2>&1; then
  warn "llama-server is running — it holds unified memory and will shrink vLLM's budget"
fi

echo "preflight passed"
