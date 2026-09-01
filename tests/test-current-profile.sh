#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

TARGET_SHA=826aacdf6d8b2699d4e367def6f17c83b06044c2
DRAFT_SHA=b3b5921a900b9e0a1e27e50bdaeb480692a6d19b
TARGET="$TMP/hf/hub/models--poolside--Laguna-S-2.1-NVFP4/snapshots/$TARGET_SHA"
DRAFT="$TMP/hf/hub/models--poolside--Laguna-S-2.1-DFlash-NVFP4/snapshots/$DRAFT_SHA"
mkdir -p "$TARGET" "$DRAFT" "$TMP/venv/bin" "$TMP/bin" "$TMP/state"
touch "$TARGET/model.safetensors.index.json" "$DRAFT/model.safetensors"

cat >"$TMP/venv/bin/vllm" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@"
EOF
chmod +x "$TMP/venv/bin/vllm"

# macOS has no /proc/meminfo. Return a synthetic 128 GiB host for the two
# memory reads and delegate the utilization calculation to the system awk.
cat >"$TMP/bin/awk" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *MemTotal*'/proc/meminfo'*) printf '%s\n' 134217728 ;;
  *MemAvailable*'/proc/meminfo'*) printf '%s\n' 134217728 ;;
  *) exec /usr/bin/awk "$@" ;;
esac
EOF
chmod +x "$TMP/bin/awk"

output="$({
  PATH="$TMP/bin:$PATH" \
  LAGUNA_HOME="$TMP/state" \
  VENV="$TMP/venv" \
  HF_HOME="$TMP/hf" \
  SKIP_PREFLIGHT=1 \
  bash "$ROOT/deploy/serve.sh"
} 2>&1)"

require_line() {
  grep -Fx -- "$1" <<<"$output" >/dev/null || {
    printf 'missing expected argument: %s\n%s\n' "$1" "$output" >&2
    exit 1
  }
}

require_line "$TARGET"
require_line poolside/Laguna-S-2.1-NVFP4
require_line '--attention-backend'
require_line FLASHINFER
require_line '--kv-cache-dtype'
require_line fp8
require_line '--enable-prefix-caching'
require_line '--max-num-seqs'
require_line 1
require_line '--max-model-len'
require_line 96000
require_line '--gpu-memory-utilization'
require_line 0.852
require_line '--host'
require_line 0.0.0.0
require_line '--port'
require_line 8888
require_line "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":7,\"method\":\"dflash\"}"

printf 'current profile arguments verified\n'
