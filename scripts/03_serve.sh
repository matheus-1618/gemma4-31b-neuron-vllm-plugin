#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run inside vllm_gemma4. The validated single-chip mode keeps server MNS=1;
# concurrent clients are queued safely by vLLM.
set -euo pipefail
MODEL="${MODEL:-/root/models/gemma-4-31b-it-text}"
TP="${TP:-4}"
LEN="${LEN:-12288}"
SEG="${SEG:-512}"
BUCKETS="${BUCKETS:-512}"
MNS="${MNS:-1}"
GMU="${GMU:-0.70}"
KV_CAP="${KV_CAP:-0.30}"
PORT="${PORT:-8000}"
APC="${APC:-1}"
[[ "$TP" == 4 ]] || { echo 'single-chip LNC2 mode requires TP=4' >&2; exit 2; }
if [[ "$MNS" != 1 && "${ALLOW_UNSAFE_MNS:-0}" != 1 ]]; then
  echo 'MNS>1 is blocked: vLLM-Neuron 0.21 may mix long prefill and decode. Use MNS=1 or explicitly set ALLOW_UNSAFE_MNS=1 for research.' >&2
  exit 2
fi
python3 - "$GMU" "$KV_CAP" <<'PY'
import sys
for name, raw in zip(('GMU', 'KV_CAP'), sys.argv[1:]):
    value = float(raw)
    if not 0 < value <= 1: raise SystemExit(f'{name} must be in (0,1]')
PY
export NEURON_SKIP_EFA_AFFINITY=1
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/root/neff_cache_gemma4}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-21600}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-21600}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-21600000}"
export VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION="$KV_CAP"
export NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=$(( LEN * 5376 * 2 ))
ADD="{\"neuron_config\":{\"num_batched_tokens_buckets\":[${BUCKETS}],\"num_seqs_buckets\":[${MNS}],\"enable_structured_outputs\":true,\"on_device_sampling_config\":{\"all_greedy\":true}}}"
APC_ARG=--no-enable-prefix-caching
[[ "$APC" == 1 ]] && APC_ARG=--enable-prefix-caching
exec vllm serve "$MODEL" --served-model-name gemma4 \
  --tensor-parallel-size "$TP" --max-model-len "$LEN" \
  --max-num-seqs "$MNS" --max-num-batched-tokens "$SEG" \
  --gpu-memory-utilization "$GMU" --kv-cache-dtype auto \
  --no-async-scheduling "$APC_ARG" --additional-config "$ADD" \
  --host 127.0.0.1 --port "$PORT"
