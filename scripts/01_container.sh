#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Create a single-chip public vLLM-Neuron container. This removes an existing
# container with the same name; model and compile caches remain on the host.
set -euo pipefail
IMG="${IMG:-public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04}"
NAME="${NAME:-vllm_gemma4}"
REPO_DIR="${REPO_DIR:-$HOME/gemma4-31b-neuron-vllm-plugin}"
DEVICE="${DEVICE:-/dev/neuron0}"
MEM="${MEM:-128g}"
CPUSET="${CPUSET:-}"
mkdir -p "$HOME/models" "$HOME/neff_cache_gemma4" "$HOME/hf_cache"
sudo docker pull "$IMG"
sudo docker rm -f "$NAME" 2>/dev/null || true
args=(
  -d --name "$NAME" --device "$DEVICE" --memory "$MEM"
  --cap-add SYS_ADMIN --cap-add IPC_LOCK --ipc=host --network host
  -v "$HOME/models:/root/models"
  -v "$HOME/neff_cache_gemma4:/root/neff_cache_gemma4"
  -v "$HOME/hf_cache:/root/hf_cache"
  -v "$REPO_DIR:/workspace/gemma4-port:ro"
  -e HF_HOME=/root/hf_cache
  -e VLLM_CACHE_ROOT=/root/neff_cache_gemma4
  -e NEURON_SKIP_EFA_AFFINITY=1
)
if [[ -n "$CPUSET" ]]; then args+=(--cpuset-cpus "$CPUSET"); fi
sudo docker run "${args[@]}" "$IMG" sleep infinity
sudo docker inspect "$NAME" --format 'cpuset={{.HostConfig.CpusetCpus}} mem={{.HostConfig.Memory}} devices={{json .HostConfig.Devices}}'
