#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
NAME="${NAME:-vllm_gemma4}"
sudo docker exec "$NAME" python3 /workspace/gemma4-port/scripts/install_plugin.py
sudo docker exec "$NAME" python3 /workspace/gemma4-port/scripts/patch_tp4_mlp.py
sudo docker exec "$NAME" python3 /workspace/gemma4-port/scripts/make_local_model.py
sudo docker exec -i "$NAME" python3 - <<'PY'
import json
from pathlib import Path
p = Path('/root/models/gemma-4-31b-it-text')
c = json.loads((p / 'config.json').read_text())
assert c['architectures'] == ['Gemma4ForCausalLM']
assert 'vision_config' not in c and 'text_config' not in c
assert list(p.glob('*.safetensors'))
print('text-only model ready:', p)
PY
