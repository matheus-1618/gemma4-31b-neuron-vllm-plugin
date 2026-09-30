#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Make the TP=4 PyTorch MLP fallback traceable without changing GELU math."""
from __future__ import annotations
import py_compile
import shutil
import sys
from pathlib import Path

OLD = 'return lambda x: torch.nn.functional.gelu(x, approximate="tanh")'
NEW = (
    "return lambda x: 0.5 * x * (1.0 + torch.tanh("
    "0.7978845608028654 * (x + 0.044715 * x * x * x)))  # TP4_INLINE_GELU"
)


def locate() -> Path:
    for base in map(Path, sys.path):
        candidate = base / "vllm_neuron/functional/mlp.py"
        if candidate.is_file():
            return candidate
    raise SystemExit("vllm_neuron/functional/mlp.py not found")


def main() -> None:
    import torch
    x = torch.linspace(-8, 8, 10001, dtype=torch.float32)
    reference = torch.nn.functional.gelu(x, approximate="tanh")
    inline = 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    max_error = (reference - inline).abs().max().item()
    if max_error > 2e-6:
        raise SystemExit(f"GELU parity failed: max_error={max_error}")
    path = locate()
    source = path.read_text()
    if "TP4_INLINE_GELU" in source:
        print(f"already patched: {path}; max_error={max_error:.3g}")
        return
    if source.count(OLD) != 1:
        raise SystemExit(f"unexpected source signature in {path}")
    backup = path.with_suffix(path.suffix + ".bak_tp4")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(source.replace(OLD, NEW, 1))
    py_compile.compile(str(path), doraise=True)
    print(f"patched {path}; max_error={max_error:.3g}")


if __name__ == "__main__":
    main()
