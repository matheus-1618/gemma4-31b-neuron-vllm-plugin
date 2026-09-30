#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install and register the Gemma4 model package in vLLM-Neuron 0.21."""
from __future__ import annotations
import py_compile
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "serving_pkg/gemma4"


def model_root() -> Path:
    for base in map(Path, sys.path):
        candidate = base / "vllm_neuron/model"
        if candidate.is_dir():
            return candidate
    raise SystemExit("vllm_neuron/model not found; run inside the public DLC")


def main() -> None:
    root = model_root()
    dst = root / "gemma4"
    dst.mkdir(exist_ok=True)
    for source in SRC.glob("*.py"):
        shutil.copy2(source, dst / source.name)
        py_compile.compile(str(dst / source.name), doraise=True)

    registry = root / "registry.py"
    text = registry.read_text()
    import_line = "from .gemma4 import Gemma4ForConditionalGeneration"
    if import_line not in text:
        lines = text.splitlines()
        index = max(i for i, line in enumerate(lines) if line.startswith("from ."))
        lines.insert(index + 1, import_line)
        text = "\n".join(lines) + "\n"
    if '("Gemma4ForConditionalGeneration", Gemma4ForConditionalGeneration)' not in text:
        anchor = "    models = ["
        entries = (
            '    models = [\n'
            '        ("Gemma4ForConditionalGeneration", Gemma4ForConditionalGeneration),\n'
            '        ("Gemma4ForCausalLM", Gemma4ForConditionalGeneration),'
        )
        if anchor not in text:
            raise SystemExit("registry models anchor not found")
        text = text.replace(anchor, entries, 1)
    registry.write_text(text)
    py_compile.compile(str(registry), doraise=True)
    print(f"installed {dst} and updated {registry}")


if __name__ == "__main__":
    main()
