#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a text-only Gemma4 model directory without changing raw weights."""
from __future__ import annotations
import json
import os
from pathlib import Path

SRC = Path(os.environ.get("GEMMA_RAW", "/root/models/gemma-4-31b-it"))
DST = Path(os.environ.get("GEMMA_TEXT", "/root/models/gemma-4-31b-it-text"))


def main() -> None:
    if SRC.resolve() == DST.resolve():
        raise SystemExit("GEMMA_RAW and GEMMA_TEXT must differ")
    for name in ("config.json", "tokenizer_config.json"):
        if not (SRC / name).is_file():
            raise SystemExit(f"missing required file: {SRC / name}")
    weights = sorted(SRC.glob("*.safetensors"))
    if not weights:
        raise SystemExit(f"no safetensors files under {SRC}")
    DST.mkdir(parents=True, exist_ok=True)
    for src in SRC.iterdir():
        if src.name in {"config.json", "tokenizer_config.json"}:
            continue
        dst = DST / src.name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())

    tokenizer = json.loads((SRC / "tokenizer_config.json").read_text())
    tokenizer.pop("extra_special_tokens", None)
    (DST / "tokenizer_config.json").write_text(
        json.dumps(tokenizer, indent=2, ensure_ascii=False) + "\n"
    )

    cfg = json.loads((SRC / "config.json").read_text())
    for key in (
        "vision_config", "audio_config", "image_token_id", "video_token_id",
        "audio_token_id", "boi_token_id", "eoi_token_id", "boa_token_id",
        "eoa_token_id", "eoa_token_index", "vision_soft_tokens_per_image",
    ):
        cfg.pop(key, None)
    text_cfg = cfg.get("text_config", {})
    if isinstance(text_cfg, dict):
        for key, value in text_cfg.items():
            cfg.setdefault(key, value)
        cfg["model_type"] = text_cfg.get("model_type", cfg.get("model_type"))
    cfg.pop("text_config", None)
    cfg["architectures"] = ["Gemma4ForCausalLM"]
    (DST / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"built {DST}; linked {len(weights)} weight shards")


if __name__ == "__main__":
    main()
