#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a streaming synthetic strict-JSON concurrency smoke."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
import urllib.request

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["label", "keywords"],
    "properties": {
        "label": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    },
}


def percentile(values, p):
    values = sorted(values); k = (len(values) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] if lo == hi else values[lo] * (hi-k) + values[hi] * (k-lo)


def valid_schema(value):
    return (isinstance(value, dict) and set(value) == {"label", "keywords"}
            and isinstance(value["label"], str)
            and isinstance(value["keywords"], list) and len(value["keywords"]) <= 8
            and all(isinstance(item, str) for item in value["keywords"]))


def request(base, index, repeat):
    system = "Classify the synthetic message and return only valid JSON. " * repeat
    payload = {
        "model": "gemma4", "temperature": 0, "max_tokens": 96, "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": f"Sample {index}: payment status question"}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "classification", "strict": True, "schema": SCHEMA}},
    }
    req = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter(); ttft = None; parts = []; usage = {}
    with urllib.request.urlopen(req, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "): continue
            body = line[6:]
            if body == "[DONE]": break
            chunk = json.loads(body)
            if chunk.get("usage"): usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    if ttft is None: ttft = time.perf_counter() - start
                    parts.append(content)
    e2e = time.perf_counter() - start
    try: parsed = json.loads("".join(parts)); valid = valid_schema(parsed)
    except json.JSONDecodeError: valid = False
    return {"request": index, "ttft_s": ttft, "e2e_s": e2e,
            "schema_valid": valid, "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--requests", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=1200)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.concurrency < 1 or args.requests < 1: ap.error("counts must be positive")
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(lambda i: request(args.base_url, i, args.repeat), range(args.requests)))
    wall = time.perf_counter() - start
    ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
    e2es = [r["e2e_s"] for r in rows]
    output_tokens = sum(r["usage"].get("completion_tokens", 0) for r in rows)
    out = {"config": vars(args) | {"base_url": "loopback"}, "wall_s": wall,
           "schema_success_rate": sum(r["schema_valid"] for r in rows) / len(rows),
           "output_tokens_per_s": output_tokens / wall,
           "requests_per_min": len(rows) * 60 / wall,
           "ttft_s": {f"p{p}": percentile(ttfts, p) for p in (50,95,99)},
           "e2e_s": {f"p{p}": percentile(e2es, p) for p in (50,95,99)},
           "results": rows}
    text = json.dumps(out, indent=2); print(text)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(text + "\n")
    raise SystemExit(0 if all(r["schema_valid"] for r in rows) else 1)


if __name__ == "__main__": main()
