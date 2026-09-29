# Gemma4-31B on one AWS Trainium2 chip

Experimental text-only `google/gemma-4-31b-it` integration for the public [vLLM-Neuron](https://github.com/vllm-project/vllm-neuron) 0.21 plugin, adapted and validated for the **trn2.3xlarge** envelope: one Trainium2 chip, TP=4, BF16 weights/KV, 12 vCPU and 128 GiB host RAM.

> This is not an official AWS or Google artifact. Gemma weights are governed by Google's Gemma terms and are not included.

## Attribution

The Apache-2.0 model-integration files in `serving_pkg/gemma4` are adapted from [Armin Agha-Ebrahim's public Armin-Neuron Gemma4 work](https://github.com/arminagha1234/Armin-Neuron/tree/fc1af21a8620c97e6f0d67f48f89a8388a569808/gemma4-31b/vllm-neuron-4k_16k_32k_64_PublicVLLM). See [`NOTICE`](NOTICE) for the exact source path and commit. Single-chip configuration, TP=4 fallback handling, safety checks and public synthetic validation were added here.

## Why this repository

The upstream reference targets larger TP=8/TP=32 configurations on trn2.48xlarge. A 31B BF16 model also fits on one Trainium2 chip, but needs a different memory and scheduler envelope:

- TP=4 gives intermediate-size/rank 5,376; the current NKI MLP kernel rejects this shape, so the PyTorch fallback must use a traceable inline GELU-tanh expression.
- `GMU=0.70` keeps enough HBM outside KV for runtime scratch.
- vLLM-Neuron 0.21 may combine long segmented prefills or mix prefill/decode when server `MNS>1`; the supported correctness-first mode uses server `MNS=1` while concurrent clients queue.
- Structured output requires Neuron structured-output support and synchronous scheduling.

## Quickstart

Use the public DLC shown below. Put an accepted, locally downloaded Gemma checkpoint at `~/models/gemma-4-31b-it`; this repository does not download or redistribute weights.

```bash
# Host
REPO_DIR=$PWD bash scripts/01_container.sh
bash scripts/02_prepare.sh

# Server (foreground, inside the container)
sudo docker exec -it vllm_gemma4 bash -lc \
  'cd /workspace/gemma4-port && bash scripts/03_serve.sh'

# Synthetic strict-JSON smoke test, from the host
python3 tests/smoke_json_concurrency.py --concurrency 4 --requests 10
```

`CPUSET` is optional on a real trn2.3xlarge. When simulating the envelope inside a larger instance, select six physical cores plus their SMT siblings local to `/dev/neuron0`; verify topology rather than copying the example blindly.

## Validated configuration

```text
max_model_len=12288
TP=4, BF16 weights and KV
server max_num_seqs=1
client concurrency=4 (queued safely)
SEG=512, APC enabled, structured JSON enabled
GMU=0.70, KV budget cap=0.30
```

A 10-request synthetic long-form classification pilot completed 10/10 HTTP/JSON/schema checks, sustained 16.68 output tokens/s and 7.70 requests/min, and peaked at 18.20 GiB per logical NeuronCore. These are workload-specific engineering measurements, not a general model leaderboard. See [`docs/SINGLE_CHIP_RESULTS.md`](docs/SINGLE_CHIP_RESULTS.md).

## H100 comparison scope

The source reference publishes useful Trainium2 TP=8/TP=32 versus H100 TTFT data. Those results cannot be used as an apples-to-apples claim for this TP=4 single-chip configuration. A fair comparison must run the same model revision, BF16 policy, prompts, output lengths, schema constraints, warm-up/cache state and load generator on both platforms, then report quality and cost per successful request.

## Layout

- `serving_pkg/gemma4/`: Apache-2.0 model integration from the attributed source
- `scripts/install_plugin.py`: install/register with vLLM-Neuron
- `scripts/patch_tp4_mlp.py`: traceable GELU fallback with numerical parity check
- `scripts/make_local_model.py`: non-destructive text-only checkpoint view
- `scripts/01_container.sh`, `02_prepare.sh`, `03_serve.sh`: reproduction pipeline
- `tests/`: public synthetic smoke and safety tests

## License

Repository code is Apache-2.0 where indicated. See `LICENSE` and `NOTICE`. Model weights are excluded and retain their upstream terms.
