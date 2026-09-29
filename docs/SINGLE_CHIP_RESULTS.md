# Single-chip engineering results

## Scope

These measurements validate capacity and serving behavior of Gemma4-31B on the trn2.3xlarge envelope. The original application data is not published; the reported shape is a generic long-form classification workload with a shared instruction prefix and strict JSON schema.

## Environment

- One Trainium2 chip, four logical NeuronCores (LNC2)
- TP=4, BF16 model and KV cache
- 12 vCPU, 128 GiB host RAM
- public vLLM-Neuron 0.21 / Neuron SDK 2.31
- max model length 12,288; segmented prefill bucket 512

## Memory experiments

These sanitized runtime summaries are published in [`results/memory_profiles.json`](../results/memory_profiles.json).

| Configuration | KV capacity | Result |
|---|---:|---|
| server MNS=4, GMU=.70 | 41,363 tokens / 3.37 full-length requests | insufficient for four worst-case 12K requests |
| server MNS=4, GMU=.73 | 57,953 tokens / 4.72 requests | memory fits, scheduler still fails under concurrent long prefills |
| server MNS=1, GMU=.70 | one active request plus client queue | correctness-first validated mode |

Increasing KV memory did not fix MNS=4. With prefix caching both enabled and disabled, vLLM-Neuron 0.21 could aggregate two prefill chunks into a graph compiled for one request or produce a mixed prefill/decode scheduler output. Experimental scheduler patches were not retained because they did not close every path.

The public serve script therefore fails closed for `MNS>1`. `ALLOW_UNSAFE_MNS=1` is available only for reproducing the upstream scheduler issue.

## Validated queued-concurrency pilot

Server MNS=1, client concurrency=4, strict JSON schema, 10 requests. The anonymized per-request timings and token counts are published in [`results/single_chip_validation.json`](../results/single_chip_validation.json); source text and task labels are excluded.

| Metric | Value |
|---|---:|
| HTTP success | 10/10 |
| Strict JSON/schema success | 10/10 |
| Output throughput | 16.68 tokens/s wall-clock |
| Requests/minute | 7.70 |
| TTFT p50/p95/p99 | 23.42 / 24.21 / 24.33 s |
| E2E p50/p95/p99 | 30.79 / 31.83 / 31.91 s |
| Peak HBM | 72.798 GiB/chip; 18.200 GiB/core |

Sanitized 5-second HBM samples and collection scope are in [`results/hbm_samples.json`](../results/hbm_samples.json).

Queued TTFT includes time waiting behind earlier requests. The result demonstrates stable admission and output throughput, not continuous-batching scaling.

## Relationship to the source H100 comparison

Armin-Neuron publishes TP=8 and TP=32 Trainium2 versus H100 TTFT measurements for Gemma4-31B at 4K–64K. See the [source documentation](https://github.com/arminagha1234/Armin-Neuron/tree/fc1af21a8620c97e6f0d67f48f89a8388a569808/gemma4-31b/vllm-neuron-4k_16k_32k_64_PublicVLLM). Those numbers show that larger Trainium2 configurations can be competitive at long-context prefill, but they are not evidence that this TP=4 configuration beats H100.

For a customer-facing comparison, report identical workload quality, TTFT, TPOT, E2E, throughput, peak memory and cost per successful 1,000 requests on both the single-chip Trainium2 configuration and the selected H100 instance.

## Optimization priorities

1. Split the 5,376-wide per-rank MLP intermediate dimension into PSUM-safe NKI tiles, preserving BF16 parity.
2. Fix prefill/decode separation in the vLLM-Neuron scheduler so server MNS>1 can be re-enabled.
3. Add a public fixed-distribution harness and run the same suite on H100.
