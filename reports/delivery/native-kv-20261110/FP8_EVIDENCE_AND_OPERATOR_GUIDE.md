# Native FP8 KV cache on Hopper

The CUDA Python Executor can keep full-attention K/V pages in E4M3 FP8.
The default BF16 path is unchanged. FP8 uses one resident byte per element,
with immutable global K/V scales and no full-cache BF16 shadow. Whole-page
copies preserve the stored bytes; hybrid GDN state retains its official dtype.

## Execution and configuration

The evaluated path uses FlashInfer FA3 with FP8 Query for Prefill and normal
mixed batches, and FlashInfer XQA for pure Decode. A fused Triton writer reads
the input strides and quantizes directly into resident pages. Query conversion
also reads strides directly, avoiding a separate BF16 contiguous copy.
XQA workspace is zeroed once before capture. Its API accepts BF16/FP16 Query;
the pinned SM90 implementation converts that Query internally to FP8.

Append these flags to the normal CUDA Python Executor launch:

```sh
--kv_cache_dtype=fp8_e4m3 \
--kv_cache_prefill_backend=fa3 \
--kv_cache_decode_backend=xqa \
--kv_cache_k_scale=0.125 \
--kv_cache_v_scale=0.5
```

These scales are the independently calibrated **Qwen3.8-27B recipe**, not
universal defaults. The scale flags default to 1.0 and require finite positive
normal FP32 values. Stored values are `FP8(clamp(x / scale, -448, 448))`;
attention applies the corresponding dequantization factors. No quality labels
were used to choose the evaluated scale candidate.

Admission targets SM90, dense full attention, BF16/FP16 activations, logical
head dimensions 128/256 and GQA groups of at least four. Actual model evidence
is H800, Q24/KV4/D256, page128. Other admitted shapes require their own model
validation. MLA/indexers, padded head dimensions, PD transfer, host offload,
external KV stores and native FP8 VMM are excluded. Unsupported settings fail
explicitly; no failure-driven fallback is added. Scheduler, Graph runner and
Prefix Cache implementations remain unchanged.

Mixed Prefill/Decode work uses the full packed prefill path. The experimental
mixed FA3/XQA split was removed after its fixed performance gate failed.
Graph evidence covers real pure-Decode replay, not every formal request or
mixed batch. The official default allocator and page lifecycle are retained.

## Existing H800 evidence

Official source is `261274e5f45449e98d7740144c3a958a29f1fd3e`; the separate
CUDA build prerequisite is `721afdabc1de1af8e594b4da3e6fa926def50821`.
The retained FP8 Python source was measured at
`d163d5de48b201e5f767f1bf21a9437cbf1aeb42`, with compiled native source
`da7c6446ff4361c8060633b04d6bfb77e6f737ff`.
The measured dependencies were Torch 2.10/cu130, FlashInfer 0.6.18.post1 and
Triton 3.6. This branch extracts the FP8 implementation from that evidence;
the extraction itself is not a newly measured serving binary.

All 1,720 frozen core quality generations and 32 multi-turn tool tasks are
retained. Displayed quality scores are percentages:

| Arm | GSM8K | LongBench macro | RepoBench-P | Tools |
| --- | ---: | ---: | ---: | ---: |
| Official BF16 | 96.7399545110 | 53.8984440135 | 43.92 | 26/32 |
| Retained FP8, V=.5 | 96.9673995451 | 53.9033424701 | 42.84 | 26/32 |

The frozen point-loss thresholds passed. RepoBench-P statistical noninferiority
at the two-point margin remains inconclusive: paired 95% interval for the
difference is `[-6.0605, 2.801]` points. Point thresholds are engineering
criteria, not a universal quality guarantee.

The service comparison reused the same natural-EOS requests, original output
caps and warmup separation. Settings were max batch32, batch tokens8192,
page128 and memory utilization ceiling .90:

| Workload | BF16 output tokens/s | FP8 output tokens/s | BF16 request/s | FP8 request/s |
| --- | ---: | ---: | ---: | ---: |
| ShareGPT C32, 512 requests | 699.865 | 684.558 | .955407 | .924249 |
| ShareGPT C64, 512 requests | 793.692 | 799.917 | 1.081007 | 1.070344 |
| GovReport C32, 64 requests | 159.351 | 204.979 | .311813 | .400447 |

GovReport output throughput improved 28.63338873007899%, while P95
per-request mean ITL increased from 68.3295669892368 to 105.63278218982386 ms.
ShareGPT showed no consistent throughput gain. Each case has one completed
window; small differences do not establish statistical improvement. Full-KV
pages were 1,346 for BF16 versus 2,693 for FP8; GDN and other allocations limit
whole-service capacity. Capacity ratios are not throughput ratios.

Exact values, complete latency/Goodput tables, inputs, failures and raw evidence
are in the [frozen research report](https://github.com/MingASA/kv_research_agent/tree/research/mxfp4-fp8-20261105/reports/experiments/mxfp4-fp8-20261105).

## Checks

Run the independent host contracts without loading the compiled executor:

```sh
CUDA_VISIBLE_DEVICES='' python -m pytest --noconftest -q \
  tests/python/test_native_kv_prefill_dispatch_cpu.py \
  tests/python/test_native_fp8_cache.py
```

On an authorized SM90 GPU, the same test file additionally checks writer bytes
against an independent Torch cast, changed-input Writer Graph replay and
strided Query conversion. Other hardware skips those device checks.
The C++ tests `NativeFP8BudgetsOnlyResidentBytes` and
`FP8BytePagesCopyAndKeepLinearStateDtypes` cover allocation accounting,
whole-page copy and hybrid state dtypes through real cache implementations.
Full C++/CUDA build and the model service are separate from host-only checks.
