Native FP8 KV pages currently need an explicit CUDA Python Executor path with matching resident-byte allocation, scale propagation and FlashInfer dispatch. This change adds opt-in E4M3 FP8 storage without changing the default BF16 scheduler, Graph runner, Prefix Cache or GDN state management.

The implementation uses a fused strided KV writer, FA3 FP8-Query packed Prefill (including normal mixed batches) and XQA pure Decode. It retains the measured strided Query conversion and one-time XQA workspace initialization. The rejected mixed Prefill/Decode split is absent. There is no full-cache BF16 shadow or failure-driven fallback.

Scope: SM90 full attention, BF16/FP16 activations, logical D128/256 and GQA groups >=4; actual model evidence is H800/Qwen3.8-27B, Q24/KV4/D256/page128. Native VMM, MLA/indexers, padded dimensions, PD transfer, host offload and external stores are excluded explicitly. NVFP4/MXFP4 and their packed-layout/private-adapter infrastructure are separate development branches.

The diff targets pinned official 261274e5 and includes the isolated CUDA/NPU build prerequisite 721afdab. New configuration flags are documented in both CLI references. K/V scales default to 1.0; the independently calibrated model recipe is K=.125/V=.5, FA3/XQA. That recipe is not a universal default.

Existing frozen evidence (measured Python d163d5de, compiled native source da7c6446; Torch2.10/cu130, FlashInfer0.6.18.post1, Triton3.6):

| Metric | Official BF16 | Retained FP8 |
| --- | ---: | ---: |
| GSM8K, 1319 items | 96.7399545110% | 96.9673995451% |
| LongBench 8-task macro | 53.8984440135% | 53.9033424701% |
| RepoBench-P | 43.92% | 42.84% |
| Multi-turn tools | 26/32 | 26/32 |
| ShareGPT C32 output tokens/s | 699.865 | 684.558 |
| ShareGPT C64 output tokens/s | 793.692 | 799.917 |
| GovReport C32 output tokens/s | 159.351 | 204.979 |

All 1720 core quality generations and original stopping rules were retained. Frozen quality point-loss thresholds passed; RepoBench-P statistical noninferiority at a two-point margin remains inconclusive (paired95% difference interval [-6.0605,2.801] points). GovReport throughput improved 28.63338873007899%, while P95 per-request mean ITL rose from68.3295669892368 to105.63278218982386ms. ShareGPT showed no consistent gain. Each service case has one completed window. FP8 had2693 Full-KV pages versus1346 for BF16; no theoretical-capacity-to-throughput equivalence is claimed.

Validation of this extraction: 13 CPU tests and18 subtests passed;3 SM90 tests skipped with CUDA hidden. C++ formatting uses clang-format20.1.6; changed/new Python passes Ruff/syntax checks. Core Writer/Query/XQA source files are byte-identical to the retained source, prepare AST is identical, and execute matches after selecting the existing FP8 branch. Real CPU storage/copy and device-ready independent Writer/Query/Graph tests are included. The common BF16 backend matches official source.

This extracted branch has not been rebuilt as a complete C++/CUDA service or newly benchmarked; normal upstream build/CI and integration checks remain required before merge. Its results reuse the explicitly identified historical artifacts, rather than relabeling a new commit as already measured.

[Full frozen quality/service report](https://github.com/MingASA/kv_research_agent/tree/research/mxfp4-fp8-20261105/reports/experiments/mxfp4-fp8-20261105).
