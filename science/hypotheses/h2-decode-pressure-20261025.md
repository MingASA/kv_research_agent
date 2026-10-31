# H2 更新：容量、payload 读取与端到端收益要分别建模

分类：下面分别标注实测、源码事实与待检验推断；没有 HBM 带宽实测。
依据：`reports/experiments/ceff-takeover-20261026T040426Z-v2/REPORT.md` 与 ceff 的 `flashinfer_fp8.py`。

## confirmed：存储账本与代码路径

48 层，8 KV heads，head_dim=128；BF16 K/V 每 token 每层 4096 B。
FP8 K/V payload 2048 B，另有两个 per-token/head FP32 scale 数组，共 64 B，resident 总 2112 B。
实际分配 pages/layer 1916→3716，token capacity 245248→475648（约 1.93946×）。

ceff `FlashInferFP8Backend.bind_kv_caches` 将 resident uint8 直接 view 成 float8。
`execute_attention` 向 FlashInfer wrapper 传的是 FP8 payload views 和两个固定 scalar scales；没有传每 token/head scale 数组。
`write_cache` 仍写 per-token/head scale arrays，代码明确为 page-copy/accounting 兼容保留。
因此 **2112/4096 是 resident 容量比例，不能不加区分地充当 FlashInfer decode 的 HBM 读取比例**。

## inference：理想化 KV payload 扫描量（不是硬件计数）

若一个 decode token 对一个序列完整扫描每层所有历史 KV，各 KV payload 恰读一次，忽略缓存命中、读放大、GQA 内部复用、额外工作区和写入：

`B_KV_payload = 48 × context_tokens × 2(K,V) × 8 × 128 × dtype_bytes`。

| context | BF16 payload/序列/token | FP8 payload/序列/token |
|---:|---:|---:|
| 8192 | 1.5 GiB | 0.75 GiB |
| 16384 | 3.0 GiB | 1.50 GiB |
| 24576 | 4.5 GiB | 2.25 GiB |

并发 C 的一批 decode 在此理想模型中乘 C；真实 kernel 的 GQA、L2、prefill chunk、调度会改变硬件流量。
这个表是机制对照的量纲估算，不能标成实测 HBM bytes，更不能除设备宣传带宽伪造实际 latency。

以原 BF16 总耗时中 attention 占比 f、实际 attention 时间改善 r、额外 normalized 开销 δ，条件模型为：

`speedup = 1 / ((1-f) + f/r + δ)`。

f、r、δ 均未测得；KV payload byte 减半只给某种理想访存模型下 r≈2 的参考，不能把它设成真实 r。
FP8 kernel、quantize/scatter、prefill、权重/MLP、Graph、CPU 调度可改变这些量。
当前输出长度不同，不能从端到端 token/s 反推 f 或 δ。

## hypothesis 与反证

1. H2a：24K/C2 的 ITL 改善主要来自 KV 访问节省。
   固定 decode length 后若 attention 部分时间不降，或访存量不降，该机制解释受反驳；不能只看 total tok/s。
2. H2b：TTFT 退化来自 prefill 中量化写入或 FP8 paged-prefill 路径。
   分离 writer 与 prefill attention；如果 writer/prefill 都没增加，而排队时间增加，则转向调度解释。
3. H2c：8K/C16 接近零收益反映 KV 节省被其他成本抵消。
   NVML GPU utilization 约 97–99% 不能证明 compute-bound；memory-bound kernel 同样可能接近 100% 活动时间。
4. H1a（独立新方案）：native-only 表示可否不分配冗余 per-token/head scales？
   理想 resident 从 2112→2048 B，固定预算的容量有约 3.125% 额外空间。
   **proposal**，不是当前优化：须修改 C++ allocation、cache-copy/slot/accounting 契约与验证，另立版本，不混进 ceff 主结果。

## 最小下一证据

先完成新派生 fixed-decode 方案与 native paired quality；在许可范围内只对 16K/C4、24K/C2 做有明确必要性的对照。
随后独立诊断 run，用我方应用内时间分解观测 writer/prefill/decode/非 attention；Graph 的事件与 replay 计时语义先 CPU/小规模验收。
正式性能主结果不开侵入式观测；未经许可不做影响其他用户的系统级 profiler。
如无法获取硬件 HBM counters，报告 app-local 时间证据与该限制，不拿 NVML memory utilization 代替 GB/s。
