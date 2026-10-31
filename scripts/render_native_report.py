#!/usr/bin/env python3
"""Render the audited takeover measurements into a new local Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    pairs = sorted((pair for run in audit["runs"] for pair in run["pairs"]),
                   key=lambda row: (row["isl_nominal"], row["osl_cap"], row["concurrency"]))
    storage = {job["arm"]: job["storage"] for job in audit["runs"][0]["jobs"]}
    bf16, fp8 = storage["bf16"], storage["fp8_e4m3"]
    table = ["| ISL 名义值 | 输出上限 | C | BF16 tok/s | FP8 tok/s | tok/s 变化 | req/s 变化 | ITL P50 变化 | TTFT P95 变化 | 实际 OSL BF16 → FP8 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in pairs:
        table.append(f'| {row["isl_nominal"]} | {row["osl_cap"]} | {row["concurrency"]} '
                     f'| {row["bf16_output_tps"]:.2f} | {row["fp8_output_tps"]:.2f} '
                     f'| {row["output_tps_change_pct"]:+.2f}% | {row["request_rps_change_pct"]:+.2f}% '
                     f'| {row["itl_p50_ms_change_pct"]:+.2f}% | {row["ttft_p95_ms_change_pct"]:+.2f}% '
                     f'| {row["bf16_osl_mean"]:.1f} → {row["fp8_osl_mean"]:.1f} |')
    rows = [row for run in audit["runs"] for row in run["rows"]]
    memory_ranges = {}
    for arm in ("bf16", "fp8_e4m3"):
        peaks = [row["gpu_memory_used_mib_max"] for row in rows if row["arm"] == arm]
        memory_ranges[arm] = f"{min(peaks)}–{max(peaks)}"
    text = f'''# ceff 原生 KV 量化接管审计与研究分析

证据采集：2026-10-26 UTC；控制端本地日期 2026-10-25。此报告由 `scripts/render_native_report.py` 从独立审计 JSON 生成，不修改历史结果。分类遵循 confirmed / reported / unverified / hypothesis / proposal。

## confirmed：已有实验恢复与测量完整性

实机发现六个 ceff run、九个负载条件、18 个 arm case。根/arm COMPLETE、state 成功与 rc=0、不可变 manifest、raw server log、AIPerf JSONL 及 phase exports 互相一致。每个 case 独立核查 128 profiling + 8 warmup，共 2304 条 profiling、144 条 warmup；profiling 生成 {sum(row["output_tokens"] for row in rows):,} 个实际输出 tokens。未重新运行任何历史实验。

所有记录有唯一 request ID，phase/session/turn/prompt-token-count 在 BF16/FP8 配对一致，warmup 和 profiling source session 无交集。server usage 与客户端 token count 一致；各 case 的均值、P50/P90/P95/P99、计数、sum 从 JSONL 重算，与根及 phase summary 一致。吞吐用实际 token 总数和 AIPerf observation window 重算。当前冻结输入、8 个 backend source 文件、binary、runner/shim 的 SHA 与 manifest 一致；模型 config hash 一致。

源码 `ceff00bbd38054d2b6b4aa44513d773f2bd174b2`；binary `277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3`；FlashInfer `0.6.18.post1`、AIPerf `0.13.0`。各组同 GPU2 UUID `GPU-a6169058-34c8-6edc-e4f5-779bf9a10421` 顺序 BF16→FP8，同 FA2 paged prefill/decode、Graph、prefix-cache=false、server argv（除 KV dtype）。Docker argv 限单 UUID、非 root、network none。容器内真实可见 UUID 的检验由当时 worker 执行并以成功返回支持；没有保存独立的容器内 UUID 原始快照。

两臂 raw server log 同时有实际 captured bucket 和 `NATIVE_KV_GRAPH_REPLAY`。C16 对照捕获 [1,2,4,8,16]，其他组 [1,2,4,8]，没有以 admission 消息替代 replay。

## confirmed：真实驻留容量

| 指标 | BF16 | FP8 E4M3（含 scale） |
|---|---:|---:|
| bytes/token/layer | {bf16["bytes_per_token_per_layer"]} | {fp8["bytes_per_token_per_layer"]} |
| pages/layer | {bf16["pages_per_layer"]} | {fp8["pages_per_layer"]} |
| KV token capacity | {bf16["token_capacity"]} | {fp8["token_capacity"]} |
| 全 48 层分配 resident GiB | {bf16["resident_bytes"] / 2**30:.4f} | {fp8["resident_bytes"] / 2**30:.4f} |

逐 tensor 的 dtype、shape、字节相乘、scale 字节、48 层合计均核验。FP8 存储为 uint8 resident pages 的 float8 alias；日志明确 `fp8_views_alias_resident_pages=true`，K=.125/V=.0625、无 quality transform。每 token 驻留下降 {100 * (1-fp8["bytes_per_token_per_layer"]/bf16["bytes_per_token_per_layer"]):.4f}%，当前分配 token 容量 {fp8["token_capacity"]/bf16["token_capacity"]:.5f} 倍。两臂总 resident 预算接近，是扩大 pages，并非压缩失效。此轮分配预算约 44.9 GiB，不能与旧 eager 的约 34.6 GiB 总量直接拼接。

这里的 token capacity 是 allocator 的存储容量证据，尚未以满载 admission/排队服务实测证明最大可用并发。

profiling request 时间段内的 NVML 采样（排除 warmup）显示：各条件 device memory 峰值，BF16 为 {memory_ranges["bf16"]} MiB，FP8 为 {memory_ranges["fp8_e4m3"]} MiB。两臂 device footprint 接近且 FP8 略高，与 allocator 使用压缩增加可用 pages 并不矛盾。采样覆盖 request span 至少 {min(row["gpu_telemetry_span_fraction"] for row in rows)*100:.1f}%，最大相邻间隔 {max(row["gpu_telemetry_max_gap_s"] for row in rows):.0f}s；这是抽样观察到的峰值，不能认证采样间隙内的瞬时峰值。device memory 包含权重、KV、Graph/workspace 等，不能当 KV payload。
NVML compute/memory-controller utilization 是活动时间比例，不是 HBM bytes/s 或计算瓶颈证明；本次没有真实带宽计数器。

## confirmed：有条件的描述性性能

下表变化统一为 FP8/BF16−1；吞吐正值更高，延迟正值更慢。**单次 run/负载条件，没有 run-level CI，不据 128 个相关请求伪造独立重复。** 名义 ISL 8192/16384/24576 的实际 server prompt tokens 为 8221/16413/24605。

{chr(10).join(table)}

数据不支持“压缩近一半必然加速”。24K/C2 和 C4 的 token/s 增长约 5.9%/6.1%，同时 req/s 下降约 9.7%/7.4%，TTFT P95 上升约 15.4%/8.0%，实际输出增长约 17.3%/14.6%。因此不能直接宣称用户请求完成得更快。16K/C8 token/s 下降约 5.9%；8K/C16 约持平，这些均保留为描述性结果。

输出上限 512/1024 不是固定 decode 长度：各 arm 实际均值 211–287 tokens，多数请求未到上限。JSONL 没保存逐请求 finish reason，提前结束的具体原因不能仅凭长度证明。相同 source/session/输入长度支持输入配对，实际 HTTP payload bytes 没有归档，不能声称完成了字节级线上请求审计。

## unverified 与 hypothesis

- **unverified**：相同 native FP8 fixed scale 的 GSM8K/LongBench 质量；满载并发容量；HBM bandwidth、kernel 时间、quantize writer 开销；每个 timed decode step 的 Graph 覆盖率；权重全内容的历史 SHA。日志只在每 server 首次 replay 时记录，不能由单个 marker 断言全部 timed steps。
- **confirmed（源码）**：xLLM chat proto/RequestParams 支持 `ignore_eos`，默认 false；temperature 默认 0。冻结 payload 没显式温度/EOS 设置。默认 greedy 也不等于跨并发/数值路径逐 token 可重复，现有记录不能认证 determinism。
- **hypothesis**：长 ISL 的局部 ITL 改善可能来自 KV 访存收益，被 prefill、量化写入或调度开销抵消；输出长度变化也改变了持续并发。当前没有硬件计数器，不能把该解释写成已确认机制。
- **proposal**：保持现有自然停止数据，另建明确 `ignore_eos=true` 的派生固定 decode 输入，先 CPU 核对 AIPerf loader 保留该字段，再以真实 response usage 验收 fixed length；两类工作负载独立报告。优先 16K/C4 基线与 24K/C2 交叉点，采用交替 arm 顺序和 run-level replicates。只有变量/统计问题明确需要才产生新 run。
- **proposal**：在 scale 不变、后端相同的原生 paired quality 通过前，不发布“质量可接受 Pareto 优势”。旧 Triton 四臂质量用于历史问题定位，不移植为本 native 实现的质量证据。

## 安全与运行限制

本会话 remote writes=0、GPU launches=0。可信许可文件缺失，不自动签发，不因空闲就使用 H800。旧 runner 的 5 秒 idle window 与共享 cache rw mounts 仅作为历史事实保留；新调度需至少政策要求的稳定检查、run 独立 cache、固定可信批准源、唯一 run 锁、预算预留/实际计量/超时终止我方进程及断线 UNKNOWN 对账。原 runner 本身没有这些完整保护。没有 OS 级配额沙箱证据。

## 可复现证据

- `audit.json`：逐 run 验收、graph markers、allocation、hash、局限。
- `cases.csv`：18 个 case 的实际 tokens、三种时间口径、吞吐与延迟分位数。
- `pairs.csv`：九个条件的两臂数值与相对变化。
- 原始镜像：`results/mirrors/takeover-20261026T040426Z-attempt02/files/SNAPSHOT.json`，372 个选择文件逐 SHA 校验。
- 只读 provenance：`reports/operations/20261026T040426Z/provenance_readonly.json`；AIPerf 安装源码 `aiperf_metric_sources.jsonl`、`aiperf_metric_dicts.source.json` 说明 observation window 与 benchmark_duration 的区别。

重算命令（输出目录必须全新）：

```bash
python3 -B scripts/audit_native_results.py --mirror results/mirrors/takeover-20261026T040426Z-attempt02/files --provenance reports/operations/20261026T040426Z/provenance_readonly.json --out reports/experiments/NEW_AUDIT_DIRECTORY
python3 -B -m unittest discover -s tests -v
```
'''
    with args.out.open("x") as stream:
        stream.write(text)
    print(str(args.out))


if __name__ == "__main__":
    main()
