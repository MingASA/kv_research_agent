# 研究路线与实验优先级：不要把当前单组任务当终点

**2026-10-27当前任务覆盖**：新任务执行 [NEXTGEN_NATIVE_KV_PLAN.md](NEXTGEN_NATIVE_KV_PLAN.md) 与 `ops/NEXTGEN_NATIVE_KV_STANDARD.md`，研究Qwen3.8-27B优先的BF16/原生FP8/NVFP4工程和生产评估。以下旧Qwen2.5矩阵作为历史保留，不再自动派发；共享服务器安全、生产代表性和原始证据规则继续有效。

**2026-10-26最新用户目标优先**：按 `ops/PRODUCTION_DECISION_STANDARD.md` 和 `plans/PRODUCTION_ROLLOUT.md` 作生产支持决策。下述固定FA2/强制512/论文式矩阵是历史设计，不自动执行。当前为真实ShareGPT/自然EOS/生产prefix与调度、现代FA3候选配对，随后补同候选质量与BurstGPT/Mooncake到达、容量和前缀负载。输入语义对账见 `reports/operations/20261026T075846Z/PRODUCTION_TRACE_FINDINGS.md`。

## P0 恢复、归档与正确性

**2026-10-26 UTC 更新（confirmed）**：首次只读巡检发现初始 ceff 16K/C4 与夜间矩阵已完成；六个 run / 九个条件 / 18 case 的 raw request、Graph、驻留存储与 provenance 已独立验收。不要执行下面“若没有实验”的分支。当前分析与局限见 `reports/experiments/ceff-takeover-20261026T040426Z-v2/REPORT.md`；后续优先固定 decode 控制、新原生 paired quality 与 `science/hypotheses/h2-decode-pressure-20261025.md` 的机制检验。GPU 许可与完整执行器仍是边界。

- 先**只读恢复** ceff Graph AIPerf 和过夜队列状态；若完成，校验 job rows、128 profiling/8 warmup、Graph 证据、token totals、原始 AIPerf 文件后分析；若异常，仅修其失败项。不要重复成功任务。
- 若没有实验，按授权先执行 **ceff 新 binary、native BF16 vs native FP8 E4M3、16K+512、C4、128 请求、8 warmup**；两臂顺序、同 GPU、同 runner、同 FA2。固定 FP8 K=.125,V=.0625。Graph capture+replay 为验收门槛。失败时暂停扩大矩阵。
- 缺失条件：GPU 安全授权、SSH 正确、代码/数据/binary hash、H800 daemon/docker 命令允许范围、同目录现有 run id 检查。

## P1 Synthetic 控制实验：因果趋势

冻结 15 组矩阵 ISL=[512,4096,8192,16384,24576] × OSL=[256,512,1024]，每组 136 条（8 warmup + 128 profiling），并发先 C=[1,4,8] 视 BF16 实际 KV capacity 和 graph buffer/持续显存。优先次序：16K512 C4 → 8K512 C4/C8 → 16K512 C8 → 24K512 C2/C4 → 8K1024 C4/C8 → 短上下文低 C。C16 仅在 BF16 预估容量和 Graph 预算有富余时尝试，**不能仅因为以前试过 C8 就默认 C16 安全**。24K/1024高压任务也需预算和 OOM 回退。允许只选最有信息增益的子集，不要求把 15×所有 C 全跑一遍。

每组在独立运行目录，重复性差/瓶颈交叉点做 paired replicates，随机化或交替 BF16/FP8 以缓和温度与机器时变偏差。可报告 TTFT、ITL、E2E latency P50/P90/P99、RPS、input/output tok/s、实际缓存容量和内存占用；不能只比较任务 wall-clock。

## P2 真实工作负载与到达分布

Mooncake FAST'25 conversation / toolagent / synthetic 各 264 的历史冻结 trace；先确认 replay 是否保留到达时间和 prefix reuse，历史 `--no-fixed-schedule` 或任意分片都会破坏原 trace 时间语义。对固定调度 vs 自然 trace 两种模式分别标注。ShareGPT 内容真实、长度分布不同；BurstGPT 可能只提供长度/到达信息而无原始用户文本，需验证具体文件与许可，不得捏造完整语义。Agentic Code、多轮工具调用属于研究计划，当前并无已确认新原生完整跑通记录。高价值工作负载不是越多越好：选择能回答调度/前缀/长上下文假设者。

## P3 质量

在新 native ceff 同 checkpoint/同 scales/同 tokenizer/同 prompt/同解码设置下 BF16 vs FP8 做至少 GSM8K 1319 + LongBench 8×50 的 paired 测试（或者明确统计支持的部分子集，不能当全量）。比较 strict/flexible EM、任务级分数与 macro、正负 flip、按 ISL/prompt 类型/答案长度分组。对 25/25 这种 Smoke 同错只记录输出一致性，不能写“模型准确率通过”。要分析 FP8 长上下文/分层激活分布、overflow/saturation/outliers、scales 与敏感 K/V。旧质量结果已完成，不应无故重复旧 Triton 四臂；新原生是不同实现且需要补测。

## P4 资源与机制

测 CUDA Graph capture overhead、Graph bucket 是否真的 replay、kernel profile、HBM traffic 和 throughput bottleneck、KV page utilization/reservation vs resident bytes/token、server CPU 占用、GPU utilization; 对 Nsight/NVProf 这种可能影响他人的系统级 profiler 要依照授权和工具安全策略，优先 app-local tracing。避免生产同时开侵入式 tracing 影响邻居。

## P5 方案空间拓展与自适应

原计划：Qwen2.5-7B V-only（BF16 K + BF16/INT8/FP8/INT4 V）、14B V-only、14B Joint，以及 per-head/block scales、RHT/orthogonal rotation、clamp/outlier 处理。**当前 v4.2 仅支持 14B native BF16/FP8 Joint，不能用它直接跑 V-only/INT8/INT4**；开发独立版本并保持相同 backend 时才构成可比 Pareto 对照。探索基于层/头/token 工作负载性质的可解释静态规则；若规则不够，再考虑轻量预测器、策略网络或在线优化，必须计入决策成本及系统复杂度。

## 终止/改向的科研规则

- 如果 C++ binary/Graph 路径不完整，终止该批 Graph 性能扩展，优先定位具体配置/Kernel 失败。
- 如果固定 scales 在质量上严重退化，停止声称可用的产线收益，转向 scale/分层校准；可以保留性能数据作为“受损质量配置下的上界”。
- 如果 FP8 低 C 无加速，不能得出量化无用：查看资源瓶颈，然后把试验预算集中到长上下文/高 C；若所有强压力仍无优势，寻找根因并提出负结果。
- 服务器繁忙或 SSH 故障时，继续本地数据科学和代码工作，不因强行追 GPU 破坏共享环境。
