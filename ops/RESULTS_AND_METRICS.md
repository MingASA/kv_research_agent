# 证据、指标与验收规范

## 当前生产协议（2026-10-26更新，优先于下文历史128+8/FA2约束）

以用户生产决策目标和每个不可变manifest为准。BF16/FP8可使用各自最佳可部署算子，明确prefill/decode、内部Q精度、K/V scales、Graph、writer与缓存初始化契约；不要把不同kernel的部署收益说成纯KV dtype因果效应。样本数来自对应协议：ShareGPT512+16、quality80+不重叠1639、连续trace的完整保留窗口等；不能将所有workload都验成128+8。

真实语义聊天保留自然EOS。Mooncake/BurstGPT匿名trace没有原始文本；为复现token工作量，可按每条原OSL明确设置ignore_eos与独立max_tokens，必须逐200响应验证length/SSE/实际usage，并明示匿名长度/前缀/到达仿真。它不代替自然聊天或质量，不是统一强制512。连续前置60秒是逻辑暖缓存流量，不单独drain/reset，也不自动证明服务器已进入稳态。

完成的trace测量必须保留每个源请求的尝试、唯一ID、真实payload、SSE/usage和错误。429/503/超时是服务结果，计入错误/拒绝，不能偷偷删掉或算成功。AIPerf必须真正结束并drain，不能缺请求、被取消或仅退出0；原summary/error_summary/退出码保留。有效测量可包含服务失败，它不等于100%服务成功。模型/Graph/源码或回放语义错误仍须标INVALID/FAILED，不能当过载现象掩盖。

固定调度使用明确来源时钟/倍率，保留offered、credit-issued、request-start/end。实际安装AIPerf的phase完成日志提供精确epoch ns；naive summary时间不可直接当UTC。报告client发送延后、从offered计起的TTFT/E2E、原HTTP计时延迟、测量cohort成功率和含drain的完成速率。request-start不是服务端arrival，前者延后不能归因服务器排队。ITL为逐请求平均间隔，不是每个token stall的尾部。候选SLO阈值明示未经用户接受。

时序补充（2026-10-27实证）：当前AIPerf request_latency截止最后非空parsed content，metadata.request_end_ns映射最后raw SSE包（可为无内容finish/usage/DONE），不是TCP EOF。匿名ignore-EOS回放可能存在较长的无可见内容尾段，具体token IDs未导出，不能直接证明全为EOS。官方ITL用content时间段和server output count，不能当所有真实decode步骤平均耗时。保留原官方指标与冻结runtime；另外报告offered至末SSE的完整完成延迟，以及(first-content→last-SSE)/(server_output_tokens−1)代理（单token不定义），明确其非硬件step计时/非逐token尾部，不能静默替换原数值。新的qualified速率若使用此代理须单列，阈值仍未经用户接受。证据toolagent_original_content_terminal_gap_CPU_v3.json与可复算脚本scripts/analyze_stream_completion_gap.py，全部已成功测量请求的raw/content/terminal时钟逐项一致。

容量区分resident bytes、physical pages、预留page0后的usable pages和真实SLO承载能力。当前BlockManager的used包含page0、utilization分母排除page0，故`kv_cache_utilization_perc`是比例且满载可能略高于1，不能误标百分数或丢弃该样本。prefix块可能可回收；结合used、waiting、running与命中计数，而非仅free判断容量瓶颈。preempted为迭代gauge，1秒采样的和不是抢占总次数。应用计数和NVML活动比例不是HBM硬件流量。

新输入可有界无损压缩传输：原传输单文件32MiB、总64MiB界不改，仅在本次全新run展开，展开≤64MiB、精确长度和原SHA验证。冻结runtime副本，未来控制器变更不能静默改变包；已标INVALID的本地包拒绝dispatch。CPU/loopback验证不进GPU结果索引。性能和完整质量分别验收；质量有损时报告实际代价，不声称同质量净收益或所有量化路线失败。

## 不可变 Experiment Manifest

每个作业至少记录 `run_id`, UTC timestamp, code git SHA + dirty status, Python scripts sha, C++ binary sha+path, git bundle commit, host/GPU UUID, driver/CUDA/Torch/FlashInfer/Triton/AIPerf versions, model checkpoint provenance, tokenizer hash, dataset source SHA, N requests/warmup, exact ISL/OSL and measured actual tokens, requests arrivals/schedule, concurrency, Graph flags + actual capture/replay proof, kv dtype/bytes/pages/scales, attention backend/prefill/decode/CPU vs GPU settings, prefix-cache state, Docker image/tag/digest if available, command and env (filtered secrets), process/container identification, logs and result pointers`。

## 完整性

成功须同时满足：程序 RC 0；每个 arm 的真实 `COMPLETE`；各并发恰好 128 profiling + 8 warmup（依 Runner 的实际生成记录核对）；请求唯一 ID/无丢失无重复；输出/输入 token 数可信（使用 server counts），Streaming/SSE 完整；无 silent fallback；BF16/FP8 两臂 attention_impl=flashinfer, planner_backend=fa2,paged_prefill=true, quality_transform=false, graph_captured_batches 非空并有 `NATIVE_KV_GRAPH_REPLAY`; `KV_ALLOCATION` 证实 dtype/pages；源码/binary/数据 hash 不漂移。指标文件必须可由原始记录再算，不能只复制 runner summary。

## 性能、显存、质量

性能：request/s, output tokens/s, total tokens/s，TTFT P50/P90/P99, ITL P50/P90/P99, E2E latency 分布、失败率、tokens 分布；若输出长度不一致，报告 per-token normalized 指标，不能用总时间隐含固定相同工作量。相对增速 =(FP8/BF16 -1) 对 throughput; latency decrease =(BF16-FP8)/BF16，并报告 jitter/重复 run CI。注意 `n=128` 的请求并非独立、相同时刻排队相关，误差估计需按 run/batch 分组。

资源：resident KV bytes/token per layer, alloc pages/layer, max supported active tokens/sequences, graph cached memory, peak/reserved GPU memory, GPU active time/energy(若安全可采), HBM bandwidth/kernel latency profiling, 比例与负载条件。BF16 4096B 对 FP8 2112B/层已在 14B eager 看到 ≈48.4% per-token 下降，**不是可直接推广的服务吞吐提升**。

质量：同题 pairing，GSM8K strict/flexible EM，LongBench 8 task scores & macro, BF16→FP8 正向/负向 flip；按上下文段长统计；使用原评分器；两臂共同答错不属于量化降级。区分质量下降情况下的潜在性能增益和“质量可接受”的实际 Pareto 曲线。报告样本覆盖、统计波动/置信区间和局限。

## 防止研究误判

- 不把历史 Triton 4 臂与最新 FlashInfer FA2 比例当作同 kernel 精度加速；不能默认 benchmark 同时涵盖模型质量和 AIPerf 性能。
- 不把 8 卡八份独立任务说成 TP/DP；资源可并行完成测试但单卡实例性能不可假扮单请求 8 卡扩展。
- 不把 CMake/Graph 启动参数或 C++ admission 消息当 GPU Graph replay；不把 Model answer semantic check 与流式接口后端正确性合成一个 PASS。
- 所有 uncertain 结果必须写 `unverified`，任何错误要求主动反驳并保存实证。
