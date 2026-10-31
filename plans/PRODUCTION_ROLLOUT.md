# xLLM KV量化生产支持决策推进表

最高依据为 `ops/PRODUCTION_DECISION_STANDARD.md`；不是论文实验矩阵。后续工作按生产收益和可落地性调整，不机械把全部格子测完。

## 首项服务评估（proposal → 通过门槛后执行）

- 模型：现有只读Qwen2.5-14B-Instruct，H800/SM90单卡，ceff既有二进制；新Python源码独立commit `7831a874`，仅Python改变，不重编译/覆盖旧binary。
- 候选：FA3 BF16-Q/BF16-KV，与FA3 FP8-Q/FP8-KV；K=.125/V=.0625，Q=1.0静态scale，显式配置，不能称已经完成质量校准。FP8-Q由单次Triton转换完成；无整个BF16 shadow cache。先真实单层算子/partial-page/Graph重放正确性门槛，再真实模型Graph和服务。
- 输入：现存ShareGPT94145完整记录的已验证source SHA `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`，均匀抽1024原对话，确定性打乱；按原文构造最后一个真实user请求及已有会话历史，1007可用。只排除无有效user/未知角色记录，不改写文本或强拼长度。当前CPU prompt min30/P501284/P952003/max3618；须以服务端实际usage复核，不把CPU估计当实测。
- 自然EOS、生成上限1024、greedy作为可实际部署的确定性聊天配置；没有强制输出512。输入全体与执行子集分别标记，首批512 profiling+16独立warmup，请求唯一，不循环假造更多独立用户。
- 服务：prefix cache开启、chunked prefill开启，实际项目token budget默认10240、并发请求上限200、显存比例默认0.8；max_seqs128给完整支持的Graph批次范围，AIPerf client concurrency128。检验正常持续需求及基于真实请求/生成上限的容量排队压力，不主动制造OOM。
- 请求内容是真实对话；到达方式是明确的closed-loop压力，不称为原始生产时间线。两臂各独立新缓存、相同输入顺序；导出真实HTTP payload与完整SSE/usage/finish reason，逐请求核验，允许合法单token回答（其ITL不定义）。
- 报告成功request/s、实际token/s、TTFT/ITL/E2E P50/95/99、自然输出长度/length-cap比例、分配容量/NVML、实际占卡及启动成本。质量未补足前仅为服务候选测量，不给净收益/可上线结论。

## 原项目覆盖，不遗忘其他workload

| 家族 | 已核实材料 | 生产用途与进入条件 |
|---|---|---|
| ShareGPT | 完整缓存672837942bytes、94145对话；1007真实上下文请求冻结 | 首项聊天服务与真实生成分布；后续按实际服务压力适配配置，不泛化到所有请求类型 |
| BurstGPT | `traces/BurstGPT_1.csv` 50853373bytes；已有peak10m冻结211466bytes | 核实原始时间单位、请求/响应长度和窗口；保留突发/到达而非改成任意闭环；长度trace不能冒称有真实对话文本 |
| Mooncake conversation | 原始source3029533bytes，旧264冻结及manifest存在 | 到达、上下文与prefix复用；先核对截取/抽样损失、原始hash block到服务payload的语义，不直接套旧no-fixed-schedule |
| Mooncake toolagent | 原始source4415857bytes，旧264冻结及manifest存在 | 多轮/agent工作集、前缀复用和高KV驻留；同样先验证loader真实请求/时间语义 |
| Mooncake synthetic | 项目路线图列出，当前具体文件/语义仍待实机核实 | 若原始生产trace代表真实长度/到达，可以作为匿名trace负载；没有生产意义的随机长度拼接不跑 |
| Code / agentic / RAG | 原计划方向；尚无已确认的全套新原生回放 | 只选实际部署会遇到、来源与格式清晰的任务；可用真实对话历史/文档请求，不用人为无关文字替代 |
| GSM8K / LongBench | 冻结1719样本与官方评分器；80条CPU pilot准备 | 候选质量门槛，和吞吐负载分开；需生产容器真实依赖/新Q量化质量验收，pilot不能声称全量通过 |
| 7B V-only / 14B Joint | 历史实现和归档不同 | 按生产候选价值选取，不混成同一精度或内核效应，也不为覆盖INT8/INT4等所有格子而跑 |

低负载、持续高压、突发和容量限制都须用相关真实请求/trace覆盖。SLO数字尚未由用户设定，不假造用户门槛；先呈现完整服务曲线、退化和资源成本，形成明确的支持条件。新版FA3/XQA/融合writer等仅在兼容/正确/质量与收益明确后进入实际候选；不会为了“同后端”而故意限制BF16或FP8到旧慢路径。

## 独立评审的核实

本机源码确实原先固定FA2，FP8-Q是新增数值近似。现装0.6.18.post1头文件已经有shared-memory repack，不能将旧SM120问题直接当H800缺补丁证据。上游[问题3484](https://github.com/flashinfer-ai/flashinfer/issues/3484)、[修复3485](https://github.com/flashinfer-ai/flashinfer/pull/3485)及[vLLM FP8说明](https://docs.vllm.ai/en/v0.18.2/features/quantization/quantized_kvcache/)支持分清KV存储与FP8算术；真实SM90性能及本候选质量仍须实测。现装FA3明确拒绝BF16-Q/FP8-KV组合，故不能只改planner字符串。

当前旧FA2轻压力九组只证明该配置下的服务表现与存储压缩。新决策以可用现代实现、实际生产负载、质量与有效服务成本为准，不以找出量化“必须加速”为目标。
