# xLLM KV Cache 量化项目汇报（截至 2026-10-27）

**截至第46条的建议：保留BF16默认，把原FP8-XQA路径作为限定模型、后端与业务的显式可选能力继续工程化；停止扩大本次BF16Q_FA2候选的额外服务GPU矩阵。完整生产支持尚未完成。** 原路径实际驻留KV容量约1.94倍、真实ShareGPT两次反序仍约+32%RPS，但C128 LongBench下降1.7705pp，低压尾延迟和原生工具存在风险。新候选完整1719质量独立验收后，LongBench仍比现代BF16低2.1491572pp，未观察整体修复；没有为它新跑AIPerf，不能继承原+32%。质量/SLO损失界未获用户接受。[原路径性能](../experiments/production-bf16-xqa-sharegpt-replicates-20261027/REPORT.md)、[原路径质量](../experiments/production-bf16-xqa-quality-c128-20261027/REPORT.md)、[第46补充与停止决定](06_第46条BF16Q完整质量与停止决定.md)

本汇报的目标是回答“xLLM 值不值得、以什么边界完整支持 KV 量化”，依据真实请求、自然停止、实际可部署的 H800 优化后端、成功率、尾延迟、任务质量与维护成本组织。历史同 FA2 和合成矩阵保留为局部证据，不作为永久主生产约束。[最高生产标准](../../ops/PRODUCTION_DECISION_STANDARD.md)

## 建议的汇报顺序

| 文档 | 回答的问题 | 主要内容 |
| --- | --- | --- |
| [01 服务、容量与成本](01_生产服务容量与成本.md) | 压缩是否真实、什么时候快、什么时候慢？ | 驻留字节与可用页；现代 ShareGPT 两次配对；现代 ToolAgent 两压力点；旧参考四 trace；成功率、实际 token、TTFT/ITL/E2E、成本边界 |
| [02 质量与工具任务](02_质量与原生工具任务.md) | 吞吐收益付出了什么质量代价？ | C128 全量 1719；C8 与历史 FA3 候选；逐题变化和 bootstrap；代码模式；完整 JSON 原生 BFCL；流式接口问题 |
| [03 算子、机制与工程取舍](03_算子机制与维护取舍.md) | 原因找到了多少、做过哪些修复、哪些投入应停止？ | Q/KV/后端关系；真实 Graph 组件计时；1.5 GiB prefill 工作区负候选；真实 K/V 分布；版本与维护契约；BF16 Q decode 算子入口 |
| [04 历史结果与故障闭环](04_历史结果与故障闭环.md) | 从早期实现到现在做过什么，失败是否被掩盖？ | 旧 Joint 四臂质量、7B V-only 的证据等级；ceff 九条件；旧 binary 门禁；NaN/夹具别名/退出竞态/取证与后处理恢复 |
| [05 证据边界与支持决策](05_证据边界与剩余决策.md) | 这些结论能支持怎样的产品决定，还缺什么？ | 证据等级、指标定义、样本独立性、资源账、支持边界、剩余决策证据与停止条件 |
| [06 第46条质量与停止决定](06_第46条BF16Q完整质量与停止决定.md) | 保留BF16 Q是否修复质量，是否继续投入此候选？ | 三配置全部任务/配对区间/逐题与cap、9任务有限buffered cohort时钟、取证恢复与成本；停止该候选额外GPU服务矩阵 |
| [来源与核查](SOURCES.md) | 每个数字来自哪里、能否复核？ | 精确报告/JSON/CSV 路径、SHA-256、关键字段、静态链接与数值检查结果 |
| [完整 run 清单](RUN_INVENTORY.md) | 索引中的每个 run 最终算什么？ | 原45行只读快照保留；第46另列补充，不把索引行数当成功实验数 |

## 可直接用于口头汇报的主线

1. **压缩实现已经跨过“是否真省 KV”的门槛。** 14B/GQA 8 KV heads/48 层上，4096→2112 B/token/layer，含 FP8 payload 与 scale，无完整 BF16 影子；同预算可用 token 容量 188928→366592。[容量原审计](../operations/20261026T075846Z/toolagent_s0125_independent_audit_v1.json)
2. **聊天服务有可重复的配置级收益。** BF16 也采用 XQA decode 后，两次相反臂序的同 GPU 配对，RPS +31.507%～+32.965%、实际输出 TPS +29.653%～+30.559%。时长汇总 RPS +32.233%，对应 observation 内 GPU 时间/完成请求约 −24.38%；这不是含加载与质量接受的完整账单。[反序比较](../operations/20261026T075846Z/modern_BF16_ShareGPT_ABBA_comparison_v1.json)
3. **更多容量没有给出普遍加速。** ToolAgent 低压全成功而吞吐相同，E2E P99 24.276→34.099 s；中压 TTFT P95 28.805→10.410 s，但完成/token 中位 54.439→85.979 ms，E2E P99 仍略慢。排队改善与生成期变慢可以并存。[两点比较](../operations/20261026T075846Z/modern_BF16_ToolAgent_pressure_comparison_v1.json)
4. **质量是当前默认启用的硬缺口。** 同生产 C128 完整 1719 题，GSM flexible +0.5307 pp；LongBench macro −1.7705 pp，配对题目重采样 95% 区间 [−3.4893, −0.1667] pp。MultiFieldQA、MuSiQue、LCC 分别约 −4.00/−4.49/−3.92 pp。[逐任务与分析](../operations/20261026T075846Z/production_C128_quality_analysis_b1.json)
5. **原生工具任务说明总准确率会漏掉服务风险。** 两臂都 182/258 匹配正确，但有 7 up/7 down；FP8 一条重复生成到 1024 cap，整批 observation 7.856→16.135 s，不能据同分认为服务等价。流式参数增量排序缺陷还需独立修复。[工具报告](../experiments/production-native-tools-20261027/REPORT.md)
6. **机制研究已开始约束投入方向。** 实际诊断中已细分 writer 的累计设备时间占全 kernel <1%，Graph 内部未单独分解；BF16 prefill 工作区算子正确但仅约 0.6% 改善且额外 1539 MiB+4 B，已停止扩大；heldout128 的真实 K/V 无饱和，保持现有 scale。当前证据没有确认 Q、K、V、某个 kernel 或 batch 为模型质量下降的唯一根因。[组件](../experiments/production-component-diagnostic-20261027/REPORT.md)、[工作区](../experiments/production-prefill-workspace-20261027/REPORT.md)、[范围](../experiments/production-kv-range-probe-20261027/REPORT.md)
7. **可部署BF16 Q候选已完成模型质量闭环，但没有整体修复。** 第46新候选LB48.0393290%，对现代BF16−2.1491572pp；对原FP8-XQA−.3786367pp、95%区间跨0，不能称显著更差，也不能称等价或修复。停止其额外GPU服务扩展，不否定原FP8-XQA条件收益，也不证明Q无影响。[第46完整补充](06_第46条BF16Q完整质量与停止决定.md)、[现成质量比较图](../experiments/production-bf16q-quality-20261027/figures-v1/quality_deltas.svg)

## 汇报的固定边界

主结果限定 Qwen2.5-14B-Instruct、TP1、H800/SM90、当前冻结后端和固定 K=.125/V=.0625。模型权重仍为 BF16。BF16/FP8 的 prefill、decode 内部 Q 精度和 writer 不全相同，结论属于可部署配置的比较，不能标成只改变 KV dtype 的因果效应。Mooncake/BurstGPT 在本项目是匿名长度/到达/prefix 仿真；真实语义/自然 EOS 证据来自 ShareGPT 与原生质量/工具任务。

原45条层保留原字节：第44是算子/Graph128成功，第45是模型前FAILED、零预测；截至45已结算9.494151518055556 GPU-h，见[原截止记录](data/BRIEFING_CUTOFF.json)。后续第46 `prod-quality-bf16q-full1719-c128-r1-20261027b` 已原COMPLETE/driver释放、123文件/44462952B与完整1719质量独立验收，实际.22991220958916667 GPU-h；截至46结算累计**9.724063727777779 GPU-h**，reserved0、无active/UNKNOWN。它修复a的模型前编排失败，不是重复成功预测或新独立用户。[第46单行摘录](data/POST45_INDEX_RUN46.csv)、[第46证据/成本/停止决定](data/POST45_EVIDENCE_RUN46.json)、[完整专题](06_第46条BF16Q完整质量与停止决定.md)

本轮汇报**封存至第46条**，确切UTC见[封存记录](data/REPORT_SEAL.json)。后续实验不自动更新本报告；仅在用户再次要求时从run46之后续接。原45行INDEX/STATE与16张表不被后续记录覆盖。本汇报未进行SSH、Docker、GPU调度、模型预测或重评分，也不修改主STATE/INDEX/代码或Git。

现有[生产证据图 SVG](../synthesis/production-decision-20261027/figures-v1/production_evidence.svg)可用于展示此前服务/质量结论；该图未覆盖后来的原生工具、工作区、范围探针与新 decode 算子，本汇报的相应表格补足这些内容。
