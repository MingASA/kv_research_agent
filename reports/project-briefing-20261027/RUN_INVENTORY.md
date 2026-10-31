# 全部45条run的证据分类

这是本报告冻结的索引快照，每行是一个运行/恢复/未启动/待验记录；不是45个成功实验，也不是45份独立用户样本。现代主服务、质量和机制结论见各专题，原状态与failed/conflict完整保留。后续新增任务不在此快照中。

来源：[45行原始CSV快照](data/INDEX_SNAPSHOT.csv)，保留原完整source/binary/dataset/GPU/manifest/validity/notes字段。下面以中文说明能计入什么结论。

| 行 | run_id | 原索引状态 | 汇报解释 |
| --- | --- | --- | --- |
| 1 | 568b70f2bb5a422faf1dd93fde28a8a2 | [COMPLETE_VERIFIED](../../reports/experiments/ceff-takeover-20261026T040426Z-v2/cases.csv) | 旧同FA2配对的单次run；六run共九条件/18case，不是现代生产参考。 |
| 2 | 88830c2a50ed48b8979b9612fc9adcd6 | [COMPLETE_VERIFIED](../../reports/experiments/ceff-takeover-20261026T040426Z-v2/cases.csv) | 旧同FA2配对的单次run；六run共九条件/18case，不是现代生产参考。 |
| 3 | 0963d35b8e6c4edcbb40a608223cc7dd | [COMPLETE_VERIFIED](../../reports/experiments/ceff-takeover-20261026T040426Z-v2/cases.csv) | 旧同FA2配对的单次run；六run共九条件/18case，不是现代生产参考。 |
| 4 | 79eed0cd8b9f44d3bf36d395a15fe4ec | [COMPLETE_VERIFIED](../../reports/experiments/ceff-takeover-20261026T040426Z-v2/cases.csv) | 旧同FA2配对的单次run；六run共九条件/18case，不是现代生产参考。 |
| 5 | 2eafdc95937a4b87b378cd8ce4eaa90a | [COMPLETE_VERIFIED](../../reports/experiments/ceff-takeover-20261026T040426Z-v2/cases.csv) | 旧同FA2配对的单次run；六run共九条件/18case，不是现代生产参考。 |
| 6 | 59c3c9df7558430b88ee00e51493a2d2 | [COMPLETE_VERIFIED](../../reports/experiments/ceff-takeover-20261026T040426Z-v2/cases.csv) | 旧同FA2配对的单次run；六run共九条件/18case，不是现代生产参考。 |
| 7 | ec931de67e0d4fb9b86819efe1fe3552 | FAILED | 旧binary失败；不能与ceff FP8拼配对。 |
| 8 | 578f3ae8238e44cab3187c7a09a26b8f | INCOMPLETE | 旧binary不完整，旧单臂保留。 |
| 9 | joint14b-full-quality-v1-8gpu | REPORTED_ARCHIVE_VERIFIED | 四臂1719/arm历史汇总档案SHA一致；未独立重评分旧prediction。 |
| 10 | native-fixed16k512-c4-r0-20261026a | [STOPPED_PRODUCTION_SCOPE_CHANGE](../../reports/operations/20261026T075846Z/formal16_after_stop.json) | 因最高目标转生产代表性而停止；无有效完整配对结论。 |
| 11 | prod-sharegpt-fa3-c128-r0-20261026b | [RECOVERED_MEASUREMENTS_RUNTIME_CONFLICT](../../reports/operations/20261026T075846Z/production_sharegpt_independent_audit_v1.json) | 首次FA3-Q候选ShareGPT测量恢复；原CONFLICT保留，RPS约+18.71%。 |
| 12 | prod-quality-fa3-pilot80-r0-20261026b | [COMPLETE_PILOT_INDEPENDENTLY_AUDITED](../../reports/experiments/production-fa3-quality-pilot-20261026/audit.json) | FA3 pilot80/arm，仅接口与明显风险；不能宣称1719质量。 |
| 13 | prod-quality-fa3-completion1639-r0-20261026b | [COMPLETE_FRAGMENT_AND_FULL_COHORT_INDEPENDENTLY_AUDITED](../../reports/experiments/production-fa3-quality-full1719-20261026/audit.json) | FA3剩余1639/arm；与pilot精确合并1719，非新增1719题。 |
| 14 | prod-quality-xqa-pilot80-r0-20261026a | [FAILED_OPERATOR_REFERENCE](../../reports/operations/20261026T075846Z/xqa_status_a2.json) | XQA NaN尾页算子失败；没有模型/pilot。 |
| 15 | prod-quality-xqa-pilot80-r0-20261026b | [FAILED_PADDED_GRAPH_REFERENCE](../../reports/operations/20261026T075846Z/xqa_status_b3.json) | padded参考夹具页表storage别名错误；没有模型/pilot。 |
| 16 | prod-quality-xqa-pilot80-r0-20261026c | [COMPLETE_PILOT_INDEPENDENTLY_AUDITED](../../reports/operations/20261026T075846Z/xqa_quality_independent_audit_c1.json) | XQA c算子/Graph/pilot80完成；全量仍需补1639。 |
| 17 | prod-quality-xqa-completion1639-r0-20261026a | [RECOVERED_FP8_FRAGMENT_CONFLICT_PRESERVED](../../reports/operations/20261026T075846Z/xqa_completion_fp8_recovered_audit_a1.json) | FP8剩余1639完整恢复，原CONFLICT；BF16缺臂，不重跑成功预测。 |
| 18 | prod-quality-xqa-missingbf16-1639-r0-20261026b | [COMPLETE_MISSING_BF16_AND_FULL_COHORT_INDEPENDENTLY_AUDITED](../../reports/experiments/production-xqa-quality-full1719-20261026/audit.json) | 仅补缺失BF161639，三run合成1719完整质量。 |
| 19 | prod-sharegpt-xqa-c128-r0-20261026a | [NOT_LAUNCHED_ABSENCE_INDEPENDENTLY_CONFIRMED](../../reports/operations/20261026T075846Z/sharegpt_xqa_absence_resolved_a5.json) | prepare未启动、独立absence核验，0GPU-h，非重复。 |
| 20 | prod-sharegpt-xqa-c128-r1-20261026b | [NOT_LAUNCHED_ABSENCE_INDEPENDENTLY_CONFIRMED](../../reports/operations/20261026T075846Z/sharegpt_xqa_absence_resolved_b6.json) | prepare未启动、独立absence核验，0GPU-h，非重复。 |
| 21 | prod-sharegpt-xqa-c128-r2-20261027a | [COMPLETE_PAIR_INDEPENDENTLY_AUDITED](../../reports/experiments/production-xqa-sharegpt-20261027/audit.json) | 旧FA3-decode BF16参考下ShareGPT；+152.58%不能冒称现代增益。 |
| 22 | prod-trace-xqa-toolagent-s1-r0-20261026d | [NOT_LAUNCHED_ABSENCE_INDEPENDENTLY_CONFIRMED](../../reports/operations/20261026T075846Z/trace_toolagent_xqa_absence_resolved_d3.json) | ToolAgent prepare未启动，0GPU-h。 |
| 23 | prod-trace-xqa-toolagent-s1-r1-20261027b | [NOT_LAUNCHED_FROZEN_POLICY_MISMATCH](../../reports/operations/20261026T075846Z/trace_toolagent_xqa_status_xz_b2.json) | 冻结policy不匹配拒绝；没有正式trace测量。 |
| 24 | prod-trace-xqa-toolagent-s1-r2-20261027c | [COMPLETE_TRACE_INDEPENDENTLY_AUDITED](../../reports/experiments/production-xqa-toolagent-original-20261027/audit_summary.json) | 旧参考原速匿名ToolAgent2088；正式271→528/1689成功。 |
| 25 | prod-trace-xqa-burstgpt-s1-r0-20261027f | [COMPLETE_TRACE_INDEPENDENTLY_AUDITED](../../reports/experiments/production-xqa-burstgpt-original-20261027/audit_summary.json) | 旧参考BurstGPT6385；正式6221/6221均成功，无吞吐增量。 |
| 26 | prod-trace-xqa-conversation-s1-r0-20261027f | [CONFLICT_MEASUREMENTS_INDEPENDENTLY_RECOVERED](../../reports/experiments/production-xqa-conversation-original-20261027/audit_summary.json) | 旧参考原conversation，末响应后8.901s冲突；只恢复测量不改COMPLETE。 |
| 27 | prod-trace-xqa-synthetic-s1-r0-20261027f | [COMPLETE_TRACE_INDEPENDENTLY_AUDITED](../../reports/experiments/production-xqa-synthetic-original-20261027/audit_summary.json) | 旧参考原synthetic；多完成但生成期中位更慢。 |
| 28 | prod-sharegpt-bf16xqa-c128-r0-20261027a | [CONFLICT_AFTER_OPERATOR_BEFORE_SERVICE](../../reports/operations/20261026T075846Z/bf16_xqa_reference_status_a3.json) | 新BF16 XQA算子通过后归属保护停止；无服务measurement。 |
| 29 | prod-sharegpt-bf16xqa-c128-r1-20261027b | [COMPLETE_SERVING_INDEPENDENTLY_AUDITED](../../reports/experiments/production-bf16-xqa-sharegpt-20261027/audit_summary.json) | 现代ShareGPT第一配对，BF16→FP8，同GPU，正式512/arm。 |
| 30 | prod-trace-bf16xqa-toolagent-s0125-r0-20261027a | [COMPLETE_TRACE_INDEPENDENTLY_AUDITED](../../reports/experiments/production-bf16-xqa-toolagent-s0125-20261027/audit_summary.json) | 现代ToolAgent s=.125，2088全成功；正式1689、无吞吐增益，尾延迟更慢。 |
| 31 | prod-trace-bf16xqa-toolagent-s025-r0-20261027b | [COMPLETE_TRACE_INDEPENDENTLY_AUDITED](../../reports/experiments/production-bf16-xqa-toolagent-s025-20261027/audit_summary.json) | 现代ToolAgent s=.25，2088全成功；TTFT改善、生成期更慢。 |
| 32 | prod-quality-bf16xqa-full1719-r0-20261027a | [COMPLETE_FULL1719_REFERENCE_INDEPENDENTLY_RESCORED](../../reports/experiments/production-bf16-xqa-quality-reference-20261027/audit_summary.json) | 新BF16 C8单臂完整1719，与已保留FP8跨run比较；未新跑FP8。 |
| 33 | prod-sharegpt-bf16xqa-c128-r2-20261027c | [COMPLETE_SERVING_INDEPENDENTLY_AUDITED](../../reports/experiments/production-bf16-xqa-sharegpt-replicates-20261027/audit_summary.json) | 现代ShareGPT反序FP8→BF16，同GPU同528来源；第二paired run。 |
| 34 | prod-quality-bf16xqa-full1719-c128-r0-20261027b | [COMPLETE_FULL1719_C128_INDEPENDENTLY_RESCORED](../../reports/experiments/production-bf16-xqa-quality-c128-20261027/audit_summary.json) | C128双臂完整1719官方重评分；LB−1.7705pp，未获质量接受。 |
| 35 | prod-diag-bf16xqa-toolagent-s0125-r0-20261027a | [NOT_LAUNCHED_ABSENCE_INDEPENDENTLY_CONFIRMED](../../reports/operations/20261026T075846Z/component_diagnostic_prepare_absence_resolved_a2.json) | 组件prepare压缩白名单拒绝于mkdir前，0GPU-h。 |
| 36 | prod-diag-bf16xqa-toolagent-s0125-r1-20261027b | [FAILED_PROFILER_PROBE_BEFORE_MODEL_GPU_RELEASED](../../reports/operations/20261026T075846Z/component_diagnostic_status_b3.json) | profiler annotation类别误计；14B前失败，费用计入。 |
| 37 | prod-diag-bf16xqa-toolagent-s0125-r2-20261027c | [COMPLETE_INDEPENDENT_DIAGNOSTIC_AUDIT_PASSED](../../reports/operations/20261026T075846Z/component_diagnostic_independent_audit_c1.json) | 组件c每臂637请求/3窗口/48forward；机制诊断，不是headline吞吐重复。 |
| 38 | prod-tools-bf16xqa-bfcl258-c128-r0-20261027a | [FAILED_BEFORE_AIPERF](../../reports/operations/20261026T075846Z/native_tool_stream_failure_a_independent_evidence_v1.json) | stream工具BF16首warm参数JSON失败；无AIPerf/FP8。 |
| 39 | prod-tools-bf16xqa-bfcl258-buffered-c128-r0-20261027b | [COMPLETE_NATIVE_TOOLS_INDEPENDENTLY_AUDITED](../../reports/experiments/production-native-tools-20261027/audit_summary.json) | buffered258+16warm；182正确/arm、7up7down，FP8单条重复到cap完整保留。 |
| 40 | prod-prefill-workspace-op-r0-20261027a | [COMPLETE_OPERATOR_ONLY_INDEPENDENTLY_AUDITED](../../reports/experiments/production-prefill-workspace-20261027/REPORT.md) | prefill BF16工作区算子数值/Graph通过；约.6%收益与额外1.5GiB，停止全模型扩大。 |
| 41 | prod-calib-bf16xqa-heldout128-r0-20261027a | [FAILED_AIPERF_CLI_ZERO_WARMUP_BEFORE_FORMAL_REQUESTS](../../reports/operations/20261026T075846Z/range_probe_a_independent_failure_evidence_v1.json) | range a显式warmup0被CLI拒绝；无正式128请求/模型统计。 |
| 42 | prod-calib-bf16xqa-heldout128-r1-20261027b | [FAILED_UNICODE_POSTAUDIT_RAW128_RECOVERED_NO_KV_EXPORT](../../reports/operations/20261026T075846Z/range_probe_b_independent_raw_recovery_v1.json) | range b原128结果独立恢复；Unicode拆行错误，统计导出前清理，不能给scale。 |
| 43 | prod-calib-bf16xqa-heldout128-r2-20261027c | [FAILED_POSTPROCESS_COMPLETE_OBSERVATIONS_INDEPENDENTLY_RECOVERED](../../reports/experiments/production-kv-range-probe-20261027/REPORT.md) | range c原FAILED；ISO精度误拒绝，128结果/1536统计独立恢复，不重跑GPU。 |
| 44 | prod-bf16q-decode-op-r0-20261027a | [COMPLETE_INDEPENDENT_OPERATOR_AUDIT_PASSED](../../reports/experiments/production-bf16q-decode-20261027/REPORT.md) | 新BF16 Q decode仅算子/动态Graph128独立完整性验收通过；无14B/AIPerf/质量或服务收益。 |
| 45 | prod-quality-bf16q-full1719-c128-r0-20261027a | [FAILED_BEFORE_MODEL_PLAN_ENVIRONMENT](../../reports/operations/20261026T075846Z/bf16q_quality_status_a2.json)；索引快照原文：FAILED_BEFORE_MODEL_PLAN_ENVIRONMENT | 原FAILED_BEFORE_MODEL_PLAN_ENVIRONMENT、driver释放；72文件独立失败审计通过，零预测/无measurement。实际.001074001823888889GPU-h已结算，累计9.494151518055556；截至45 reserved0，本地b不计结果。 |

第44条已完成实际算子与独立完整性审计，以[最新独立审计](../operations/20261026T075846Z/bf16q_decode_operator_independent_audit_a1.json)为准，没有模型质量/服务收益。第45条原FAILED/driver释放、72文件独立失败审计与实际费用均已核对结算，以[独立失败审计](../operations/20261026T075846Z/bf16q_quality_a_independent_failure_evidence_v1.json)和[截止覆盖记录](data/BRIEFING_CUTOFF.json)为准。原45行快照保持，当时本地b的CPU准备不计模型结果；若有第46后续，单独列于下节。

历史未启动/失败没有被删除，不把probe修复a/b/c当统计重复；未重新运行任何成功GPU实验。本汇报仅生成本新目录的文档、附表与来源核查。

## 原45条之后：第46单独补充

原45行CSV/STATE与原16表保持字节不变，第46不混入原快照。新b修复a的模型前失败，未重跑成功预测，不增加独立用户。

| 后续序号 | run_id | 最新独立证据 | 决策及限制 |
| --- | --- | --- | --- |
| 46 | prod-quality-bf16q-full1719-c128-r1-20261027b | [原COMPLETE/1719完整重评分独立验收](../../reports/operations/20261026T075846Z/bf16q_full1719_quality_independent_audit_b1.json) | 未观察整体质量修复；停止该BF16Q_FA2候选额外服务GPU矩阵，无新AIPerf、非Q单因子 |

[第46单行原索引摘录](data/POST45_INDEX_RUN46.csv)可能保留取证时pending文字，以更晚完整audit为准；[06完整专题](06_第46条BF16Q完整质量与停止决定.md)、[后续证据/成本](data/POST45_EVIDENCE_RUN46.json)、[封存UTC与下次续接边界](data/REPORT_SEAL.json)。仅在用户再要求时从run46之后续接，不自动跟踪新实验。
