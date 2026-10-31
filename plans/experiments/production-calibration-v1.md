最新2026-10-27T11:59:38.595966+00:00：实际观测已完整独立恢复，原c FAILED保留；无clipping，scale不变，停止本global调参方向。正式结论/局限/原始失败/复算见reports/experiments/production-kv-range-probe-20261027/REPORT.md。以下为历史设计状态。

最新2026-10-27T11:33:03.782463+00:00：b完整128原始服务结果已独立恢复，但8个合法U+2028暴露JSONL分行错误，且旧顺序导致真实GPU统计未导出。原FAILED保留，83文件逐SHA。c只修复物理LF读取和先导出后审计，已冻结/CPU，尚无新scale。以下历史状态以本段和STATE为准。

当前续接（2026-10-27T10:49:21.867649+00:00）：最新现代BF16/XQA C128完整质量LongBench−1.7705pp；source5e2c99f9已实现Graph安全跨请求探针，99/196CPU和真实wire通过、唯一包冻结，尚无新激活/scale或GPU结果。实际设计见production-kv-range-probe-v1.md，以下保留旧提案状态。

# proposal：生产代表性校准与质量修复

背景：XQA部署候选完整1719中GSM flexible+0.4549pp、LongBench macro−1.3358pp、LCC−5.38pp。保留这些原始分数；没有用户接受的质量损失界。当前正式服务测试因SSH不稳定尚未启动，不把此提案替代服务测试或默认扩成GPU矩阵。

confirmed源码事实：ede2b9e7的KVScaleCalibration每层每phase仅记录第一次prefill/decode，输出绝对最大值并按10%余量取2次幂全局K/V scale。向该旧observer发送更多请求不会扩大其观测覆盖。现有K=.125/V=.0625不能因此声称经过生产代表性校准。

本地已冻结128条未进入当前528条ShareGPT服务样本的真实请求。从其余479条按prompt长度四分位各取32，SHA排序seed20261027，逐payload保持原始messages和自然EOS/cap1024。父来源与SHA、原始session/index、排除集合及精确控制字段保留。共143037个CPU估计prompt tokens，范围32–2980；不声称覆盖长文、32K或完整代码生产分布。没有加载质量答案、没有采集模型激活、没有新scale或GPU结果。路径results/prepared/calibration-sharegpt-heldout128-20261027a2；父payload采用含LF的唯一canonical SHA，a2额外封存preparer SHA，与a1的128条payload完全相同，旧CPU提案保留。

后续仅在正式服务数据说明该候选有部署价值时实施：独立分支/非计时BF16诊断，修正observer为跨请求/层/phase/必要decode位置的有界聚合；禁止CUDA Graph捕获内.item或同步，正式性能路径不开observer。记录有效token数、各层/头K/V范围、固定scale覆盖/饱和比例、采样位置与截断，以及新scale的确定规则。先使用独立真实任务输入，不依据评测答案选scale；不把短聊天校准外推到长文/代码。

优先评估可由现有FA2-prefill/XQA-decode消费的固定scalar方案，明确相对当前kernel无全cache BF16 shadow。per-layer/head/block等表示只在实际kernel/API支持并计入维护/运行成本时另立方案，不能偷偷把性能主路径换成慢参考attention。校准数据误差只用于生成候选，不作为生产质量认证。

只重测变更配置需要的算子/Graph、质量pilot与必要完整质量，不重跑已成功的旧配置预测。最终仍以真实任务质量、原计划workload的成功率、延迟和每GPU完成成本决定支持范围。GPU启动必须可信许可、120秒稳定空闲、精确UUID/归属/容器/预算及持久监督全部通过；SSH未知状态先只读恢复，不创建副本。

2026-10-27 CPU表示分析补充：全部正有限BF16位模式的四个E4M3 QDQ对照中，在两个scale共同的normal/未饱和区间，各1761个取值的反量化结果完全相同。K=.125覆盖到56、最小非零2^-12；V=.0625覆盖到28、最小非零2^-13。这说明2次幂scalar变化不会改善所有normal值的相对精度；应先测真实clipping/underflow覆盖，再决定是否值得校准，不能假定换global scale能修复当前质量损失。此为CPU表示数学fixture，不是模型激活频率、Triton/GPU计时或质量归因。可复算脚本scripts/analyze_fp8_power2_scales.py；含脚本SHA/Torch版本的证据fp8_power2_scale_CPU_algebra_v2.json。原v1保留，两版数值一致。
