# 有界BF16 prefill工作区：数值通过，停止当前候选扩展

H800/SM90、source8af6e702的operator-only验证已COMPLETE并driver释放；52原始文件/353804bytes全部SHA通过，原始数值日志、真实capture/replay、改context/padded bucket、共享页去重/非连续FP8 stride/active-only转换/跨层复用/下批active更新通过。dense-reference prefill相对L2 .002285，decode/Graph约.0233–.0247；独立审计核对封存证据，未重复GPU执行。

固定记录形状来自真实组件诊断c window index2/forward0：1seq/510query/1022KV/48层。使用随机fixture值，双方同Q/resident FP8 pages，8次交替臂序。计时包含host planner、活跃页收集和48次materialize+attention，不含完整模型、服务或质量。

| 中位计时 | 直接resident FP8 FA2 | 单层BF16工作区FA3 | 降幅 |
|---|---:|---:|---:|
| CUDA elapsed，ms | 3.815776 | 3.793008 | .5967% |
| wall，ms | 3.839516 | 3.818228 | .5545% |

差异小且原始样本重叠，不称统计显著或有生产收益；不从8个fixture样本外推run级置信区间。总额外预留1613758468B（1536MiB数据加3MiB+4B元数据），新真实14B净容量尚未测量。该形状未显示足以抵消显存/维护成本的部署价值，停止当前候选的完整模型/质量/AIPerf扩展，保留代码和负成本证据；不是证明所有FA3、融合或长上下文实现均无价值。

本run没有加载14B、没有AIPerf、没有模型任务质量或服务吞吐结果。measurement_success只表示明确operator gate完成。实际占用.0922060775GPU-h（含JIT/失败保护/回收），累计9.1379426631/初始12、无预留/active/UNKNOWN。未改共享依赖/其他任务，旧结果不重跑。

证据：reports/operations/20261026T075846Z/prefill_workspace_independent_audit_a1.json；终态prefill_workspace_status_a7.json；镜像results/mirrors/prod-prefill-workspace-op-r0-20261027a-terminal-c1/files。可复算scripts/audit_prefill_workspace_run.py。

下一步回到已证实的质量风险，先使用未进入服务528样本的128条真实ShareGPT请求，采集有界跨层/phase K/V分布，检查固定K=.125/V=.0625的实际饱和/下溢。保持BF16 FA3+XQA/生产Graph，GPU聚合后在无请求的边界导出；不以旧首次prefill/decode observer冒充全覆盖，不用质量答案调参，不把短聊天外推长文/代码。
