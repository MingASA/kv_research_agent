# BF16 Q + FP8 KV：H800数值和生产Graph128门槛通过

2026-10-27。实际算子证据已独立验收，未新增模型质量或服务性能结论。现有支持建议仍是 BF16 默认、FP8 按模型/业务条件可选；完整生产支持尚未完成。

CPU源码核实：C++不限制batch16，Python Graph支持128；官方 v0.6.18.post1 FA2在kernel内把FP8 K/V片段转成BF16，保留BF16 Q并用BF16 MMA，提供可实际部署的精度/速度候选。H800实际安装decode.py、prefill.cuh、mma.cuh均逐SHA匹配该官方tag。来源与SHA保存在冻结输入，官方源码见 [prefill.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18.post1/include/flashinfer/attention/prefill.cuh) 与 [mma.cuh](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18.post1/include/flashinfer/mma.cuh)。数值检验与该精度路径一致；这不能证明原模型质量下降来自Q，也不是模型服务收益。

本地实现78项检查/旧576种配置逐项一致/控制器203项通过，源代码78004505e2db44e88994899e8ea3e25c109978a7，独立分支production/fp8-bf16q-decode-v1。详细准入和停止规则见 plans/experiments/production-fp8-bf16q-decode-v1.md。

| 实际H800检查 | 相对L2（BF16 Q / representable FP8 KV dense参考） |
|---|---:|
| prefill17query/257context | .002796911 |
| 小Graph capture/replay | .002805407 |
| 缩短context130 / padded4 | .002733474 |
| 同Graph128，128live/context257 | .002245334 |
| 同Graph128，126live/context130 | .002214775 |
| 同Graph128，1live/context130 | .002164884 |
| 同Graph128，127live/context257 | .002161153 |

全部小于冻结门槛.01，另有逐元素rtol/atol检查。同Graph128含真实FP8 writer与attention，QKV保留packed token stride；动态indptr/last-page/padding、resident与metadata指针固定、page0 payload不变、重复bind保留live prefix通过。只有单层fixture，Graph128数据context为130/257；完整模型/长文/真实调度与质量仍须后续检查，不从此表推端到端或Q量化因果。

run prod-bf16q-decode-op-r0-20261027a 原始COMPLETE、measurement与日志一致，55文件/369269bytes逐SHA。无14B加载、无AIPerf，实际.042364935 GPU-h，driver释放，总9.4930775161、预留0。独立审计没有重新执行GPU数值算子，只验证封存原始证据与contract：reports/operations/20261026T075846Z/bf16q_decode_operator_independent_audit_a1.json；镜像results/mirrors/prod-bf16q-decode-op-r0-20261027a-terminal-a1/files。

下一步在新的质量工具分支扩展明确BF16Q profile，固定同生产C128/Graph/prefix/scales/input1719与原官方scorer，复用已审计现代BF16参考。只有质量取舍有实际改善才继续必要目标服务；不重复旧算子run，不用小fixture认证模型。
