# 唯一V scale覆盖候选：真实生产配置的完整质量回归

状态：proposal；候选没有FP8新预测，尚无质量修复或服务收益。报告封存于第46条，本次只维护必要实验状态和证据。

依据：第47条真实长文QA128，独立核验91份文件/1536组GPU计数；K最大45.25，V最大37，V有68个prefill饱和，出现在28/46/47层的6个头。V饱和值占71774453760个prefill值的9.474e-10；按各组最大值给出的clipping SSE上界3364.546875，仅为V总体QDQ SSE的8.36636e-5。稀有现象不等于主要质量根因，不能据此提前声称修复。

事前规则来自已在该运行派发前提交的`production-longcontext-kv-range-v1.md`，control commit `ba902582e86eb0f39d1b12a2727dbb94c86a2f5a`：无clipping不调scale；有饱和只取能覆盖观察最大值的最小两次幂scale。故唯一候选K=.125保持不变，V=.0625→.125；不用答案、旧预测或分数搜索。校准记录ID/完整record/context与原LongBench400均无精确重叠，不保证语义/训练集独立。

唯一运行：`prod-quality-xqa-vcover-full1719-c128-r0-20261027a`。新source `1d34d421c219a4497ccf93b2e928d35ee07119aa`，分支`production/fp8-vscale-coverage-v1`。所有推理与scorer源码逐SHA保持原现代XQA参考字节；只有算子检查工具支持显式scale、runner对单臂CPU计划选择正确环境，实际推理仍使用原显式arm环境。无C++变更/新binary，无库升级。真实门槛必须运行候选K/V scale，不允许以旧默认scale门槛代验。

模型、BF16权重、tokenizer、原1719记录/顺序/prompt/官方scorer/自然EOS/各任务cap、128seq/10240tokens/200admission/.8、Graph/prefix/chunked prefill保持原配置。resident FP8 FA2 prefill+XQA decode；XQA内部Q仍FP8，不误称BF16 Q。只生成候选FP8一臂，保留已有现代BF16和原FP8-XQA完整结果，不重复成功基线。不同run/GPU/缓存与实际调度仍限制单因果解释。

执行前：79源码CPU通过/1项CUDA模块明确skip（未称GPU通过），215唯一控制器CPU通过；47冻结资产和实际frozenpolicy通过；完整1719真实数据/protocol/count/order SHA的runner dry-run通过，计划环境V=.125并等于唯一FP8臂。冻结校准audit/proposal/参考audit，运行时在加载模型前重新校验，再实际新scale数值/Graph门槛。来源证明与数值门槛都不能代替模型质量。

预算最多单卡3600秒=1GPU-h、输出5GiB；执行前总9.8491395797/原12GPU-h，预留0，不重置。先fresh只读对账，120秒空闲/预约/UUID/共享CPU-RAM-disk/非root隔离容器/驱动可见性/watchdog通过；未知/断线只读恢复原run，不重发。

完成后逐SHA/真实Graph/完整1719 API输出、usage、finish reason、官方独立重评分；与两份保留参考按原ID逐题比较，报告bootstrap区间/任务级变化及限制，不把同一题再次运行算新用户。质量接受阈值仍未由用户给定。若无整体修复，停止该scale候选额外GPU服务矩阵；若有明确改善，再仅补必要真实生产服务验证。该候选不会自动继承旧约+32%的吞吐收益，也不替代后续代码/agent任务需求。

依据文件：`reports/operations/20261026T075846Z/longcontext_range_independent_audit_a1.json`、`longcontext_V_scale_coverage_candidate_a1.json`、`vscale_coverage_full_actual_dry_run_CPU_a1.json`；冻结package位于`results/prepared/packages/prod-quality-xqa-vcover-full1719-c128-r0-20261027a/`。
