# proposal：原生 FlashInfer 固定 scale 的配对质量研究 v1

尚未启动；不沿用旧 Triton 四臂质量作为 native FP8 的质量认证。
依据：`science/RESEARCH_THESIS.md` H3/H4；本次已验收的 native resident/Graph 性能结果；历史质量汇总仅作问题线索。

## 问题与配置

Qwen2.5-14B-Instruct、同 checkpoint/tokenizer、同 FA2 paged attention，BF16 vs native joint FP8 E4M3。
固定 K=.125/V=.0625，quality_transform=false；不在评测集上按答案回调 scale。
Graph 状态两臂一致，保留实际 capture/replay、allocation 和 scale 证据。
质量必须允许自然 EOS，明确 temperature=0、top_p=1、相同 max_tokens/stop/chat template/截断规则。
不能复用 fixed-decode proposal 的 ignore_eos=true 来比较任务正确率。

冻结数据：GSM8K 1319、LongBench 8×50，共每臂 1719。
初次只读定位 frozen data、官方 scorer、source SHA、prompt/template/tokenizer/scorer hash、max input/output 配置。
NarrativeQA 按完整记录/ID 校验 50 条，允许 49 个不同 prompt；不重分片、不删除合法样本。

## CPU 先行

1. 检查当前质量 runner 的真实后端选择。旧 joint runner 可能固定 Triton / eager，不能直接把 arm 参数改成 FP8 就称 native。
2. 在本地源码的新 Git branch/worktree 实现明确 native 模式或新的 runner，保留旧实现与旧结果。
3. 用 CPU import/mock 验证：生成的 server argv/env 选择同 FA2、精度/scales 正确、Python source/binary 路径一致、manifest 版本固定。
4. 在生产依赖集合的无 GPU 路径导入官方 LongBench metrics 并实际调用必要评分函数；不以 dry-run/hash-only 替代依赖检验。
5. 建立逐题输出、finish reason、usage、长度耗尽、prompt/source ID 与两臂匹配的不可变归档。评分失败不能写 complete。

## 逐步执行（均以可信许可和执行器就绪为前提）

先用固定、明确记录的分层子集做协议 pilot，例如 GSM8K 64 条 + LongBench 各任务 2 条；这 80 条是 proposal，冻结选择规则后才能用。
pilot 用于找接口/严重退化，不给全 benchmark 结论。每 arm 结果保留，完整阶段可复用**协议完全一致且原始索引明确**的已成功题目，避免重复生成。
模型加载/长 prompt 各占多少时间目前没有足够 native 质量实测，先预留 1 GPU、≤2 GPU-hours 的有限 pilot envelope（并非授权）。
以真实 pilot 的 token/耗时和上界估算全量成本；在有效许可剩余额度内自主选择单 GPU 顺序或多个独立单 GPU shard。
多卡分片合并按原始索引统一评分，不称 TP/DP，也不以八卡平均分代替逐题合并。

## 分析与预先声明

- GSM8K strict/flexible EM、正/负 flip、共同错误、格式化失败、length exhaustion。
- LongBench 8 个任务及 macro、逐题差分，按任务、实际 input length、输出长度分组。
- 配对二元结局用 McNemar/差分 CI；LongBench 以题/任务结构报告配对重采样，8 个任务的外推能力有限。
- **proposal 的质量门槛**：native FP8 相对 BF16 的 GSM8K flexible EM 和 LongBench macro 非劣界各暂定 1 percentage point。
  这不是既定用户容忍度；正式执行前冻结门槛及统计方案。CI 跨越非劣界时写“不确定”，不能以 p>0.05 断言等价。
- strict 与 flexible 方向相反时先分离答案格式和算术正确性，不能把更常输出 `####` 当模型能力提升。
- saturation/outlier/scale 观测须在独立非计时诊断 run，明确 calibration 集与 evaluation 集边界；不在性能主数据混入观测开销。

## 决策

质量严重下降 → 停止“可用配置”的性能扩展，研究 scale、K/V 敏感性、per-head/block 等独立实现。
质量可接受但速度近零收益 → 优先容量与高压力机制，报告负结果，不强行扩大矩阵寻找显著值。
质量与性能均有有条件收益 → 再开展 arrivals/prefix 明确的真实 trace 与可解释自适应策略。

所有结果仍需 state/events/index/next-actions 同步；manifest、输入、源码、binary、评分器 hash、失败日志不可覆盖。
