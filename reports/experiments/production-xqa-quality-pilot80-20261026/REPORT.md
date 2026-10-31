# XQA生产候选：80条配对质量pilot

**confirmed：两臂各80条原始预测、usage/prompt/模型权重/完整源码/binary/Graph和官方重评分已独立验收；尚未认证生产质量。** GSM flexible BF16为57/64，XQA为58/64；LongBench仅八任务各最短/最长两条，macro43.4274→39.2495%，下降4.1779pp，主要来自RepoBench-P的.485→.12。NarrativeQA/HotpotQA/LCC等此小样本与BF16持平，Qasper略升。不能据此判断完整任务平均损失或去掉Q近似的因果效果。

部署为H800 GPU0、Qwen2.5-14B、source ede2b9e7及既有ceff二进制；BF16 FA3参考对FA2 BF16-Q/FP8-KV prefill加XQA decode（内部FP8 Q、PDL开启），K.125/V.0625、prefix/Graph/生产128/10240/200与.8显存比例、质量C8。两臂同一真实检查点identity cf67b8e03cf159dd6017e36da84c28e143e59ce766261db2f6c97aabca7aae73。新版算子不同于旧FA3候选，不能复用其质量或吞吐作为本候选认证。

真实单层GQA40/8算子、partial/replan/非连续Q与padded CUDA Graph均通过，decode relativeL2约2.47%。同Graph/PDL控制中unused K为NaN仍10240输出有限，unused V为NaN导致全部输出NaN；恢复最大有限FP8残值后参考重验通过。初始化新payload为零是明确部署要求；同指针rebind保留有效内容，无BF16 shadow。此前a失败保留，b测试夹具别名错误已在c真实复验，不称上游padded kernel故障。

run prod-quality-xqa-pilot80-r0-20261026c真实COMPLETE、driver释放，164原始文件3,485,272bytes逐SHA镜像。占用0.246866345GPU-h，含编译、加载、Graph、预测与回收。没有该XQA候选的新AIPerf性能结果。

下一步仅补原冻结1639条，与此80精确不重叠，绑定同候选父批result/audit与引擎hash，合并1719再判断质量与原生产workload测量。少量pilot不足以证明非劣；1pp工程候选界未经用户接受。不在评测答案上调整scale。

证据：audit.json；原始镜像results/mirrors/prod-quality-xqa-pilot80-r0-20261026c-c1/files；operator与terminal只读证据reports/operations/20261026T075846Z/xqa_status_c7.json。只使用我方GPU0，4–7他人任务未使用或干预。
