# 同生产FA3候选：完整冻结1719条质量评估

**confirmed：每臂1719条原始预测完整、独立重评分通过；当前候选有质量损失，不宜直接认证为生产默认配置。** GSM8K flexible下降0.6823个百分点，LongBench八任务macro下降2.0075个百分点。此前ShareGPT约18.7%请求吞吐收益仍是该配置的性能测量，不能称同质量净收益，也不能因此否定所有KV量化方案。

1319道GSM8K及LongBench八任务各50条，由已验收pilot80与不重叠completion1639按原始索引合并。72份冻结源分片逐SHA恢复，原记录、prompt/tokenizer/API prediction、usage、finish reason、模型权重、源码、binary、评分器和Graph证据均验收；两个root COMPLETE/逐臂封存与driver释放匹配。没有重复生成已成功的80条。原始补测152文件64,886,492bytes逐SHA镜像；第一次240秒传输超时的残片保留，第二次只读下载成功，不涉及GPU重跑。

| 指标 | BF16 | FA3 FP8-Q/FP8-KV | 差值（百分点） |
|---|---:|---:|---:|
| GSM8K strict EM | 68.7642% | 69.3707% | +0.6065 |
| GSM8K flexible EM | 89.3101% | 88.6277% | −0.6823 |
| NarrativeQA | 32.5762% | 28.2598% | −4.3164 |
| Qasper | 42.6841% | 40.3713% | −2.3129 |
| MultiFieldQA-en | 48.3998% | 46.8157% | −1.5841 |
| 2WikiMQA | 58.7115% | 57.5262% | −1.1853 |
| HotpotQA | 64.7322% | 63.2667% | −1.4655 |
| MuSiQue | 37.6155% | 36.4596% | −1.1559 |
| LCC | 67.4200% | 62.5800% | −4.8400 |
| RepoBench-P | 49.6600% | 50.4600% | +0.8000 |
| LongBench 8任务macro | 50.2249% | 48.2174% | −2.0075 |

GSM flexible逐题33改善、42退化、1244持平；strict上升不能替代对答案内容的flexible判断。代码两任务两臂均50/50触及冻结生成上限，按原官方codesim协议评分，不能把length本身算模型失败或偷偷提高输出上限。

配对bootstrap5000次、seed20261026：GSM flexible差值95%区间[−1.9712,+0.5307]个百分点；LongBench按任务内重采样后macro差值[−3.7777,−0.2962]。该区间描述对相似样本的抽样不确定性；冻结集合的分数差已经实测。工程非劣界1pp仍是提案、未经用户接受，两项均不足以证明此界下非劣。不是完整LongBench或所有生产能力的认证。

按实际prompt长度分桶的LongBench差值并非单调：<4K约−2.89pp、4–8K约−2.52pp、8–16K约−1.86pp、16–24K约+1.00pp、24–32K约−5.18pp。各桶任务组成不同，不能由此断言上下文长度的因果效应。

## 实现与生产含义

同H800 GPU0/Qwen2.5-14B，BF16-Q/BF16-KV对FA3 FP8-Q/FP8-KV，Q=1/K=.125/V=.0625；prefix、Graph、chunked prefill、128/10240/200调度、显存比例0.8，质量并发8。pilot源码68fedda1，completion为6bbe8779，仅质量scope工具改变，实际引擎完整source digest相同；binary SHA仍277bd8…b21f3。模型权重逐shardidentity为cf67b8e03cf159dd6017e36da84c28e143e59ce766261db2f6c97aabca7aae73。

这不是纯KV dtype效应：新增FP8 Q和算子/数值路径共同构成配置。不能从质量损失直接归因于Q、K、V或校准中的某一项。下一步优先核查可实际部署的先进路径和校准，并以真实任务验证；不在评测答案上调参，不反复扩大受损配置的性能矩阵。

现装FlashInfer的SM90 XQA虽然API接受BF16 Q，实际mha_sm90.cu的F16QToF8Converter仍在内核中转成FP8，不能把API dtype当作“没有Q量化”的证明。它仍可能通过融合和decode优化改善服务，需要显式算子/质量验收。FA2 paged prefill接受BF16-Q/FP8-KV，可作为与先进decode组合的部署候选，而非把旧FA2限制为永久基线。

原计划BurstGPT及Mooncake conversation/toolagent/synthetic继续保留；来源、连续到达和prefix准备已有CPU证据，正式trace服务尚未测。先改善/确认质量配置，再给正常/高压/突发/容量边界的服务曲线，最终决定支持范围及维护成本。

完整1719质量共0.843364 GPU-hours，其中补测0.588695；本会话四个实际作业累计1.303045 GPU-hours（包含被停止的合成任务及失败后的回收），额度已按driver释放结算。当前无我方活跃GPU任务。

证据：audit.json、task_scores.csv；只读镜像results/mirrors/prod-quality-fa3-completion1639-r0-20261026b-c2/files与pilot80镜像；复算工具scripts/merge_quality_cohort.py。源文件、父批绑定audit、失败终态及历史结果均未覆盖。
