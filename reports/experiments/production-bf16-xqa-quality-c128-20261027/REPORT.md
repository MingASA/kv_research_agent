# 生产客户端C128：完整1719条配对质量

confirmed：同一H800 GPU3、同source11e7cddb、同完整1719源记录/顺序和实际服务客户端C128包络，两臂原始预测、官方重评分及模型/输入/prompt/usage/finish/Graph/封存均独立验收。root正常COMPLETE、driver释放，168文件67729166bytes逐SHA镜像。GSM8K flexible+.5307pp，LongBench八任务macro−1.7705pp；质量损失不能用ShareGPT吞吐增益覆盖，不建议当前配置作通用默认。

| 指标 | BF16 FA3+XQA | FP8 FA2+XQA | 差值(pp) |
|---|---:|---:|---:|
| GSM8K strict | 68.5368% | 71.3419% | +2.8052 |
| GSM8K flexible | 89.3101% | 89.8408% | +0.5307 |
| narrativeqa | 32.5448% | 34.5297% | +1.9849 |
| qasper | 42.9912% | 41.7424% | -1.2488 |
| multifieldqa_en | 48.6327% | 44.6327% | -4.0000 |
| 2wikimqa | 58.7115% | 58.1732% | -0.5383 |
| hotpotqa | 64.7322% | 63.3989% | -1.3333 |
| musique | 37.6155% | 33.1269% | -4.4886 |
| lcc | 67.1400% | 63.2200% | -3.9200 |
| repobench-p | 49.1400% | 48.5200% | -0.6200 |
| LongBench 8-task macro | 50.1885% | 48.4180% | -1.7705 |

GSM flexible41改善/34退化/1244持平；strict+2.8052pp明显大于flexible+.5307pp，不能把格式变化说成同等数学能力提升。NarrativeQA上升约1.98pp；MultiFieldQA−4.0000pp、MuSiQue−4.4886pp、LCC−3.9200pp，是与生产长文/多跳问答/代码能力有关的风险。代码指标为官方字符串相似度，不能当代码执行或agent成功率。

配对bootstrap5000次、seed20261026，GSM flexible差95%区间[-0.7582,+1.8196]pp；LongBench按任务内重采样macro差95%区间[-3.4893,-0.1667]pp。后者在0以下，不抹去抽样/每任务50条的外推限制。1pp仍是未经用户接受的工程提案；既不能证明此界下非劣，也不以p值或区间替代业务容忍度。题目重采样不是GPU重复运行区间。

## 与C8的敏感性比较

| 包络/参考 | BF16 LongBench macro | FP8 LongBench macro | FP8−BF16 |
|---|---:|---:|---:|
| C8新BF16参考与保留FP8 | 50.1252% | 48.7526% | −1.3726pp |
| 本次C128同完整cohort配对 | 50.1885% | 48.4180% | −1.7705pp |

C128相对C8新BF16macro+.06325pp、FP8−.33468pp，差值扩大约.39793pp。GSM flexible逐题C128对C8新BF16为18改善/17退化，FP8为34/32；总分相近不代表输出完全相同。RepoBench由C8相对新参考+2.04pp转为本次−.62pp，不能只选某批有利任务结论。完整逐题/任务对比在analysis JSON中，不扩大独立题目数。

两批使用同原问题、模型权重、tokenizer/5shot/评分/任务生成上限。C8新BF16是完整1719单臂，保留FP8按80+1639两批独立缓存；本次C128双臂都是完整原顺序，GPU/时间/并发/缓存历史不同。已核验新BF16参考和本次source/runtime digests一致、服务配置只改变客户端concurrency8→128，旧FP8来源单列。因此只能描述部署包络敏感性，不能把全部变化归因batch大小、KV dtype、某个kernel或调度中单一因素。greedy不保证跨数值/调度逐token一致。

## 生产配置及验收范围

Qwen2.5-14B-Instruct，实际逐shard权重identitycf67b8e03cf159dd6017e36da84c28e143e59ce766261db2f6c97aabca7aae73；ceff binary SHA277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3。source11e7cddbfe33eb21d209c7f48f5e461ac56545eb仅更新质量工具/attestation，全部推理引擎文件与服务source1c2字节相同。Torch2.10.0+cu130/Triton3.6.0/Transformers5.3.0；真实backend FlashInfer.6.18.post1。BF16 FA3 BF16-Q/BF16-KV prefill+SM90 XQA BF16 decode，对FP8 FA2 BF16-Q/FP8-KV prefill+SM90 XQA内部FP8-Q decode，PDL=true、K.125/V.0625、有限新payload初始化/reserved page0。

同prefix/Graph/chunked-prefill/128seq/10240token/200admission/.8、自然EOS/greedy/原5shot和官方scorer。客户端C128不是每个实际GPU batch固定128。每臂先在同隔离容器CUDA隐藏实际调用scorer，再真实数值/partial/replan/非连续Q/padded Graph门槛，通过后才加载14B并真实capture/replay；没有静默eager或BF16 shadow。两臂同源码/模型/输入和顺序，完整1719，原C8成功预测没有冒充本次新预测。

两代码任务两臂均50/50触及冻结length上限，仍按原官方协议评分，不能把length直接判错或偷改上限。GSM length为3/1，其他逐任务终止数保留canonical audit。QA/代码覆盖只是代理，不认证全部生产多轮工具、RAG或代码执行任务；TP1/H800单模型不能外推多卡TP/DP、其他模型或所有KV格式。

实际占卡0.470636590GPU-h，含两臂JIT/权重审计/加载/Graph/预测/回收，driver释放ceil微秒计费与ns审计有微小舍入边界。所有已完成服务、质量与失败累计约8.098758379GPU-h，初始12自限未重置；当前无我方活跃/UNKNOWN，不触碰他人GPU。

## 生产含义

先进BF16下ShareGPT两次反序RPS提升31.5–33%、actual输出29.7–30.6%是真实服务收益；ToolAgent低压无吞吐收益/尾延迟变差，中压TTFT改善但生成期变慢。它们与本次质量代价共同支持限定模型/后端/业务的可选FP8工程化，尚不支持通用默认启用或同质量净收益。后续按真实生产任务与组件时间证据选修复，不按评测答案调scale，也不铺受损配置矩阵。

证据：canonical reports/operations/20261026T075846Z/production_C128_quality_independent_audit_b1.json，SHA225142374bcffb2a198c4fcc2fc0ec8190a16297598e66e1c7fc02e2cbc5080a；reports/operations/20261026T075846Z/production_C128_quality_analysis_b1.json，SHA222836530894dd80791d2e70a1ceaffe5af5ce3395dc531724e1df549325f1c5；audit_summary.json/task_scores.csv。完整镜像results/mirrors/prod-quality-bf16xqa-full1719-c128-r0-20261027b-c1/files。可复算scripts/audit_quality_run.py及scripts/analyze_quality_batch.py；171项控制器CPU回归和独立四个质量分析变异拒绝通过，实际原始重评分与图表生成均退出0。图表reports/synthesis/production-decision-20261027/figures-v1包含SVG/PNG/数据CSV及全部输入/输出SHA。
