# BF16也采用XQA后的完整质量参考

confirmed：新的BF16 FA3-prefill/XQA-BF16-decode参考完成GSM8K1319+LongBench八任务各50，108原始文件43709322bytes逐SHA镜像、实际模型权重/输入/tokenizer/prompt/usage/finish/Graph/封存和官方重评分通过。与保留的同FP8部署候选预测相比，GSM flexible+0.4549pp、LongBench macro-1.3726pp，LCC−5.38pp。更先进的BF16参考没有消除已见质量风险，不能认证生产默认或同质量净收益。

这是新BF16单臂和已经验收FP8预测的跨run比较，不是假装新跑了一组FP8配对。FP8原c80+a1639预测全部保留，原a CONFLICT永久保留，没有重复成功预测、没有按评测答案调参。

| 指标 | 新BF16-XQA参考 | 保留FP8-XQA候选 | 差值(pp) |
|---|---:|---:|---:|
| GSM8K strict | 69.7498% | 69.9773% | +0.2274 |
| GSM8K flexible | 89.2343% | 89.6892% | +0.4549 |
| narrativeqa | 32.4695% | 32.3234% | -0.1461 |
| qasper | 42.6740% | 40.8026% | -1.8714 |
| multifieldqa_en | 48.3477% | 46.5245% | -1.8231 |
| 2wikimqa | 58.7115% | 58.6429% | -0.0686 |
| hotpotqa | 64.6238% | 63.3989% | -1.2250 |
| musique | 37.6155% | 35.1089% | -2.5065 |
| lcc | 67.4200% | 62.0400% | -5.3800 |
| repobench-p | 49.1400% | 51.1800% | +2.0400 |
| LongBench macro | 50.1252% | 48.7526% | -1.3726 |

GSM flexible逐题41up/35down/1243tie；旧BF16参考比较为39/33/1247，aggregate flexible相同不代表每题输出相同。新BF16 strict69.7498%，与FP8只差+.2274pp，strict的格式影响不能代替flexible内容判断。LongBench代码相似度不是执行正确率；LCC新BF16仍67.42%，FP862.04%，解释性/格式偏移风险继续保留。

沿用配对任务内bootstrap5000/seed20261026，GSM flexible差95%区间[−.8340,+1.7437]pp；LongBench macro[−3.1020,+.2686]pp。1pp仍是未经用户接受的工程候选界，不能由跨零断言等价/非劣；也不抹去冻结集合实测下降。重采样单位是题目，非GPU运行；只有一次新BF16参考，且C8不认证全部生产batch。

## 对应配置与比较边界

新BF16 source11e7cddbfe33eb21d209c7f48f5e461ac56545eb，质量工具/strict attestation改变，全部推理引擎文件与已经实测服务source1c2be159字节一致。BF16 FA3 BF16-Q/BF16-KV prefill+SM90 XQA BF16 decode，PDL=true、有限新payload初始化和reserved page0 padding。新质量在H800/GPU3先通过真实GQA40/8、非连续Q/partial/replan/padded Graph和官方scorer，再真实14B capture/replay与预测；root正常COMPLETE、driver释放，实际占卡约0.296332673GPU-h，含JIT/权重审计/加载/Graph/评分/回收。

保留FP8 sourceede2b9e7，FA2 BF16-Q/FP8-KV prefill+SM90 XQA内部FP8-Q decode，K.125/V.0625。两边ceff binary、实际逐shard权重identity cf67b8e03cf159dd6017e36da84c28e143e59ce766261db2f6c97aabca7aae73、模型metadata/tokenizer/scorer/5shot、自然EOS/原任务生成上限、C8、prefix/Graph/chunked-prefill/128seq/10240token/200admission/.8一致。官方evaluator全AST在只移除新strict manifest profile识别项后完全相同，评分/提取/prompt逻辑未改。源码SHA和完整runtime SHA分别保留，不谎称全runtime相同。

原FP8按80+1639两批、各批独立缓存；新BF16按完整原始1719顺序，batch/prefix历史不同。greedy与相同模型不保证跨调度/数值路径逐token一致，因此比较可描述两个部署配置的质量表现，不能把每个变化单独归因KV dtype或XQA kernel。新BF16 LongBench macro50.1252%，旧FA3 BF16为50.0885%，差约+.0368pp，是同冻结记录的描述性敏感性观察，不证明质量完全相同。

## 生产决策

先进BF16下ShareGPT单次服务增益仍为RPS+32.9647%、actualoutputTPS+30.5593%，当前做一次同GPU反序独立复验；原两条ToolAgent压力曲线FP8臂仍在运行。质量代价应与这些收益同时呈现。当前适宜结论是有条件支持候选、限制默认启用与质量敏感业务，尚无用户接受的质量/SLO界，不将该配置认证为通用默认。质量修复应先根据真实clipping/underflow或适配算子证据选方案，不盲调global scalar或扩大受损配置矩阵。

证据：audit_summary.json、task_scores.csv；完整reference audit reports/operations/20261026T075846Z/BF16_XQA_full_quality_independent_audit_a1.json（SHA5a40a904937d538b10bcba74e48f3ac8512818f1fbcbf833886696e3b3928898），comparison reports/operations/20261026T075846Z/BF16_XQA_full_quality_existing_FP8_comparison_a1.json（SHA29e0d1efccd2c7ce5e94cf380457d4a4e2a60ac76a90f8e9778c51dc452e7114）。原始镜像results/mirrors/prod-quality-bf16xqa-full1719-r0-20261027a-c1/files。保留FP8完整1719 audit原SHA7caebece84319d1bc21a612c13eb9acd1bb2a5179357f0daccbc06a97940d39e不变。新原始记录重新核对原72shard全局索引，未改原数据/评分器/旧终态。

后续状态更新（2026-10-27）：ShareGPT反序两pair、ToolAgent两现代压力点均已独立验收；同完整1719/C128双臂质量也已验收，LongBench−1.7705pp/LCC−3.92pp、未获生产接受。见对应replicates/pressure/quality-c128报告。本C8参考原分数/输入/终态不变，跨批边界继续保留。
