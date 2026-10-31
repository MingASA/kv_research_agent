# XQA生产候选：完整冻结1719条质量评估

confirmed：每臂1319道GSM8K与LongBench八任务各50条原始预测已按原始索引完整合并、独立官方重评分。GSM flexible比同批BF16高0.4549pp，LongBench macro低1.3358pp；LCC下降5.38pp，不能直接认证生产默认，也不能据此否定KV量化整体价值。下一步测量该明确质量代价下的真实服务收益，不称同质量净收益。

| 指标 | BF16 | XQA FP8 KV候选 | 差值（百分点） |
|---|---:|---:|---:|
| GSM8K strict EM | 68.5368% | 69.9773% | +1.4405 |
| GSM8K flexible EM | 89.2343% | 89.6892% | +0.4549 |
| narrativeqa | 32.4615% | 32.3234% | -0.1381 |
| qasper | 42.9965% | 40.8026% | -2.1939 |
| multifieldqa_en | 48.1708% | 46.5245% | -1.6463 |
| 2wikimqa | 58.7115% | 58.6429% | -0.0686 |
| hotpotqa | 64.7322% | 63.3989% | -1.3333 |
| musique | 37.6155% | 35.1089% | -2.5065 |
| lcc | 67.4200% | 62.0400% | -5.3800 |
| repobench-p | 48.6000% | 51.1800% | +2.5800 |
| LongBench 8-task macro | 50.0885% | 48.7526% | -1.3358 |

GSM flexible逐题39改善、33退化、1247持平。bootstrap5000次/seed20261026，GSM差值95%区间[-0.8340,+1.7437]pp；LongBench任务内配对重采样macro差值区间[-3.0819,+0.3166]pp。1pp只是未获用户接受的工程候选界，整体评估inconclusive；这不抹去冻结集合已测得的质量差。代码样本生成上限按原协议，BF16 LCC49/50、FP850/50为length终止，RepoBench均50/50；不偷偷改上限或把length直接判错。

## 部署与完整恢复链

H800/GPU0、Qwen2.5-14B-Instruct，source ede2b9e7c61573825645a6b710a7534d0091bac2，既有ceff binary SHA277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3。BF16参考为FA3 BF16-Q/BF16-KV；候选为FA2 BF16-Q/FP8-KV prefill加SM90 XQA decode（内部FP8 Q、PDL开启），K=.125/V=.0625、有限padding初始化、无BF16 shadow。prefix/Graph/chunked prefill开启，128/10240/200调度、显存比例.8、质量C8。算子/partial/replan/padded Graph及真实模型capture/replay均已独立验收。

三个run精确合并：c/pilot两臂80 COMPLETE；completion a的FP8 1639已恢复验收；missingbf16 b仅补缺失BF16 1639 COMPLETE。a在最后FP8预测之后4.60127秒触发归属保护，root CONFLICT永久保留，触发PID的具体退出状态未记录，原因仍unverified。FP8 measurement与59封存文件、原始预测在冲突前已完整完成；恢复仅补缺失BF16，不重跑FP8或pilot80，不改a为COMPLETE。

三批同模型权重identity cf67b8e03cf159dd6017e36da84c28e143e59ce766261db2f6c97aabca7aae73、完整runtime digest e4246fef58f91d13cc60b8e770ab71d1aa0edf89c711852574dab28f20f32e5e、数据/评分器/tokenizer/配置绑定均验收。只读镜像c164文件3,485,272bytes、a104文件41,902,528bytes、b106文件42,334,503bytes，每文件SHA核验；所有GPU已驱动确认释放。三批完整质量实际0.829428522GPU-hours，整个会话已结算2.222348501GPU-hours，失败与回收计入，不重置初始12GPU-h自限。

相较旧FA3-Q候选，当前LongBench相对同批参考的损失较小，是两个部署配置的描述性观察。BF16本次baseline与旧批也有变化；greedy跨批/调度不保证逐token一致，没有隔离Q/K/V或kernel因果贡献，不将它说成单独去掉Q量化的收益。质量C8代理亦不等于所有C128/生产任务质量认证。

## 后续生产决策

正式进入已冻结的自然ShareGPT C128及连续toolagent原速到达/长度/前缀工作量服务配对。前者保留真实文本/自然EOS，后者匿名合成内容但逐请求OSL、来源到达与prefix关系固定，分别报告成功率、错误、发送延后、尾延迟、排队/KV利用、drain和每GPU完成成本。继续原计划BurstGPT、conversation、synthetic有价值的覆盖与容量压力曲线；不为补矩阵盲跑。当前质量损失如在目标业务不可接受，则此配置不具可上线净收益，须独立校准/其他可部署配置。

证据：audit.json、task_scores.csv；原审计SHA 7caebece84319d1bc21a612c13eb9acd1bb2a5179357f0daccbc06a97940d39e；scripts/merge_quality_cohort.py三路恢复模式；原始镜像results/mirrors/prod-quality-xqa-{pilot80-r0-20261026c-c1,completion1639-r0-20261026a-conflict-c1,missingbf16-1639-r0-20261026b-c1}/files。原终态、父批绑定audit、预测和失败证据均保留。

补充代码模式审计已更正至v3：100条官方分数逐题复核不变，LCC4改善/8退化，Repo7改善/5退化。早期v1/v2将“未找到合格行时的默认空串”和“实际合格空行”合计，不能据其字段推断纯空白输出。逐原文检查发现解释性长句/格式偏移；没有整段whitespace输出，不将其解释为纯评分器问题或代码已正确。细节见CODE_ERROR_ANALYSIS.md与xqa_code_quality_patterns_v3.json，原评分器和主分数不变。
