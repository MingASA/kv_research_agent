# 先进BF16参考下的ShareGPT真实端到端收益

confirmed：同H800 GPU1、Qwen2.5-14B，BF16也采用XQA decode后，FP8请求吞吐提高32.96%、实际输出吞吐提高30.56%，ITL P50下降11.17%。两臂512测量+16预热，clean COMPLETE、driver释放、92文件189009356bytes逐SHA镜像、全部原始payload/SSE/usage/自然终止/Graph及驻留KV独立验收。它是当前主要生产基线；后续质量参考、C128完整配对及正常负载曲线现均已验收，见文末链接；不认证默认支持。

## 配置与公平性

source1c2be15976b7fbb1d8b79e469659b2c8e52cff00，既有ceff binary SHA277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3。BF16为FA3 BF16-Q/BF16-KV prefill+SM90 XQA BF16 decode；FP8为FA2 BF16-Q/FP8-KV prefill+SM90 XQA decode内部FP8 Q，PDL=true，K=.125/V=.0625。不同prefill、内部Q精度、writer和驻留表示共同构成部署差异，不声称只改KV dtype的因果效应，也不证明全生态的绝对最优。

prefix cache、真实Graph capture/replay、chunked prefill、128seq/10240token/200admission/.8相同；每臂单独新容器/缓存，同UUID先BF16后FP8，切臂重新120秒稳定空闲。每臂先通过真实数值/partial/replan/非连续Q/padded Graph及有限padding门槛，再加载模型。原a归属CONFLICT保留，无服务数据，不改成成功。

同冻结ShareGPT528个真实上下文请求、原顺序、greedy、自然EOS/cap1024、closed-loop C128；测量输入均值1120.586tokens，session与prompt-token摘要相同。这是持续压力协议，不是原生产到达时间线；16条预热不自动证明完整稳态。

## 实测

| 指标 | BF16 FA3+XQA | FP8 FA2+XQA | FP8相对变化 |
|---|---:|---:|---:|
| 请求吞吐(req/s) | 4.765 | 6.336 | +32.96% |
| 实际输出吞吐(tok/s) | 2567.819 | 3352.528 | +30.56% |
| 平均实际输出(tokens) | 538.861 | 529.113 | -1.81% |
| TTFT P50(ms) | 4448.843 | 128.835 | -97.10% |
| TTFT P95(ms) | 7081.155 | 4740.215 | -33.06% |
| TTFT P99(ms) | 7355.672 | 6190.769 | -15.84% |
| 逐请求平均ITL P50(ms) | 37.183 | 33.030 | -11.17% |
| 逐请求平均ITL P95(ms) | 46.533 | 42.012 | -9.72% |
| 逐请求平均ITL P99(ms) | 139.899 | 118.598 | -15.23% |
| E2E P50(ms) | 23398.491 | 17274.197 | -26.17% |
| E2E P95(ms) | 41979.361 | 34612.594 | -17.55% |
| E2E P99(ms) | 43435.910 | 39223.593 | -9.70% |

两臂测量成功均512/512；actual输出275897→270906，平均输出下降1.809%，stop/length465/47→463/49，不能假称完全相同输出工作量。observation107.444079→80.806482秒；actual输出吞吐提高30.56%亦表明差别不只是更早停止。示例TTFT≤5秒/逐请求平均ITL≤100ms的qualified完成速率2.866607→5.952493req/s；阈值未经用户接受、不含语义质量，不称正式SLO goodput。

resident4096→2112B/token/layer，pages1477→2865，物理tokens189056→366720，容量约1.93974倍（含预留page0），48层resident约37.170→37.177GB。总预算接近来自分配更多pages，不能否认压缩；不等于最大无拒绝或SLO服务容量。NVML采样峰值66727→66911MiB，非连续峰值、非KV字节、非硬件HBM流量。此聊天run没有scheduler/KV占用归因。

整项实际占卡约0.253461439GPU-h，含JIT、加载、Graph、预热、测量与回收；逐ns审计和ceil微秒账本有不足1微秒舍入差。steady完成请求的GPU时间按请求速率估算下降24.79%，未经质量接受，不代表标准生产cold-start或同质量成本。

## 对原结论的影响

旧FA3-decode BF16对FP8 XQA的+152.584%请求吞吐是旧部署基线的真实测量，保留不抹除。新的+32.965%表明BF16算子选择对增量收益判断影响很大。两次运行GPU/时间不同，不用跨run比例精确归因某个kernel；当前同GPU新配对给公平生产比较依据，仍只有一次配对，无run级CI，不从512个相关排队请求伪造独立重复。

已有同FP8候选1719质量相对旧FA3 BF16参考为GSM flexible+.4549pp、LongBench macro−1.3358pp、LCC−5.38pp。新BF16-XQA质量不能自动沿用旧参考，现单独补全原1719 BF16，不重复FP8预测；质量工具改变、推理引擎与本性能版本字节一致分别留证。C8质量不认证全部生产batch，代码相似度不等于代码执行成功。没有用户接受的质量损失界，不称同质量净收益。

当前原四家族均已独立审计：BurstGPT短API由offered限制，吞吐无增益；其他三家族原速严重过载。新的ToolAgent s=.125/.25保留全部请求/前缀而改变到达速率，且采用本BF16参考，用来补服务压力曲线；原FA3曲线不能直接冒称相同基线。后续由质量、成功率/延迟/每GPU有效服务与维护代价共同决定完整支持范围。

证据：audit_summary.json、metrics.csv；完整audit reports/operations/20261026T075846Z/bf16_xqa_reference_b_independent_audit1.json，SHAc3e7a09b7412e8169a34cb416b0369c80f00b46eb9a44461f6ab4e31f4551120；原始镜像results/mirrors/prod-sharegpt-bf16xqa-c128-r1-20261027b-c1/files。性能阶段未重新hash全部权重内容，只核验冻结模型路径/config/已有身份，完整质量另做实际权重identity。原终态、失败、bundle与测量不覆盖。

后续质量已补证：production-bf16-xqa-quality-reference-20261027/REPORT.md。新BF16完整1719独立通过，保留FP8相对其GSMflex+.4549pp、LongBench−1.3726pp、LCC−5.38pp；跨run/不同批次缓存历史边界保留，不认证同质量净收益。服务反序复验和正常/中间压力曲线已独立验收：production-bf16-xqa-sharegpt-replicates-20261027与production-bf16-xqa-toolagent-pressure-20261027。后续C128质量完整配对LongBench−1.7705pp，见production-bf16-xqa-quality-c128-20261027；本报告首个pair原指标不变。
