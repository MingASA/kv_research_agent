# XQA候选：真实ShareGPT C128端到端服务配对

confirmed：同一H800 GPU2、Qwen2.5-14B、真实ShareGPT上下文/自然EOS，两臂各512测量+16预热的原始payload/SSE/usage、source/binary、resident KV、实际Graph与封存均独立验收。相对当前FA3 BF16参考，请求吞吐+152.58%（2.526倍），输出吞吐+148.19%，ITL中位数-55.05%。root正常COMPLETE、driver释放，没有把冲突改成功。

这是两个实际部署配置的比较，尚未证明最优BF16或仅KV量化的收益。BF16参考为FA3 prefill/decode；FP8为FA2 BF16-Q/FP8-KV prefill加SM90 XQA decode（内部FP8 Q、PDL开启）。下一步必须补可部署的FA3 BF16 prefill+XQA BF16 decode参考，再判断完整支持KV量化的增量价值。不是为纯dtype论文对照，而是让生产BF16也得到合适的先进算子。

| 指标 | FA3 BF16参考 | FP8/XQA候选 | 相对变化 |
|---|---:|---:|---:|
| 请求吞吐(req/s) | 2.505 | 6.327 | +152.58% |
| 实际输出吞吐(tok/s) | 1352.491 | 3356.753 | +148.19% |
| 平均实际输出(tokens) | 539.945 | 530.553 | -1.74% |
| TTFT P50(ms) | 8348.988 | 127.239 | -98.48% |
| TTFT P95(ms) | 13996.465 | 4924.702 | -64.81% |
| TTFT P99(ms) | 15597.886 | 6163.090 | -60.49% |
| 逐请求平均ITL P50(ms) | 73.961 | 33.244 | -55.05% |
| 逐请求平均ITL P95(ms) | 80.394 | 42.972 | -46.55% |
| 逐请求平均ITL P99(ms) | 161.153 | 142.840 | -11.36% |
| E2E P50(ms) | 45002.067 | 17392.638 | -61.35% |
| E2E P95(ms) | 81772.532 | 34354.152 | -57.99% |
| E2E P99(ms) | 85738.785 | 38634.256 | -54.94% |

两臂均512/512测量请求成功，warmup各16另计，输入均值1120.586tokens、session和实际prompt计数摘要完全一致。实际输出276452→271643，平均长度下降1.74%，不能把整项吞吐差解释成相同输出长度；同时实际输出tokens/s提高148.2%，结果也不只是早停。stop/length为460/52→467/45。observation窗口204.402071→80.924322秒，未用benchmark_duration替代。closed-loop C128是持续压力协议，不冒称原生产到达时间线。

质量必须单独保留：同FP8部署候选的完整1319 GSM8K+8×50 LongBench，GSM flexible+0.4549pp、LongBench macro−1.3358pp，LCC−5.38pp；没有用户接受的质量/SLO损失界。代码质量原评分和解释性输出风险见production-xqa-quality-full1719-20261026/CODE_ERROR_ANALYSIS.md，不把相似度当执行正确率。本结果不是同质量净收益或生产默认认证。

同prefix/Graph/chunked prefill、128/10240/200调度、默认显存比例.8、独立新缓存。source ede2b9e7c61573825645a6b710a7534d0091bac2、既有ceff binary SHA277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3，未重编译。K=.125/V=.0625，驻留bytes/token/layer4096→2112、pages1477→2865、物理分配token容量189056→366720（约1.94倍，含预留page0），没有BF16 shadow。NVML采样峰值66827→66907MiB；总预算接近不是压缩失败。最大SLO服务容量仍待真实trace压力曲线，不将NVML activity当HBM GB/s。

整项实际占用0.278770429GPU-h，含算子JIT/加载/Graph/预热/测量/回收，启动私有缓存不代表标准生产cold-start；steady completed-request GPU时间约下降60.41%，未加入质量接受条件。只有一次配对，不能从512个相关排队请求伪造run-level重复或置信区间。当前尚无scheduler/KV占用和硬件HBM归因。

原计划继续：toolagent完整2088连续窗口已重新冻结并由唯一supervisor派发；BurstGPT6385、conversation1025、synthetic995保留，先验收首个真实trace的SSE/usage/原OSL，再按资源和信息价值并行覆盖。使用各自最佳可部署kernel、真实压力和质量，不能由这个点宣称所有场景均加速。

证据：audit.json、metrics.csv；92个源文件189205679bytes只读镜像并逐SHA验收，镜像results/mirrors/prod-sharegpt-xqa-c128-r2-20261027a-c1/files。root结果SHA d1edd69a77c2f282901600830c903c9ebabc6a3b1beec8c7ebe9aafe7af31bfc。audit c2只修正旧模板对COMPLETE/XQA的描述，所有sealed arms/metrics/差值/容量/计费与c1完全一致，旧c1保留。性能阶段未重新hash全部权重内容，只验证冻结模型路径/config/已有身份，不能冒充每次性能加载的权重全内容校验。此前两次ShareGPT prepare均独立NOT_LAUNCHED/0GPU-h，未计为两次统计重复。
