# 原项目生产 workload 对账

分类：confirmed 输入/源码审计，尚未运行 trace GPU 实验。机器可读证据为 `production_trace_audit.json`，只读镜像含12个文件及逐文件SHA。

完整 Mooncake conversation / toolagent / synthetic 分别为12,031 / 23,608 / 3,993请求。三个source SHA都与历史冻结manifest一致；264条旧子集确实来自原始记录，`origin_line_indices`实际是从1开始的行号。子集是随机稀疏抽样，不能用于声称完整生产到达或缓存复用。新回放采用完整来源或连续时间窗口，不沿用关闭固定调度的旧方式。

Mooncake时间单位为毫秒，hash对应512个原始token块；原始文本和token因隐私未提供。实际xLLM页128不能取代trace的512块。必须检验生成文本经过模型tokenizer与chat template后，共享前缀和长度仍符合声明。三类源的全保留、无限容量前缀复用token比例代理约37.3% / 57.0% / 65.0%，只是来源关系指标，不能当作引擎真实hit rate。[来源说明](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md)

conversation / toolagent / synthetic的输入P95分别约39.6K / 26.1K / 48.9K，部分原请求超过本模型32K上下文。窗口及明确上下文过滤必须报告；不能声称把被过滤的完整原始trace全部重放。toolagent输出中位数30，体现大量短输出调用，不可用强制统一512替代。

BurstGPT旧peak10m为6,385请求、平均10.64RPS、单秒最高29个，原CSV时间单位为秒；本文件没有Session ID、原始文本，两个模型的聚合流量不能不加解释地当单个14B实例需求。现装AIPerf loader把秒乘1000并合成指定长度prompt，不会还原语义或会话。须按单实例压力明确缩放时钟、保留突发形状，报告offered/actual rate、排队/拒绝/完成和drain time。[来源与字段](https://github.com/HPMLL/BurstGPT/blob/main/README.md)

自然EOS聊天与匿名长度trace回答不同问题。后者若按原逐请求响应长度仿真，必须标注长度回放及其合成内容，不能作为自然语义质量或完整生产请求回放；若使用自然EOS，也不能声称实际响应长度已复现原trace。两类证据分别支持真实聊天和调度/容量部署判断，质量仍用同生产后端的真实任务评估。

## BurstGPT完整来源已核验

原CSV 1,429,737条，原始字节SHA与旧冻结manifest一致。按原源首个时间戳5秒对齐600秒分桶，再依旧协议过滤非正输入/输出及输入+输出超过31744，5106号窗口为 `[3063605,3064205)`；6385条与旧peak文件逐行、顺序、重复记录均完全一致，且为eligible最高计数窗口。不能误用从0对齐的 `[3063600,3064200)`，会漏掉尾端101条。完整审计与0起源行号见 `burstgpt_source_independent_audit_v1.json`。这是来源验收，无GPU/模型性能证据。
