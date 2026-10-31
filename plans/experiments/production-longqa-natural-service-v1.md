# 真实长文QA自然停止服务：有限饱和请求集

状态：CPU冻结候选，未启动GPU。先完成当前conversation现代参考配对及独立验收，再按预算和实机状态决定派发。项目汇报仍截止第46条，不自动更新。

生产问题：输入密集的文档问答中，FP8的额外容量是否足以抵偿FA2预填充/量化写入代价？此前ShareGPT偏短；匿名trace不能代替真实文档与自然停止。这里保留Qasper/MultiFieldQA-en/HotpotQA/MuSiQue原问题、文档、提示模板及官方32/64/128生成上限，不补字、不强制输出、不把它称为真实业务到达或带检索器的RAG系统。

唯一候选 `prod-longqa-bf16xqa-natural128-c128-r0-20261027a`：复用已经独立验明来源的128条真实QA，每类32条；4103–20227输入token，总1,458,582，84条≥8K、34条≥16K。canonical SHA `1a36f83244b09b3edd85895586ec324624640ade004eb3ece672e94ee5dcfe5a`，与原BF16范围观察输入逐字节相同；原400质量题ID/记录/精确上下文不重叠。该请求集已经用于BF16激活观察，不能称为对本轮研究完全未见的新盲测，也不据此改变scale。

来源为官方LongBench revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`，档案SHA `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`；公开源包含答案，选择不依赖答案/既有预测，远端请求不携带答案。只作内部评估，保留各任务来源约束。

使用既有source `11e7cddbfe33eb21d209c7f48f5e461ac56545eb`、同一已核验ceff binary/14B权重/FlashInfer0.6.18.post1。BF16 FA3 prefill+XQA BF16 decode，对residentFP8 FA2 BF16-Q prefill+XQA内部FP8-Q decode，K.125/V.0625；两臂同UUID顺序配对。启用Graph/prefix/chunked prefill、128seq/10240token/200admission/.8。没有激活计数器或profiler；保留一秒私有bvar采样及NVML活动采样，均不解释成HBM流量。

AIPerf0.13原始inputs_json、C128、128正式请求、自然EOS，省略不合法的显式warmup0。服务启动后仍有真实首源请求的独立admission probe，前缀可能已缓存；不是完整冷启动或建立了稳态的结果。这个有限饱和请求集比较真实完成吞吐、TTFT/官方ITL/末SSE时间与队列/驻留现象，不把一次cohort总量除时长称为持续SLO容量。正式请求上限为200以内，不刻意制造OOM；出现失败保留全部raw并先分析，不删除请求凑成功。

复用原FP8部署方案已有1719题的质量风险，不因本次服务成功称质量通过，也不把V=.125候选的结果混入本方案。只报告既有文档QA请求上的性能；没有新用户总体、代码或多轮工具能力声明。

门槛：输入/manifest/tokenizer/算子源码/binary SHA、实际完整AIPerf CLI、冻结模块依赖与policy均通过CPU；两臂各重新至少120秒空闲/预约/队列/UUID/共享资源/容器/驱动/watchdog准入，真实算子与Graph先于14B加载。单卡整个作业最多1800秒=.5GPU-h、输出5GiB，沿用第一12GPU-h账本。SSH中断只读恢复同run，不重复启动。完成后核验原终态/释放、全部原始请求/usage/自然停止/cap/Graph/采样和SHA，再决定下一步。

CPU证据：`reports/operations/20261026T075846Z/longqa_service_frozen_and_full_CLI_CPU_a1.json`；224控制器CPU tests通过。准备器 `scripts/prepare_longqa_service.py`，冻结30资产，不改推理实现。尚无此候选的实际GPU、AIPerf服务或收益结论。
