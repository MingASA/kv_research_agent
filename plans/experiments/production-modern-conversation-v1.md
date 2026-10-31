# 下一生产服务候选：conversation的现代BF16参考

状态：CPU proposal，尚未启动GPU；须先完成当前V覆盖1719结果的独立审计和配置决定。项目汇报仍截止第46条，不自动加入本方案或后续结果。

当前差距：ShareGPT和ToolAgent两个压力点已使用BF16 FA3+XQA参考；原conversation/synthetic/BurstGPT的BF16还是FA3 decode。不能把其比值当成对最佳已验收BF16部署的收益。优先补conversation的容量/前缀复用场景；BurstGPT原低压双方6385请求全成功且完成速率受offered11.52RPS限制，暂不为填矩阵重跑。synthetic按剩余决策价值再选。

唯一CPU候选：`prod-trace-bf16xqa-conversation-s1-r0-20261027a`，source `11e7cddbfe33eb21d209c7f48f5e461ac56545eb`，即已完整质量/服务验收的现代参考源。BF16 FA3 prefill+XQA BF16 decode，对resident FP8 FA2 BF16-Q prefill+XQA内部FP8-Q decode，K=.125/V=.0625。与V覆盖候选分列；若当前质量结果改变部署配方，须先重新选择相应协议，不直接派发这份旧scale提案。

沿用原1025个conversation请求、窗口/逻辑warm与measurement标记、原始相对到达时间、逐请求OSL、prefix和顺序。payload SHA `0d06c694a17804ce99b99b0b28b5d99e47cf904ed99de226c7e9624bbca93afd`，逐字节等于旧完成/恢复的运行，arrival scaling=1。它是匿名时序/长度/前缀仿真，非原始文本/自然EOS语义质量评估；ignore-EOS只是复现来源的逐请求已记录长度，不能以此替代自然ShareGPT服务或模型质量。

本次改变了BF16 decode参考，使用同GPU顺序配对重新运行FP8对臂，是控制GPU时间/缓存历史所需的配对，不增加独立用户。旧run `prod-trace-xqa-conversation-s1-r0-20261027f` 的原CONFLICT终态与独立恢复证据保持不变，不以新运行改写它。

保留prefix/Graph/chunked prefill、128seq/10240token/200admission/.8，AIPerf固定schedule/连续warm cache/逐请求payload。所有拒绝、失败、排队、drain、实际发送时钟、usage和内容/末SSE时钟均保留。HTTP200并发上限纯文本notice计失败。按successful goodput、全体成功率与尾延迟解释；输出更长、NVML活动或采样gauge不冒充加速/HBM带宽。

32资产/完整模块依赖闭包/实际frozenpolicy/原input解码和Git bundle HEAD已CPU验证。`prepare_trace_package.py`改为复制当前完整controller模块集合，避免当前policy及production新增必需模块遗漏；默认7200秒保持，显式候选收窄到3600秒。未改推理代码、全局软件或既有结果。

预算最多单卡3600秒=1GPU-h，两臂在同一空闲UUID顺序运行，输出10GiB上限；第一12GPU-h窗口不重置。每臂均重新120秒稳定空闲、无他人占用/已知预约/排队，UUID、共享CPU/RAM/disk、私有非root容器、驱动可见性与watchdog准入。SSH断线只读对账同run，不发副本。部署资格待当前质量结论；没有本候选测量/接受/SLO容量成功声明。

CPU依据：`reports/operations/20261026T075846Z/modern_conversation_frozen_CPU_a1.json`；package `results/prepared/packages/prod-trace-bf16xqa-conversation-s1-r0-20261027a/package.json`。执行前重新fresh S0和账本检查；完成后独立raw/Graph/usage/到达审计，不以workerseal代替服务结论。


2026-10-27执行续接：a的原pre-model CONFLICT已独立审计，57文件/40,435,677B SHA一致，模型/AIPerf/arm测量均未开始，driver已释放。未知PID归属未证实，原终态保留。新run `prod-trace-bf16xqa-conversation-s1-r1-20261027b` 使用独立准入GPU1，replicate1仅编排重试，不作新统计重复/独立用户。相同scientific experiment_key、1025payload、source11e7、同算子/scale/服务；冻结prior审计作第33资产。仅增加未知PID诊断字段，检查强度不变，fresh120秒/预约/共享资源/容器可见性/watchdog全走。221 CPU tests、actualfrozenpolicy和完整真实AIPerf CLI已通过；S0静态范围通过，实际模型/两臂Graph/AIPerf/服务收益仍待。预算最多1GPU-h，累计10.10149413/第一12GPU-h，报告封存46不更新。


2026-10-27T16:14Z后续CPU选点：原rate b的BF16 worker已完成但配对raw尚待。只准备一个较低到达候选 `prod-trace-bf16xqa-conversation-s025-r0-20261027c`，offered测量.865625RPS，完整原1025记录/前缀/逐请求OSL/逻辑warm不变，只有timestamp除.25；不是原始真实业务到达，也不增加独立用户。显式复用b精确runtime所有SHA/sourcebundle，完整依赖闭包、现行ownership/safety代码相同；当前与actualfrozenpolicy、严格现有跨压力比较器均通过，227CPU tests。旧retry_evidence不继承给新科学压力点。仅CPU提案，无GPU；先完成当前配对raw与真实长文QA，再按结论/预算/整个3600秒运行边界决定。源窗口300秒变为1200秒，含两臂共2400秒到达，启动/切臂/JIT/收尾也在边界内；不足以可靠完成则不派发。汇报不更新。
