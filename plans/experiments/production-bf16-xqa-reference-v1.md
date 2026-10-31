# proposal：让BF16也使用H800上可部署的XQA decode

动机：首个干净ShareGPT配对中，FA3 BF16对FA2-prefill/XQA-decode FP8的req/s提高152.584%、outputtok/s提高148.190%，但尚未证明当前BF16是最佳部署基线。这是支持决策中的重要混杂：如果BF16同样受益于XQA，KV量化的增量价值会更小；若FP8仍有优势，再结合质量/容量/其他workload判断。目标是公平的生产选择，不是固定同kernel的论文控制。

独立本地source `/home/mingasa/projects/xllm_quant_kvcache-agent-bf16-xqa-20261027`，commit1c2be159（精确40位SHA在包内），基于已测ede2b9e7。只放行BF16的FA3 paged prefill + XQA decode配置，并让实际operator admission识别此配置。既有BF16 XQA执行、resident BF16 pages、unit K/V scales、SM90 wrapper不改；现有FP8 dispatcher/writer/scale/kernel不改。旧EDE工作树与正在运行的trace继续冻结不变，不热改、不升级共享FlashInfer，不重编译既有ceff binary。

明确配置：BF16 FA3 BF16-Q/BF16-KV prefill + SM90 XQA BF16 decode；FP8 FA2 BF16-Q/FP8-KV prefill + SM90 XQA decode内部FP8 Q。两者PDL=true、有限全新payload初始化、reserved page0 padding、同Graph/prefix/chunked-prefill/128seq/10240token/200admission/.8。XQA输入Q的contiguous与页表更新开销计入服务。允许不同prefill以采用各自可用实现；不称纯KV dtype因果。

服务协议为同一新run、同一空闲H800 UUID的BF16→FP8，冻结真实ShareGPT原上下文、自然EOS/cap1024、512测量+16预热、closed-loop C128，输入与前测一致。新配对是因BF16生产算子改变而必需，避免把旧GPU/内核/调度结果拼成同一run。不是重复旧run或已有预测，不把此前传输失败当重复样本。初轮一个配对，不从512个排队相关请求伪造run级CI。

CPU确认：75 passed/1 CUDA skipped；640组FP8 profile决策与EDE完全相同，638组其他BF16决策不变，仅2组明确FA3+XQA/BF16新配置获准；147个控制器测试通过，包括缺Graph/错误内部Q/错误planner拒绝和operator失败前不加载模型。CPU不等于H800通过。每臂必须在同一受监督私有容器先完成真实单层GQA40/8、非连续Q、partial/replan、padded Graph、有限尾页参考与NaN K/V诊断，并恢复有限态。relativeL2≤.05及rtol.08/atol.015不降低，失败不加载模型或铺其他基线矩阵。

每个成功响应逐payload/SSE/DONE/usage/finish reason审计，实际Graph capture/replay、resident bytes/pages、source/binary/input SHA、AIPerf真正结束和driver释放均验收。原始错误/失败保留。正式报告比较req/s、实际tok/s、输出长度、TTFT/ITL/E2E尾部、每GPU完成成本和约1.94倍物理KV容量；匿名trace的scheduler/KV容量证据另列，不能将NVML活动比例当HBM带宽。

质量边界：旧FP8完整1719的LongBench−1.3358pp/LCC−5.38pp继续明示；新source只改变BF16选择门槛，FP8核心文件未变有SHA证据，但完整runtime provenance不同。新的BF16数值/模型质量不能靠旧FA3参考自动认证；先operator admission，再服务，随后依据实际增量收益选有必要的质量补证/独立校准。没有用户接受的质量界，不称同质量净收益或默认支持。

资源为既有授权内1卡、整项3600秒/1 GPU-h、最多3GiB新输出；这是Agent预算预留，含JIT/加载/Graph/预热/失败/回收。原三家族与toolagent各2 GPU-h预留加已结算2.501118931，再加本项1，最坏11.501118931≤初始12，不重置或扩大。实际GPU4只是候选；每次仍先精确UUID/归属/预约/120秒稳定空闲/容器可见性/CPU/RAM/磁盘/独立watchdog检查。未知SSH状态只读恢复，不重复派发，foreign始终拒绝。

## 单次反序稳健性复验提案（2026-10-27）

主b已clean COMPLETE/完整raw验收，RPS+32.9647%、outputTPS+30.5593%；它只有一次约80/107秒observation的配对，512个相关请求不能代替独立run。为支持生产投入判断，仅安排一次同GPU1、反序FP8→BF16的r2c复验，不铺并发矩阵。保留同source1c2/ceff binary/528真实请求/所有frozenruntime与bundle资产/SHA/配置/scientific key，仅新run、replicate2与执行臂序；不是新的独立用户样本，也不重复旧失败run。质量损失与新参考待补继续披露。

本项≤3600秒/1GPU-h、3GiB输出，含两次稳定空闲/JIT/加载/Graph/预热/回收；当前settled4.678642933+现有最坏6+本项1=11.678642933≤初始12，不重置或扩大。GPU1仍须fresh身份/归属/无其他占用或已知预约、120秒空闲、容器隔离/CPU-RAM-磁盘与watchdog。CPU实际冻结policy及全部原资产SHA通过不是GPU通过。只有真实两臂raw完成/driver释放/独立重算后才计为第二个配对；两run仍不提供强总体CI，不将arm序/跨时间差倒推纯dtype或kernel因果。
