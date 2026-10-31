# burstgpt原速连续服务：独立验收

confirmed：两臂全部6385次成功，测量6221/6221，完成速率完全相同；FP8未显示整体吞吐收益。TTFT P95略慢，ITL P50较低，都是此单次短API负载的描述性观察，不外推所有短请求。 root正常COMPLETE、driver释放，独立原始审计通过。

同一H800 UUID顺序配对，source ede2b9e7、既有ceff binary；BF16 FA3 prefill/decode，对FP8 FA2 BF16-Q prefill+SM90 XQA内部FP8-Q decode，PDL=true，K.125/V.0625。同prefix/Graph/chunked prefill/128seq/10240tokens/200admission/.8，原逐请求OSL、来源到达/前缀、所有尝试与错误保留。匿名文本不等于原语义请求，不能替代质量。

全体6385次尝试/臂，其中164逻辑warm、6221测量；测量offered RPS=11.520370。每个成功响应按原payload/源OSL/usage/SSE/DONE逐项验收，AIPerf真正结束。该原peak10分钟窗口有4秒前导空白，按实际首请求归一化后offered596秒，原逻辑warm只有56秒、测量540秒；完整6385顺序/重复均保留。原数据两个模型聚合流量映射单Qwen14B实例，不还原语义，平均源输出约4.7tokens。ITL只对多于1个输出token定义，测量中3721条有效，其余单token不填0混算。

原时间戳按秒分组，1秒固定bvar采样与突发相位可能混叠；FP8 running中位0对BF16的9不能作为资源减少或容量优势证据。成功请求/实际时延和物理分配证据仍有效；不为了修复这项低价值采样而重跑成功GPU实验。

| 指标（测量arrival cohort） | FA3 BF16 | FP8/XQA |
|---|---:|---:|
| 尝试数 | 6221 | 6221 |
| 成功数 | 6221 | 6221 |
| 成功率(%) | 100.000 | 100.000 |
| 错误/拒绝数 | 0 | 0 |
| 含drain完成req/s | 11.520 | 11.520 |
| 实际完成输出tok/s | 53.415 | 53.415 |
| offered TTFT P50(ms) | 206.088 | 210.546 |
| offered TTFT P95(ms) | 329.569 | 338.490 |
| 逐请求平均ITL P50(ms) | 14.328 | 12.667 |
| offered E2E P95(ms) | 365.851 | 363.217 |
| offered窗口后drain(s) | 0 | 0 |

所有HTTP200都按实际内容判断成功；错误逐raw归类见audit_summary中的failure_analysis。此批所有失败均为并发上限纯文本notice，不是JSON SSE error；BurstGPT无此失败。成功cohort配对为{'both_success': 6221, 'BF16_only': 0, 'FP8_only': 0}；未完成请求不删除，成功条件延迟和全体成功率分开报告。

应用采样：bf16: running P50=9.0, waiting P50=0.0, used比例P50=0.0169377；fp8_e4m3: running P50=0.0, waiting P50=0.0, used比例P50=0.0。物理pages1477→2865，resident bytes/token/layer4096→2112、容量约1.94倍；有效used包含page0，free/prefix可回收，不能只用free=0或把prefix与used相加推断耗尽。1秒采样非原子，preempted是迭代gauge，NVML活动比例不是HBM GB/s。

FP8候选完整1719质量代价仍为LongBench macro−1.3358pp、LCC−5.38pp，未经接受；新BF16-XQA参考与正常/中等到达速率另立明确协议，不沿用本旧FA3参考的比例。只有一个paired run，不伪造run级CI。性能运行未重新hash全权重内容；质量、最佳部署基线与维护成本共同决定支持范围。

证据：audit_summary.json、metrics.csv；完整独立audit reports/operations/20261026T075846Z/burstgpt_f_independent_audit1_py312.json，SHA 3bf842b9578e1fb66877111d2809ff28ee8c40c9fcc8f2d83f9ce939d4b7a314；只读镜像results/mirrors/prod-trace-xqa-burstgpt-s1-r0-20261027f-c1/files，每文件SHA通过。Python3.12严格原始重算，无GPU重复；原冲突/失败不改成功。实际占用约0.522967767GPU-h，含加载/JIT/Graph/预热/回收，ceil微秒计费与ns审计略有舍入差。

后续计时口径补证：官方request_latency截止末非空content，offered E2E截止末raw SSE，并非TCP EOF。完整尾段/每服务端输出token的新增代理P50为14.528→12.917ms；原官方数据保持不变，代理不是硬件step计时。所有成功测量请求逐raw时钟验证通过；证据reports/operations/20261026T075846Z/burstgpt_original_content_terminal_gap_CPU_v1.json及STREAM_TIMING_FINDINGS.md。匿名ignore-EOS输出可能存在无可见内容的尾段，具体token IDs和原因未隔离，不替代自然语义服务或质量。
