# conversation原速连续服务：独立验收

confirmed：原速下两臂均明显过载。FP8含drain完成速率提高150.12%，但延迟/成功率仍不能认证正常生产服务；不能称满足SLO的goodput。 root保持CONFLICT/无COMPLETE，driver释放；两臂完整measurement/SHA/原请求独立恢复验收。冲突发生于最后FP8响应之后8.901353秒，BF16更早完成；原终态不改写。触发同一先前本方PID/cgroup/stateR，旧事件未采kernel flags，具体原因仍unverified。只恢复测量，不重跑已完成请求。

同一H800 UUID顺序配对，source ede2b9e7、既有ceff binary；BF16 FA3 prefill/decode，对FP8 FA2 BF16-Q prefill+SM90 XQA内部FP8-Q decode，PDL=true，K.125/V.0625。同prefix/Graph/chunked prefill/128seq/10240tokens/200admission/.8，原逐请求OSL、来源到达/前缀、所有尝试与错误保留。匿名文本不等于原语义请求，不能替代质量。

全体1025次尝试/臂，其中194逻辑warm、831测量；测量offered RPS=3.462500。每个成功响应按原payload/源OSL/usage/SSE/DONE逐项验收，AIPerf真正结束。FP8测量成功373/831，BF16为219/831；完成子集不同，不能由成功条件下的时延分位断言同一批请求的纯加速。

| 指标（测量arrival cohort） | FA3 BF16 | FP8/XQA |
|---|---:|---:|
| 尝试数 | 831 | 831 |
| 成功数 | 219 | 373 |
| 成功率(%) | 26.354 | 44.886 |
| 错误/拒绝数 | 612 | 458 |
| 含drain完成req/s | 0.391 | 0.977 |
| 实际完成输出tok/s | 130.376 | 333.392 |
| offered TTFT P50(ms) | 242816.294 | 119908.772 |
| offered TTFT P95(ms) | 263218.363 | 128517.068 |
| 逐请求平均ITL P50(ms) | 84.973 | 80.713 |
| offered E2E P95(ms) | 306646.101 | 187004.433 |
| offered窗口后drain(s) | 320.454 | 141.635 |

所有HTTP200都按实际内容判断成功；错误逐raw归类见audit_summary中的failure_analysis。此批所有失败均为并发上限纯文本notice，不是JSON SSE error；BurstGPT无此失败。成功cohort配对为{'both_success': 192, 'BF16_only': 27, 'FP8_only': 181}；未完成请求不删除，成功条件延迟和全体成功率分开报告。

应用采样：bf16: running P50=21.0, waiting P50=150.0, used比例P50=0.873306；fp8_e4m3: running P50=43.0, waiting P50=150.0, used比例P50=0.889316。物理pages1477→2865，resident bytes/token/layer4096→2112、容量约1.94倍；有效used包含page0，free/prefix可回收，不能只用free=0或把prefix与used相加推断耗尽。1秒采样非原子，preempted是迭代gauge，NVML活动比例不是HBM GB/s。

FP8候选完整1719质量代价仍为LongBench macro−1.3358pp、LCC−5.38pp，未经接受；新BF16-XQA参考与正常/中等到达速率另立明确协议，不沿用本旧FA3参考的比例。只有一个paired run，不伪造run级CI。性能运行未重新hash全权重内容；质量、最佳部署基线与维护成本共同决定支持范围。

证据：audit_summary.json、metrics.csv；完整独立audit reports/operations/20261026T075846Z/conversation_f_independent_audit1_py312.json，SHA e73db3e5c0d5d5bacb14a5fce2a667af1a88fbfc3c1d7865aa714c69b4a3ab1a；只读镜像results/mirrors/prod-trace-xqa-conversation-s1-r0-20261027f-conflict-c1/files，每文件SHA通过。Python3.12严格原始重算，无GPU重复；原冲突/失败不改成功。实际占用约0.486532079GPU-h，含加载/JIT/Graph/预热/回收，ceil微秒计费与ns审计略有舍入差。

后续计时口径补证：官方request_latency截止末非空content，offered E2E截止末raw SSE，并非TCP EOF。完整尾段/每服务端输出token的新增代理P50为89.914→89.184ms；原官方数据保持不变，代理不是硬件step计时。所有成功测量请求逐raw时钟验证通过；证据reports/operations/20261026T075846Z/conversation_original_content_terminal_gap_CPU_v1.json及STREAM_TIMING_FINDINGS.md。匿名ignore-EOS输出可能存在无可见内容的尾段，具体token IDs和原因未隔离，不替代自然语义服务或质量。
