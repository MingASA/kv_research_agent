# 内容延迟与完整SSE完成：实际AIPerf口径核验

confirmed：实际安装源码和原四家族全部成功测量请求的raw/metadata/官方指标逐项吻合。request_latency截止最后非空parsed content；request_end_ns由最后raw SSE包映射epoch，可包含无内容finish/usage/DONE。后者不是TCP EOF；两者不等于同一个完成时间。当前匿名ignore-EOS长度回放有些请求最后内容早于length完成数十秒，具体token IDs未导出，不能将EOS或某个kernel/调度原因写成事实。

保留全部原官方raw/summary与冻结runtime；offered E2E按原时钟截止最后SSE，已经包含此尾段。额外报告(first-content→last-SSE)/(server_output_tokens−1)的完成时间代理，单token不定义。这不是硬件decode step、每个token stall或语义质量goodput。官方ITL只用content时段及服务端output count，不能当全部真实token的逐步耗时。不同成功子集与原FA3-BF16参考的边界不变。

| workload | arm | 已逐raw校验的成功测量请求 | 末SSE减末内容P95(ms) | 最大(ms) | 完成时间/token代理P50(ms) |
|---|---|---:|---:|---:|---:|
| toolagent | bf16 | 271 | 10932.494 | 41482.988 | 88.016 |
| toolagent | fp8_e4m3 | 528 | 8717.689 | 52623.508 | 86.947 |
| burstgpt | bf16 | 6221 | 2.059 | 6714.183 | 14.528 |
| burstgpt | fp8_e4m3 | 6221 | 2.114 | 6079.999 | 12.917 |
| conversation | bf16 | 219 | 28181.900 | 50961.311 | 89.914 |
| conversation | fp8_e4m3 | 373 | 27157.403 | 86138.856 | 89.184 |
| synthetic | bf16 | 342 | 25713.179 | 64643.636 | 121.039 |
| synthetic | fp8_e4m3 | 381 | 26190.041 | 122819.626 | 218.847 |

合计14556条。所有raw末包与metadata时钟逐整数相同，最后非空content复算request_latency一致，代理与官方ITL+tail/(OSL−1)逐条一致；未放宽统计比较、未修改旧结果。2个CPU边界fixture覆盖usage/DONE尾段保留和时钟漂移拒绝。各输入audit/raw/安装源码/分析脚本SHA均在对应JSON。

原toolagent官方ITL P50 85.054→84.656ms，新增代理88.016→86.947ms；synthetic为121.039→218.847ms，原先多完成但decode更慢的结论保留。conversation代理89.914→89.184ms，不能从官方内容ITL推断全部decode步骤改善。BurstGPT短API尾段很小，约2ms的P95，不改变无整体吞吐收益结论。

原匿名trace与真实ShareGPT自然EOS服务分别报告。新两条压力曲线完成后用同样脚本分析，不能热改运行时、删掉尾段或强行停止已合规任务。原始质量/服务SLO仍未由用户接受。

证据：toolagent_original_content_terminal_gap_CPU_v3.json, burstgpt_original_content_terminal_gap_CPU_v1.json, conversation_original_content_terminal_gap_CPU_v1.json, synthetic_original_content_terminal_gap_CPU_v1.json；源码来自results/mirrors/protocols-20261026T075846Z/files/support/.venv-aiperf/lib/python3.12/site-packages/aiperf，直接核对request_latency_metric、record_processor_service、record_models和openai_chat。可复算脚本scripts/analyze_stream_completion_gap.py。
