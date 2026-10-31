# 原速ToolAgent：容量压力下更多完成，仍未达到正常服务水平

confirmed：同一H800 GPU1、原始toolagent连续300秒到达/长度/prefix工作量，两臂各2088次尝试，399前置逻辑warm、1689测量。root正常COMPLETE、driver释放，89源文件278771564bytes逐SHA验收，所有payload/到达/源OSL/SSE/usage及错误独立复算。FP8测量期成功271→528（16.05→31.26%），含drain完成速率+119.81%。这是过载下的有条件部署收益；两臂均不能提供正常低延迟服务。

| 指标（测量arrival cohort） | FA3 BF16 | FA2-prefill/XQA-decode FP8 |
|---|---:|---:|
| 成功请求数 | 271 | 528 |
| 成功率(%) | 16.045 | 31.261 |
| 拒绝数 | 1418 | 1161 |
| 含drain完成速率(req/s) | 0.727 | 1.598 |
| 实际完成输出tokens | 43734 | 103371 |
| 实际完成输出tok/s | 117.295 | 312.784 |
| offered TTFT P50(ms) | 146204.393 | 72448.656 |
| offered TTFT P95(ms) | 171458.638 | 80968.168 |
| 逐请求平均ITL P50(ms) | 85.054 | 84.656 |
| offered E2E P95(ms) | 210636.496 | 130090.859 |
| client发送延后P99(ms) | 13.923 | 11.760 |
| offered窗口后drain(s) | 132.856 | 90.486 |

offered测量RPS两臂7.0375，240秒测量窗口，前置60秒连续流量不drain/reset。整个实际请求span约432.856→390.486秒；测量cohort完成速率按其240秒窗口加相应drain计算。warm cohort成功293→324/399另列，不冒称60秒已建立稳态。所有完成请求逐条达到原OSL，并非统一强制512；匿名合成文本只用于长度/前缀/到达系统判断，不还原真实语义，不替代ShareGPT自然EOS或质量。

HTTP状态两臂均2088个200，但并非全部成功。独立逐失败raw检查：全部1524→1236次失败（其中测量1418→1161）都是200下的并发上限纯文本提示，被解析成packet.name/value=null，AIPerf报InvalidInferenceResultError。不是JSON SSE error对象；不把200当成功、不删失败、不把它说成KV数值损坏。现服务把此类拒绝用200纯文本返回也是部署接口债务，报告保留实际行为。完整的错误/官方exit码见audit_summary及原始audit。

延迟分位是成功请求条件分布，两个完成子集不同：219个测量请求双臂成功、52仅BF16、309仅FP8；平均完成OSL也会变化。因此不把成功子集TTFT变化说成同批同长请求的纯加速。客户端发送延后P99约14/12ms，不能解释数十至上百秒TTFT；request-start仍不是服务端arrival，时钟/队列/计算边界分开。示例TTFT≤1/5/10秒且平均ITL≤100ms的测量qualified速率两臂均0；这些只是工程示例阈值，非用户SLO或同质量goodput。

## 容量与调度观察

同约34.62GiB resident KV预算，4096→2112B/token/layer，1477→2865物理pages（含预留page0），物理分配tokens189056→366720，约1.94倍，无BF16 shadow。测量加drain的1秒bvar采样无读取错误：有效used比例P50 .8828→.8890；running requests中位20→39、最大32→48；waiting中位177→155。采样free blocks中位都0，但prefix pool可回收，不能仅由free判断耗尽；有效占用、等待及实际驻留序列共同支持容量压力解释。

采样prefix命中token比约44.24→43.03%，来自有限采样counter差值，不是全请求精确事件比；prefix池与有效used不能简单相加。preempted迭代gauge采样均0不证明每次迭代绝无抢占。没有硬件HBM流量，NVML活动比例不是带宽。算子/数值路径不同，尚未隔离KV容量、attention算子与prefill的各自贡献。

source ede2b9e7、既有ceff binary SHA277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3；BF16 FA3 prefill/decode，FP8 FA2 BF16-Q/FP8-KV prefill+SM90 XQA decode内部FP8 Q，PDL=true，K.125/V.0625。prefix/Graph/chunked prefill/128seq/10240tokens/200admission/.8相同；client concurrency1024保留offered，而服务自行排队/拒绝。只有一个配对，无run级CI。

完整同FP8候选1719质量仍为GSM flexible+0.4549pp、LongBench macro−1.3358pp、LCC−5.38pp，未被用户接受；匿名trace不认证质量，旧C8质量也不等于所有服务batch。当前收益不称同质量净收益或生产默认。BF16也需使用经H800验收的先进XQA参考后再判断增量价值。后续保留同一时间/前缀形状，降低到达速率覆盖正常服务区间，不能仅靠这一严重过载点宣称完整生产支持。

## 审计与资源

实际占用按driver释放ceil计费0.418655367GPU-h，含JIT/模型/Graph/两臂预热与回收；审计ns求和与ceil微秒有不足1微秒差。root结果SHA7695e7e3148d2ec979698d4bf76d2a8d2448d152dd97875259aa7a99e97191df。性能运行未重新hash全部权重内容，模型路径/config与既有身份绑定不能冒充每次权重全内容核验。

原本地Python3.10审计c1严格比较失败，只31个mean字段有1e-16至1.5e-10舍入差。使用与远端一致的Python3.12重新执行原审计，全部对象严格相等通过；未放宽比较或改原结果，c1失败和diff保留。正式全量audit：reports/operations/20261026T075846Z/toolagent_trace_independent_audit_c2_py312.json，SHA4c7c44217dccec39425460d9c047336d48e833e3320ebe2f126b2cc4a9d417b3；精简汇总audit_summary.json、metrics.csv。原始镜像results/mirrors/prod-trace-xqa-toolagent-s1-r2-20261027c-c1/files。逐失败归类脚本scripts/analyze_trace_failures.py，旧传输/模型前NOT_LAUNCHED提案不计GPU重复。

后续计时口径补证：官方request_latency截止末非空content，offered E2E截止末raw SSE，并非TCP EOF。完整尾段/每服务端输出token的新增代理P50为88.016→86.947ms；原官方数据保持不变，代理不是硬件step计时。所有成功测量请求逐raw时钟验证通过；证据reports/operations/20261026T075846Z/toolagent_original_content_terminal_gap_CPU_v3.json及STREAM_TIMING_FINDINGS.md。匿名ignore-EOS输出可能存在无可见内容的尾段，具体token IDs和原因未隔离，不替代自然语义服务或质量。
