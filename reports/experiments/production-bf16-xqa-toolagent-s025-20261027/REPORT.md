# ToolAgent中间压力：TTFT改善与decode耗时上升并存

confirmed：先进BF16-XQA参考下，原2088请求/prefix/逐请求OSL的连续时钟缩放s=.25，两臂都完整成功。测量1689/1689、warm399/399，offered1.759375RPS，含drain完成速率只提高0.789%；TTFT P95由28.805秒降为10.410秒，官方ITL及末SSE完成/token代理均明显变慢。不能从排队改善推断所有服务延迟改善，也不能仅靠100%成功认定可接受SLO。

| 指标 | BF16 FA3+XQA | FP8 FA2+XQA | 相对变化 |
|---|---:|---:|---:|
| 完成req/s | 1.714 | 1.728 | +0.79% |
| offeredTTFTP50ms | 3867.812 | 2267.015 | -41.39% |
| offeredTTFTP95ms | 28805.496 | 10409.535 | -63.86% |
| offeredTTFTP99ms | 33324.120 | 15413.518 | -53.75% |
| 官方contentITLP50ms | 52.415 | 83.140 | +58.62% |
| 末SSE完成/token代理P50ms | 54.439 | 85.979 | +57.93% |
| 完整offeredE2EP50ms | 10997.177 | 8661.601 | -21.24% |
| 完整offeredE2EP95ms | 47125.853 | 43698.688 | -7.27% |
| 完整offeredE2EP99ms | 62240.080 | 63796.489 | +2.50% |

两臂完整测量生成304758个服务端输出tokens，成功cohort完全相同，原input/OSL/prefix全部逐请求验证。offered每臂1200秒，前置240秒逻辑warm连续不reset，测量960秒；drain25.213639→17.499495秒，测量含drain985.213639→977.499495秒。actualoutTPS变化与完成RPS相同，当前流量大多受offered限制，不据此推断最大容量只有+.8%或所有workload无收益。所有2088次HTTP200内容均有效，无并发上限notice/错误。

末SSE完成/token代理按(first content→last raw SSE)/(server输出count−1)，比只看官方content ITL更完整，非硬件step计时/逐token尾部。3条单token测量不填0混算。示例TTFT≤5秒且此平均代理≤100ms的qualified速率.57957→.62097RPS；10秒为.79983→.87468RPS，阈值未经用户接受、匿名内容不认证语义质量，不称正式生产goodput。完整E2E P99从62.240→63.796秒略慢，尾部风险不被P95改善掩盖。

## 容量与调度

相同.8分配预算，pages1477→2865，物理tokens189056→366720，4096→2112B/token/layer、无BF16 shadow。1秒应用采样running中位16→20（max30→57）、waiting中位6→0（max84→42），有效used比例中位.796748→.398045，采样prefix token hit fraction约.4204→.4282。更多容量与等待减少吻合，但比例分母不同、prefix可回收，不能把归一化used下降说成allocator总显存减少；两臂resident均约37.17GB。没有HBM计数器或kernel时间分解，FA2 prefill、writer、不同批处理和内部Q数值均可能参与decode耗时上升，具体因果未隔离。

source1c2be159、同H800 GPU2、既有ceff binary；BF16 FA3 BF16-Q/BF16-KV prefill+SM90 XQA BF16 decode，对FP8 FA2 BF16-Q/FP8-KV prefill+SM90 XQA内部FP8-Q decode，PDL=true、K.125/V.0625、有限初始化/reserved page0。prefix/Graph/chunked-prefill/128seq/10240token/200admission/.8相同；客户端1024保留源offered调度，不等于服务器1024同时执行。仅改变连续源时钟4倍，匿名合成内容不还原真实工具文本或语义能力，ignore-EOS仅为逐原OSL工作量仿真。

完整C8质量相对新BF16参考为GSMflex+.4549pp、LongBench−1.3726pp、LCC−5.38pp，未获接受；新的C128质量包络单独验证。只一对run/压力点，不伪造run级CI。原速FA3-decode BF16不是本先进参考，不将旧严重过载百分比拼入同基线曲线。较低速s=.125已独立验收：无吞吐增益、尾延迟更慢，完整两点见production-bf16-xqa-toolagent-pressure-20261027；后续C128完整质量LongBench−1.7705pp见对应正式报告。

## 验收与取证

root cleanCOMPLETE、driver释放，89文件395316425bytes逐SHA镜像；所有payload/usage/原OSL/完整SSE/clock/成功与失败、AIPerf真结束、Graph/驻留及bvar独立重算通过。canonical audit reports/operations/20261026T075846Z/toolagent_s025_independent_audit_v1.json（SHA4c42db71a841f362b1ea71df8a5b026990c3933de8c9603eab5e8376cf58d96d），audit_summary.json/metrics.csv及toolagent_s025_content_terminal_gap_v1.json。性能阶段没有逐次重hash全部权重，完整质量另有actualidentity。

原raw单文件约150MB超过旧128MiB取证界。新explicit256MiB只限精确我方run路径且worker成功封存SHA匹配的raw export，源与本地1MiB分块读，单批总512MiB、XZ解压内存128MiB、展开tar520MiB和新目录/路径/软链接/SHA检查保持；169项CPU验收后实际H800全部89文件与独立audit通过；后续完整控制器验收为171项。无远端临时文件、无GPU重复、未修改旧raw或在跑runtime。实际占卡约0.874401231GPU-h，含完整启动/预热/回收，计费按driver释放ceil保留微小舍入边界。
