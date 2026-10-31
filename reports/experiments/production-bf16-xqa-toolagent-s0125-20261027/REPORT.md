# ToolAgent较低压力：吞吐相同，FP8延迟较慢

confirmed：同一H800 GPU4、先进BF16-XQA参考、同2088条原请求/prefix/逐请求OSL，时钟s=.125连续拉长8倍。每臂399逻辑warm和1689测量全部成功，offered .8796875 RPS，测量1920秒内均完成，无窗口后drain。FP8没有完成吞吐收益，TTFT P95从6.186秒升至7.388秒，完整E2E P99从24.276秒升至34.099秒；低压不能只由更大物理KV容量认定有收益。

| 指标 | BF16 FA3+XQA | FP8 FA2+XQA | FP8相对变化 |
|---|---:|---:|---:|
| 完成req/s | 0.880 | 0.880 | +0.00% |
| offered TTFT P50(ms) | 1611.483 | 1822.590 | +13.10% |
| offered TTFT P95(ms) | 6186.089 | 7387.651 | +19.42% |
| offered TTFT P99(ms) | 8925.125 | 10604.816 | +18.82% |
| 官方 content ITL P50(ms) | 61.300 | 70.848 | +15.57% |
| 末SSE完成/token代理 P50(ms) | 61.651 | 71.091 | +15.31% |
| 完整 offered E2E P50(ms) | 6649.728 | 7534.500 | +13.31% |
| 完整 offered E2E P95(ms) | 18808.843 | 19883.776 | +5.72% |
| 完整 offered E2E P99(ms) | 24275.703 | 34099.013 | +40.47% |

两臂测量均生成304758个服务端输出tokens，actual outputTPS均158.728125；同一完整成功cohort，不是接受子集变化。每臂总offered2400秒，前置480秒的399逻辑warm连续不reset；不能自动证明稳态。所有2088次HTTP200内容均有效，无拒绝/错误，AIPerf真实退出0。client发送延后P99约11.35/12.26ms，request-start仍不是服务端arrival。单token测量3条，ITL和完成/token代理不填0混算。

新增末SSE完成/token代理是(first-content→last raw SSE)/(server output count−1)，非硬件decode step或逐token stall。完整offered E2E包含无可见content尾段；官方content ITL保留不改。示例TTFT≤5秒且代理≤100ms的qualified完成速率.396875→.340625 RPS，10秒为.481771→.458333；阈值未经用户接受，匿名内容不含语义质量，不称正式生产goodput。没有达到所有请求低TTFT的默认SLO认证。

## 容量、调度与部署范围

1秒测量采样running中位4→5、最大均29；waiting中位均0、P95 9→10、最大22→23。有效used比例P50 .163957→.092179、最大.903117→.627444，prefix命中采样token比例.423401→.429026。此点等待总体较少；更大容量并未改善所测延迟。比例分母不同、prefix块可回收；不能把归一化used下降当allocator总显存减少，也不能从free接近0单独认定耗尽。

resident4096→2112 B/token/layer、physical pages1477→2865、物理tokens189056→366720（含reserved page0，约1.94倍）。两臂都使用相同.8预算，FP8没有全量BF16 shadow。source1c2be159/既有ceff binary/现装FlashInfer.6.18.post1；BF16 FA3 BF16-Q/BF16-KV prefill+SM90 XQA BF16 decode，对FP8 FA2 BF16-Q/FP8-KV prefill+SM90 XQA内部FP8-Q decode，PDL=true、K.125/V.0625、有限新payload初始化/reserved page0。prefix/Graph/chunked-prefill/128seq/10240token/200admission一致，客户端1024保持offered，并非GPU执行1024序列。

匿名合成内容只复现来源到达/长度/512token前缀关系，逐条ignore-EOS到原OSL，不还原工具文本或自然EOS质量。不同prefill、内部Q、writer与resident表示均属部署差异；无kernel分解/HBM计数，不能从本表隔离某一项原因。C8完整质量已知LongBench−1.3726pp/LCC−5.38pp，未经用户接受；后续同输入C128质量已独立验收，LongBench−1.7705pp；详见production-bf16-xqa-quality-c128-20261027，不以此trace认证质量。

与s=.25合并的两点报告仅为相同冻结runtime/source bundle/2088 payload和源cohort的观测，分别GPU4/GPU2且时间不同，每点内部同GPU两臂。一个paired run/点，不给run级CI或精确容量阈值；旧FA3-decode BF16原速点不混入此曲线。中压TTFT改善与生成期耗时上升并存，低压则无吞吐增量且尾延迟较慢，适用条件需与真实ShareGPT/质量共同判断。

## 完整性与资源

root cleanCOMPLETE/driver释放，89文件398197787bytes全部SHA、原始payload/SSE/usage/源OSL/clock/全部尝试、Graph/驻留及bvar独立重算通过。大raw采用已验收的scoped256MiB/XZ流式只读取证，未修改既有raw/runtime，未重跑GPU。性能未逐次重hash全部权重，完整质量另有actual identity。实际占卡约1.525413045 GPU-h，含加载/JIT/Graph/warm/回收，ceil微秒账本保留微小舍入边界。

证据：audit_summary.json、metrics.csv；canonical reports/operations/20261026T075846Z/toolagent_s0125_independent_audit_v1.json，SHA903c806a0f12573857bea558599c334ce867e175f034239d48da73126a1aa286；末SSE分析reports/operations/20261026T075846Z/toolagent_s0125_content_terminal_gap_v1.json，SHAd4b28c518e2ab7792a78ce5890afa5aaf74b1808cd49e82dfd007a2e6406ad72。镜像results/mirrors/prod-trace-bf16xqa-toolagent-s0125-r0-20261027a-c1/files；生产压力join为reports/operations/20261026T075846Z/modern_BF16_ToolAgent_pressure_comparison_v1.json，校验相同全源记录/payload SHA、runtime/bundle、模型/后端/服务配置及每次尝试/时钟/输出量，不重算或修改原官方分数。
