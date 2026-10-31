# 先进BF16参考的同GPU反序复验

confirmed：两次同H800 GPU1、相同source1c2/ceff binary/完整冻结runtime与bundle/ShareGPT528请求及配置的真实配对均独立raw验收、clean COMPLETE和driver释放。执行顺序相反。请求吞吐增益范围31.507%～32.965%，actual输出吞吐29.653%～30.559%，ITL中位数约下降10%～11%。这给当前生产配置的收益提供反序运行证据，仍不是总体置信区间或同质量认证。

| 执行顺序 | BF16→FP8 req/s | req/s变化 | BF16→FP8 actualouttok/s | tok/s变化 | ITL P50变化 |
|---|---:|---:|---:|---:|---:|
| bf16→fp8_e4m3 | 4.7653→6.3361 | +32.965% | 2567.82→3352.53 | +30.559% | -11.168% |
| fp8_e4m3→bf16 | 4.7868→6.2950 | +31.507% | 2570.40→3332.60 | +29.653% | -10.164% |

两run observation按时长汇总的req/s增益32.233%、actualouttok/s增益30.105%；这只汇总同GPU完成请求/actual tokens与observation时间，排除JIT/加载/cold-start，不能当全生命周期生产成本。两项整项实际占卡合计0.506793346GPU-h，JIT/加载/Graph/预热/回收均保留。每run自然EOS输出长度不同，原始perrun均值/终止原因不抹除；不是固定同输出文本或同语义质量的实验。

BF16为FA3 BF16-Q/BF16-KV prefill+SM90 XQA BF16 decode；FP8为FA2 BF16-Q/FP8-KV prefill+SM90 XQA内部FP8-Q decode，PDL=true，K.125/V.0625。同prefix/Graph/chunked-prefill/128seq/10240tokens/200admission/.8，closed-loop客户端C128，每run512测量+16预热，独立缓存、同UUID，切臂120秒稳定空闲。bundle/input/runtime全部asset SHA和scientific key相同，run_id与replicate不同，臂序相反，四臂实际source session/prompt count摘要完全匹配。

只有两个独立paired run。复用同512测量+16预热来源，不称1024个独立用户或把1024条排队相关请求bootstrap成run级CI。观测范围有助判断稳健性，不证明无限未来生产负载的收益，也不把不同prefill/internalQ/writer的部署差异归因纯KV dtype。旧FA3-decode BF16参考的+152.584%保留为另一个部署基线，不混入本复验统计。

当前新BF16-XQA完整C8质量参考对保留FP8为GSMflex+.4549pp、LongBench−1.3726pp、LCC−5.38pp，跨run/分批缓存历史明示，尚无用户接受的损失界。为贴近实际服务客户端C128，同完整1719、同顺序双臂C128质量验证现已独立验收，GSMflex+.5307pp、LongBench−1.7705pp、LCC−3.92pp；详见production-bf16-xqa-quality-c128-20261027。它是改变batch包络的质量补证，不扩大独立题目数。完整质量与其他工作负载/压力决定支持范围，不由这里的吞吐决定默认启用。

证据：audit_summary.json、metrics.csv；canonical reports/operations/20261026T075846Z/modern_BF16_ShareGPT_ABBA_comparison_v1.json，SHA8eba54cc42093a88358513817a9af2aef9a1924ba1239849ae45f18c401eca51。两原始镜像prod-sharegpt-bf16xqa-c128-r1-20261027b-c1与r2-20261027c-c1分别92文件189009356/188896504bytes，每文件SHA/原payload/SSE/usage/Graph/分配与封存通过。第二次本地audit v1仅CLI参数不支持退出2，无GPU或数据重跑；修正v2独立验收通过，旧错误日志保留。
