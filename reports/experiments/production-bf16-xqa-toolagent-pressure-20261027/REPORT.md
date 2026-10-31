# ToolAgent生产压力的两点观察

confirmed：先进BF16也采用XQA decode后，同2088条原匿名toolagent请求/payload/前缀关系/逐请求OSL/完整冻结运行时与source bundle，在两个到达速率下完成正式同GPU双臂测量。每点两臂都2088/2088成功，其中399连续逻辑warm、1689测量；无拒绝/错误。低速FP8没有吞吐增益且尾延迟更慢；中速主要降低等待与TTFT，同时生成期间每输出token完成时间变慢。物理KV约1.94倍不等于所有生产负载加速。

| offered req/s | arm | 含drain完成req/s | TTFT P95秒 | 完整E2E P95秒 | 完整E2E P99秒 | 完成/token代理P50 ms |
|---:|---|---:|---:|---:|---:|---:|
| 0.879687 | bf16 | 0.879687 | 6.186 | 18.809 | 24.276 | 61.651 |
| 0.879687 | fp8_e4m3 | 0.879687 | 7.388 | 19.884 | 34.099 | 71.091 |
| 1.759375 | bf16 | 1.714349 | 28.805 | 47.126 | 62.240 | 54.439 |
| 1.759375 | fp8_e4m3 | 1.727878 | 10.410 | 43.699 | 63.796 | 85.979 |

s=.125时clock拉长8倍，每臂总2400秒、测量1920秒，无末尾drain；s=.25拉长4倍，每臂1200秒、测量960秒，BF16/FP8 drain25.214/17.499秒。每点两臂相同测量输出304758tokens；两个压力点复用同一批来源，不是更多独立用户。完成吞吐分别+0.0%/+.789%，需求限制下的完成速率不能给出最大服务容量。前置逻辑warm不drain/reset，不自动认证稳态。

低压FP8 TTFT P95约慢19.42%，完整E2E P99约慢40.47%；配对来源首字约87.7%请求更慢。按真实input长度桶<4K/4–8K/8–16K/16–32K，TTFT配对中位差(FP8−BF16)约+126/+169/+320/+754ms；每桶分别487/870/208/124个测量请求。长度桶的任务/前缀/到达/OSL组成不同，不能由趋势隔离长度或kernel因果。中压TTFT P95改善63.86%，但完成/token代理P50慢57.93%，完整E2E P99慢2.50%；较好排队与较慢生成期可同时存在。

示例TTFT≤5秒且完成/token平均代理≤100ms：低速qualified完成率.396875→.340625RPS，中速.579570→.620972RPS。它们只是工程展示阈值，不是用户已接受SLO，不含语义质量，不称同质量生产goodput。需要按目标业务实际质量与延迟容忍度选支持范围，不能把少量较快指标合成为全面收益。

## 部署、机制与外推边界

source1c2be159/现装FlashInfer.6.18.post1/既有ceff binary/单H800/Qwen2.5-14B。BF16 FA3 BF16-Q/BF16-KV prefill+SM90 XQA BF16 decode，对FP8 FA2 BF16-Q/FP8-KV prefill+SM90 XQA内部FP8-Q decode；PDL=true、K.125/V.0625、有限新payload初始化/reserved page0，无BF16 shadow。prefix/Graph/chunked prefill/128seq/10240token/200admission/.8相同，客户端1024保留offered，不是GPU并行1024序列。

每个pressure pair内部同UUID、BF16→FP8，120秒稳定空闲与隔离保护；两压力点用GPU4/GPU2且时间不同，一对run/点，无run级CI，曲线连线只表达两个观测，不声明精确SLO容量或压力因果。旧FA3-decode BF16原速过载点采用不同参考，未混入本比较。

低压waiting中位两臂均0，中压6→0，running中位4→5及16→20；更多可用驻留与中压等待减少吻合。有效used比例分母不同、prefix可回收、1秒采样非原子，不用free接近0认定耗尽，不将NVML活动比例当HBM GB/s。prefill、writer、Graph与调度的各自贡献未隔离；下一诊断提案production-prefill-writer-diagnostic-v1仅在真实门槛通过后实施。

匿名文本只仿真原到达、长度、512token前缀关系，每请求ignore-EOS达到原OSL，不还原真实工具内容或自然EOS能力。AIPerf官方request_latency止于末非空content；完整offered E2E止于末raw SSE，后者不是TCP EOF。新增完成/token代理=(first-content至末SSE)/(server output count−1)，单token不定义，非硬件decode step/逐token stall。原raw/summary/冻结runtime未修改。

C8已测FP8相对新BF16参考GSMflex+.4549pp、LongBench−1.3726pp、LCC−5.38pp；生产C128同完整1719双臂质量已独立验收：GSMflex+.5307pp、LongBench−1.7705pp、LCC−3.92pp；详见production-bf16-xqa-quality-c128-20261027。1pp工程界未经用户接受。真实ShareGPT自然EOS的两次反序收益31.5–33%单列，不能将匿名trace替代语义质量或默认可上线证明。

## 证据

canonical reports/operations/20261026T075846Z/modern_BF16_ToolAgent_pressure_comparison_v1.json，SHA1d0d466acc341dc9f8df73cc1a6eb6dbf7f5bb5b2bf78b281c82cce235956298；脚本scripts/compare_trace_pressure.py核验全部源记录/payload SHA/完整runtime+bundle、模型/后端/服务配置和逐尝试源时钟/OSL。四个CPU变异拒绝通过；独立请求audit原样保留。两镜像各89文件398197787/395316425bytes，所有SHA及所有请求/usage/SSE/时钟/Graph/bvar分别验收。整项实际占卡约1.525413045+.874401231 GPU-h，含JIT/加载/Graph/warm/回收，不拿observation速率代表cold-start成本。输入长度派生分析ToolAgent_paired_latency_input_groups_CPU_v1.json不产生新GPU或预测。
