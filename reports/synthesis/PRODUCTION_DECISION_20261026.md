# xLLM KV量化生产支持决策

更新2026-10-27T11:59:38.595966+00:00。当前建议是有条件支持并继续工程化可选FP8路径，保留BF16默认。先进BF16也用XQA后，真实ShareGPT两次反序配对仍有31.5–33.0%请求吞吐收益、约1.94倍物理KV容量；生产C128全量质量则实测LongBench下降1.77pp，且低压力长上下文尾延迟更慢。这足以支持继续投入明确范围的FP8能力，尚不足以把当前配置作为通用默认、同质量净收益或所有KV格式的生产认证。

目标来自用户：决定xLLM是否完整支持KV量化，以真实服务的质量、成功率、吞吐、延迟、容量和维护成本作判断，不为论文或矩阵而测。当前已完成正式服务/质量这一轮闭环，机制归因和下一配置修复继续按信息价值推进。

## 当前生产选择

| 选择 | 建议 | 依据与条件 |
|---|---|---|
| 将受验证FP8作为显式可选特性工程化 | 有条件支持 | 聊天吞吐有反序重复收益、驻留容量收益真实；限定已测模型/SM90后端/Graph/scale配置，并展示任务质量代价 |
| 全业务默认改成当前FP8 | 暂不建议 | C128 LongBench−1.7705pp、MultiFieldQA−4.00pp、MuSiQue−4.49pp、LCC−3.92pp，低压力尾延迟更慢，用户未接受质量/SLO损失界 |
| 正常低压、延迟优先长上下文服务 | 当前点优先BF16 | ToolAgent .880RPS下两臂全成功/吞吐相同，FP8 TTFT和完成尾延迟更慢；只适用于此测量范围 |
| 容量/排队压力明显的服务实例 | 保留FP8候选 | 中压TTFT和waiting改善，但生成期耗时上升；按目标业务完整SLO/质量选择，不能只看TTFT |
| 原生工具调用、流式agent接口 | 当前保留BF16默认并修复接口 | 完整JSON两臂182/258官方参数匹配，FP8出现重复生成尾部；流式参数排序增量缺陷已CPU复现，完整JSON成功不能认证流式 |
| 质量敏感代码/多跳QA | 保留BF16默认并先修复候选 | 官方代码相似度不代表执行正确率；当前代理已显示风险，生产任务需要对应成功指标 |
| INT8/INT4、V-only、其他模型、TP/DP或其他GPU | 当前证据不足 | 历史不同实现不作本路径认证；按实际部署价值选择下一候选，不机械填格式/规模矩阵 |

这里的可选配置指模型/服务实例在执行前明确选择受支持路径；不是从部分失败的FP8运行中静默转换缓存或回退继续服务。完整产品支持仍需维护配置验证、真实算子/Graph门槛、版本契约、质量配方和观测，当前结果不是已经完成全部生产工程。

## 已独立验收的主要服务结果

| workload/压力 | confirmed结果 | 生产含义与限制 |
|---|---|---|
| 真实ShareGPT上下文/自然EOS，closed-loop C128 | 两次相反臂序、同GPU1/source/runtime/bundle/528原请求；RPS+31.507–32.965%、actual输出TPS+29.653–30.559%、ITL P50约−10–11%，两臂各512/512成功 | 当前主要生产增量依据；输出长度分别约少1.4–1.8%，不是同输出/同语义质量；两个run范围不是总体CI |
| 现代BF16-XQA ToolAgent .8796875 offeredRPS | 两臂2088/2088全成功，测量1689/1689、无drain；完成速率相同，TTFT P95 6.186→7.388秒，完整E2E P99 24.276→34.099秒 | 低压无整体吞吐收益，尾延迟更慢；同源配对约87.7%请求FP8首字更慢 |
| 同现代参考ToolAgent 1.759375 offeredRPS | 两臂全成功；含drain完成RPS+.789%，TTFT P95 28.805→10.410秒，完整E2E P95 47.126→43.699秒，P99 62.240→63.796秒 | 排队改善与生成期变慢并存：末SSE完成/token代理P50 54.439→85.979ms（+57.93%），不是全部延迟改善 |
| BurstGPT原peak10分钟、旧FA3-decode BF16参考 | 6385/6385全成功，测量6221/6221；两臂完成11.52037req/s相同，TTFT P95 329.57→338.49ms | 原平均输出约4.7tokens、offered限制，无整体吞吐收益；1秒bvar与秒级突发混叠不作资源优势证据 |
| Mooncake原速toolagent/conversation/synthetic、旧FA3参考 | 测量成功271→528/1689、219→373/831、342→381/791；完成RPS分别+119.812/+150.125/+35.789% | 三者原速严重过载，成功子集不同、尾延迟高；synthetic生成期更慢，不称正常SLO goodput |

ShareGPT两run按observation时长汇总RPS+32.233%、actual输出TPS+30.105%，对应steady完成请求GPU时间约-24.38%。这只用完成数/实际token/observation，未加入JIT/加载/cold-start与质量接受条件，不能当完整账单成本。两run复用同512测量+16warm来源，不称1024个独立用户。旧FA3-decode BF16参考对FP8的+152.584%保留为另一部署基线，不取代当前先进BF16、不混入复验统计，也不用跨GPU/run比值精确归因kernel。

两个现代ToolAgent点严格核验同2088源记录/payload SHA/源OSL/prefix/全frozenruntime+bundle，仅缩放连续时钟；每点内部同GPU两臂，点间GPU4/GPU2且时间不同，一对run/点，不给run级CI或精确SLO容量阈值。399前置逻辑warm连续不drain/reset，不自动证明稳态。当前两点受offered限制的完成RPS不能用来估计峰值吞吐。原FA3参考高压点不是这条同基线曲线的第三点。

## 原生工具任务与生产prefill定位

完整BFCL v4 live_simple原258题、16独立warm任务，真实messages/tools/自然EOS/greedy/1024 cap、C128，BF16 FA3+XQA对FP8 FA2+XQA。109文件逐SHA、真实API/Graph/双臂AIPerf及原官方参数匹配独立通过。两臂均182/258=70.5426%；175同时正确、69同时错误、7up/7down，不能按同总分宣称逐任务等价或质量接受。

完整JSON有限cohort observation 7.856→16.135秒，RPS32.840→15.990（−51.31%）、实际输出TPS1681.18→958.85（−42.97%），输出token13208→15471。FP8的live_simple_175-101-0重复sourceMapping至1024 cap、无tool调用、13.623秒；BF16同题78tokens/2.777秒，两臂都被官方判错。保留全部258和尾部，不剔除重跑。P95完整JSON延迟4994.45→4770.20ms，P99 5944.11→5992.24ms，指标分化与总cohort尾部同时报告。这是一轮短closed-loop任务集，不是稳态峰值，更不是一般tool业务−51%的结论。全JSON无流式TTFT/ITL；参数匹配不是外部工具执行或完整agent成功率，两个参考不一致题仍在官方分母。

前一次streaming API任务在BF16首warm失败，无AIPerf/FP8结果。原C++ parser/API增量序列化被本地CPU复现：后到的字典序较早参数重排JSON，旧字节偏移继续输出，合法输入在chunk1/2精确产生历史坏wire；另一些chunk产生合法但错误参数。原始模型token顺序没有导出，不称完整历史token回放。未做客户端修补或业务热改；这是共享baseline的生产接口债务，不能归因为KV量化。

真实Graph组件诊断c已独立验收116文件/3窗口/每臂637请求；writer设备时间BF16约.61%、FP8约.40%，prefill attention优先修复。单个相同记录元数据批次（1seq/510query/1022KV）attention48层1.749→3.534ms、writer.204→.193ms，但不是相同activation或重复算子因果实验。不同17/18批次和query量不用于92%因果slowdown，没有HBM字节测量。

针对该实际瓶颈，新source8af6e702在resident FP8之外预分配单层1536MiB BF16 scratch，按活跃唯一页解量化后用FA3 BF16-Q prefill，decode/XQA/Graph不变。88项CPU/576旧profile等价/离线SM90编译/192控制器测试及冻结policy通过，H800真实数值/Graph/页生命周期及52文件独立验收现已通过；记录形状CUDA中位仅-.5967%/wall-.5545%，增加1539MiB+4B，不显示足够部署价值，停止当前候选的完整模型扩展。这不是服务收益或对所有FA3方案的否定。下一步针对质量风险测真实K/V饱和/下溢，而非盲调scale。

## 生产客户端C128的完整质量

同一新run、完整原始1719/相同顺序、两臂同source11e7/模型权重/官方评分、自然EOS/greedy/任务cap、生产服务配置；真实算子/Graph和168个原始文件逐SHA、逐题重评分已通过。C128是客户端并发上限，不是每个实际GPU batch固定128。

| 指标 | BF16 FA3+XQA | FP8 FA2+XQA | 差值(pp) |
|---|---:|---:|---:|
| GSM8K flexible | 89.3101% | 89.8408% | +.5307 |
| LongBench八任务macro | 50.1885% | 48.4180% | −1.7705 |
| NarrativeQA | 32.5448% | 34.5297% | +1.9849 |
| MultiFieldQA | 48.6327% | 44.6327% | −4.0000 |
| MuSiQue | 37.6155% | 33.1269% | −4.4886 |
| LCC | 67.1400% | 63.2200% | −3.9200 |
| RepoBench-P | 49.1400% | 48.5200% | −.6200 |

全部任务分数和原始预测在C128正式报告/audit。GSM strict+2.8052pp但flexible仅+.5307pp，格式影响与内容能力分开。配对bootstrap5000/seed20261026：GSM flexible差95%区间[−.7582,+1.8196]pp；LongBench任务内重采样macro[−3.4893,−.1667]pp。LongBench区间在0以下，仍不外推为所有生产任务损失；1pp工程非劣界未经用户接受，不能由不显著或总分接近断言等价/可上线。每任务50题、题目重采样不是GPU运行重复。

先前C8新BF16参考对保留FP8为50.1252→48.7526%（−1.3726pp）。本次C128差扩大约.3979pp；C8旧FP8使用80+1639两批缓存，本次全1719同顺序，GPU/时间/调度历史不同，仅作为敏感性描述，不隔离batch因果或扩成更多独立问题。C128对C8 GSM逐题新BF16有18up/17down、FP8有34/32，总分接近不等于逐题相同。两代码任务均按原length上限评分，不将length判错或偷改cap；代码相似度不是代码执行或真实agent工具成功率。

## 容量、后端与生产维护

当前主配置为H800/SM90/TP1/Qwen2.5-14B，BF16 FA3 BF16-Q/BF16-KV prefill+SM90 XQA BF16 decode；FP8 FA2 BF16-Q/FP8-KV prefill+SM90 XQA内部FP8-Q decode，PDL=true、K=.125/V=.0625、有限新payload初始化/reserved page0。prefix/真实CUDA Graph/chunked prefill、128seq/10240token/200admission/.8相同。服务source1c2be159，质量11e7cddb全部推理引擎文件与1c2字节一致，复用已核验ceff binary，无共享依赖升级或BF16 shadow。

resident每token每层4096→2112B，下降48.4375%；physical pages1477→2865、含page0容量189056→366720tokens，约1.94倍。扣page0 usable为188928→366592tokens。allocator按相同预算分配更多pages，总预留显存接近不否认压缩；物理容量不是最大SLO承载能力。14B权重仍BF16、GQA已有40Q/8KV heads，MLP/线性计算未量化，不应期待整个模型按2倍加速。

中压应用采样waiting中位6→0、running16→20，更多驻留与等待缓解吻合；低压waiting中位均0、running4→5。used包含page0/分母排除page0、prefix池可回收、1秒采样非原子，不能只由free=0或归一化used判断耗尽。NVML活动比例不是HBM GB/s；组件诊断已提供限定窗口内的writer/prefill attention/Graph时间，kernel内部Q算术和HBM流量仍未单独测量。当前差异是实际可部署配置比较，不是仅KV dtype因果。

生产化必须保留有限新cache初始化、page0 padding/非连续Q/partial/replan/Graph数值验收、pinned backend与实际import版本、模型/scale配方、容量/accounting和指标口径。现装FlashInfer.6.18.post1；已对照上游.7.2.post1 SM90核心及14直接wrapper/header字节相同，不证明整个后端/JIT/性能相同。存在适用改进再隔离升级，不盲升或永久锁旧版本。

原过载三家族全部HTTP200并发上限纯文本notice保留，不能按200算成功，也不是KV数值损坏或JSON SSE error。这是生产API拒绝契约债务，需与量化支持一起明确处理；当前未悄悄改服务/错误码。conversation原root CONFLICT在末响应后8.901秒，measurement完整独立恢复，原终态保留，不改成功、不重跑。其他失败/回执超时/不启动记录均保留。

AIPerf官方request_latency截止末非空content，metadata.request_end_ns截止末raw SSE（非TCP EOF）。长匿名ignore-EOS回放可有无可见内容尾段，具体token IDs未导出，不把EOS或kernel原因写成事实。完整offered E2E和新增(first-content→末SSE)/(server输出count−1)代理单列，代理不是硬件step/逐token stall。自然ShareGPT和匿名长度/前缀/到达仿真分别支持不同判断，匿名trace不替代语义质量。

## 下一轮闭环

当前所有我方GPU作业已正常结算/driver释放，原冲突如实保留；无active/UNKNOWN。累计9.4507125811GPU-h、预留0，初始12自限未重置。共享机授权持续按需空闲，不使用他人正在用/已知预约GPU、不干预他人任务；没有跨用户OS级SSH/GPU排他。

组件诊断和原生工具任务已完成并独立审计，不重复启动。prefill工作区修复已验证数值正确，但成本screen不足以支持扩展，按预注册原则停止。下一项为独立BF16/Graph的有界跨请求K/V分布诊断，先CPU/实际小Graph门槛，再heldout真实聊天，不根据质量答案选scale。校准heldout128仍仅输入冻结，无激活/新scale，不用评测答案调参。保留流式API缺陷的CPU证据；源码修复与新binary需独立版本和必要接口验收。

证据入口：[ShareGPT反序复验](../experiments/production-bf16-xqa-sharegpt-replicates-20261027/REPORT.md)、[现代ToolAgent压力对比](../experiments/production-bf16-xqa-toolagent-pressure-20261027/REPORT.md)、[C128完整质量](../experiments/production-bf16-xqa-quality-c128-20261027/REPORT.md)、[C8新参考](../experiments/production-bf16-xqa-quality-reference-20261027/REPORT.md)。[导出SVG](production-decision-20261027/figures-v1/production_evidence.svg)、[PNG](production-decision-20261027/figures-v1/production_evidence.png)、同目录数据CSV/manifest含全部输入和产物SHA；生成器scripts/render_production_decision.py，已目视检查，无新增GPU。原四家族与旧FA3/FA2报告保留其基线和范围；STATE/EVENTS/INDEX/NEXT_ACTIONS跟踪实时状态。

本次新增证据：[原生工具任务](../experiments/production-native-tools-20261027/REPORT.md)、[真实组件诊断](../experiments/production-component-diagnostic-20261027/REPORT.md)。figures-v1仅覆盖此前服务/质量结果，未加入本次工具/组件数据。

[工作区算子及停止决策](../experiments/production-prefill-workspace-20261027/REPORT.md)：没有新增14B服务结果。

## 真实范围诊断的新增决定

heldout128 BF16/Graph的真实统计已独立恢复，87文件逐SHA/1536分组完整；Kmax38.5、V24.375，无FP8饱和，QDQ相对L2约2.65%，underflow误差能量上界不足总误差5e−9。writer token214986与143037prompt−2048cache+74125output−128严格对账。按预导出规则保持K=.125/V=.0625，不做无依据global调参。原c FAILED（时间导出精度后处理）如实保留，不当新FP8质量或服务收益；200CPU通过，无需GPU重跑。短聊天结果不覆盖长文/code或FP8反馈，也不把质量问题归因给Q。下一步先核实实际可部署的BF16 Q decode路线；若没有生产路径则不做GPU网格。详见[模型统计与恢复证据](../experiments/production-kv-range-probe-20261027/REPORT.md)。
