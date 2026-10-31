# 生产组件诊断：prefill attention是下一优化重点

目标是解释现代BF16-XQA参考下低压ToolAgent的FP8延迟退化，并指导可部署修复。保留生产Graph/prefix/调度及原连续90秒source窗口（399warm+238measurement），s=.125；本进程有界观察不是新的headline吞吐复验、自然语义质量或HBM流量实测。

| 尝试 | 已确认事实 | 终态与下一步 |
|---|---|---|
| a/source dcbea58c | 新kind未加入压缩传输白名单，mkdir前拒绝；两次同boot S0确认无目录/进程/容器/选用卡计算任务 | 本地NOT_LAUNCHED/0GPU-h，旧编号/replicate不重用；已补真实解码预检 |
| b/source dcbea58c | BF16真实数值/Graph算子通过，小Graph profiler检查拒绝；未加载14B/运行AIPerf或FP8臂 | rootFAILED、driver释放，0.052761936GPU-h全结算，48文件21,747,368bytes逐SHA保留 |
| c/source82bed5fd | 仅修复探针CPU/GPU annotation类别；源码CPU92+skip1、控制器181、真实frozenpolicy/压缩解码/全部资产SHA与freshS0通过 | 两臂真实637请求、各3窗口/48forward与Graph通过；rootCOMPLETE/driver释放、116文件208050818bytes逐SHA镜像及独立审计完成 |

b的原probe.trace确有6个CUDA kernel、2次cudaGraphLaunch、单逻辑设备0。CPU user_annotation2个与GPU gpu_user_annotation2个同名；旧检查仅按name计为4，要求2，确认为检查器误计。新检查显式识别CPU类别，仍要求真实kernel/launch/设备，缺CPU annotation不能以GPU annotation冒充。b未执行到数值输出assert，不能回溯为完整探针通过。c仍需重新全部实机门槛，原b不改成功。

每臂三个窗口≤16forward，forward完成边界检查6秒，trace≤32MiB；实际序列/token和已有host上下文长度随窗口记录。不采shape/stack/memory，不系统attach或改driver/perf权限。精确AIPerf phase时钟、窗口进程/原始trace SHA及真实Graph活动独立核对后才分析；CPU scope、CUDA kernel累计时间与重叠busy union分开，Graph只归属整体，不伪造其内部writer分解。无法关联的correlation保留unassigned。

证据：reports/operations/20261026T075846Z/component_diagnostic_b_raw_failure_and_CPU_repair_v1.json、component_profiler_failure_probe_readonly_b1.json、component_diagnostic_prepare_absence_resolved_a2.json；镜像results/mirrors/prod-diag-bf16xqa-toolagent-s0125-r1-20261027b-failure-c1/files。控制器累计已结算8.151520315GPU-h，第一12GPU-h自限未重置；c整项≤1.5GPU-h/3GiB，仍重新120秒空闲/归属/预约/UUID/私有容器/共享资源/watchdog。GPU5–7他人占用不使用。阶段性提交不结束授权研究闭环。

## 2026-10-27T08:15:23.663151+00:00 实际轨迹与部署判断

confirmed：637请求/臂（399连续逻辑warm+238measurement）全部成功，逐payload/SSE/usage/原OSL/固定到达、源码/二进制/输入/封存/真实Graph及CUDA活动均独立验收。三窗口内BF16/FP8共28079/28938个实际CUDA kernel，31/30次模型Graph replay。GPU窗口也用baseTimeNanoseconds+ts独立校验在实际测量phase内；没有把CPU scope当设备时间。

| arm | prefill批次 / querytokens | prefill attention设备累计ms | writer设备累计ms | writer占全部kernel时间 | Graph decode批次 | decode整模型累计kernel时间/批中位ms |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 17 / 138589 | 2090.391 | 51.029 | 0.614% | 31 | 20.063 |
| fp8_e4m3 | 18 / 149546 | 4016.832 | 40.491 | 0.399% | 30 | 20.212 |

这三窗口实际query总量和batch组成不同，不能把2090→4017ms当纯dtype造成的92%退化。设备writer仅占全部kernel累计时间的约0.61%/0.40%，没有支持“量化writer设备时间过高是主要瓶颈”的旧猜测；FP8 writer CPU scope总68.84ms对BF16的22.10ms，CPU提交与设备执行仍分开。Graph只按整模型归属，不能从约20ms/批的中位数推断请求生成期在生产中等价。

一组相同记录元数据（window2第一批、510 querytokens/1sequence、当前KV1022）中，48层prefill attention设备累计1.749435→3.534492ms，writer .204257→.193314ms，materialize .149917→.153375ms；其余未标组件的kernel累计19.422688→19.463848ms。这是优先优化FA2 FP8-KV prefill的实际线索，仍只是单个批次、非重复微基准、未核对张量值完全相同，不能推广为所有生产请求的因果倍数。

元数据解释更正：immutable字段kv_context_tokens原样存了C++ getter的CUDA累计KV长度向量（0起、sequence数+1），相邻差才是各sequence含当前query的KV长度。source py_attention_metadata.cpp getter与batch_input_builder.cpp的CUDA累加逻辑及全部96个forward的布局已核对；不把累计值误当每序列长度，也不改raw。99.708%/99.761%的kernel时间可关联到记录forward；窗边界外约24.24/24.22ms和未细标组件均保留，不为凑100%强配。

下一步先补真实原生工具调用的函数/参数正确率与服务表现，再据实际H800兼容/质量选择prefill修复。FA3 FP8-Q是新增数值近似，不能只切planner或把旧FA3全路径质量当FA3+XQA组合认证；当前源码/FlashInfer版本保持，优先独立可部署方案。保持BF16默认/FP8显式可选建议，质量损失仍未经用户接受。没有HBM硬件流量或新headline性能复验声明。

原始镜像results/mirrors/prod-diag-bf16xqa-toolagent-s0125-r2-20261027c-terminal-c1/files；independent audit component_diagnostic_independent_audit_c1.json、活动归属component_diagnostic_activity_times_c2.json和summary component_diagnostic_descriptive_summary_c1.json均在reports/operations/20261026T075846Z。可复算scripts/analyze_component_trace.py与summarize_component_times.py；总kernel/busy union/CPU scope和外窗未归属时间分别保存。c实际占卡0.596491044GPU-h，含全部加载/JIT/Graph/warm/观测/回收；累计8.748011358、第一12不重置，剩余约3.252GPU-h。未触碰GPU5–7他人任务，不在验收/提交后结束研究。
