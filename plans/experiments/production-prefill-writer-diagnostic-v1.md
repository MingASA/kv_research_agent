# proposal：定位生产长上下文低压力的FP8开销

依据已独立通过的现代BF16-XQA ToolAgent两压力点。s=.125两臂完整成功/相同输出量/吞吐，无drain；FP8 TTFT P95 6.186→7.388秒、完整E2E P99 24.276→34.099秒。按同源请求配对，约87.7%测量请求FP8首字更慢，各input长度桶均有该现象；长度、prefix、到达和OSL组成不同，不从桶间差断言因果。s=.25时容量/等待减少与TTFT改善、生成期耗时上升并存。

本诊断只为选择可实际部署的prefill/writer/调度修复，不补吞吐矩阵，不把源代码推断写成瓶颈。当前C128完整质量先收尾；本案尚未实现/封存/启动，无新GPU结果，不计为已完成机制定位。

需回答：生产FA2 FP8-KV prefill是否比FA3 BF16更耗时、额外量化writer/materialization/scale写入占多少、Graph decode与prefill交错/实际batch是否解释低中压差异。依赖当前FlashInfer.6.18.post1与SM90实际路径，不因版本号旧就盲升；0.7.2.post1核心及直接headers字节相同不代表完整后端等价。

优先使用本方进程内PyTorch/CUPTI活动轨迹，保留生产Graph和实际kernel，关闭stack/shape/memory等无关采集，预先冻结有限窗口、事件数/输出大小；真实模型前在同隔离GPU容器验收CUPTI可用性/只观测本进程/Graph kernel可见性。不得系统级attach、申请全机profiler权限、修改driver/perf设置或观测他人GPU任务。工具不支持时停止此路径，继续源码/CPU分析，不静默改eager后声称生产Graph归因。

候选输入复用同ToolAgent窗口的连续前90秒source：完整前60秒逻辑warm后只观测首30秒source测量，s=.125，即480+240秒offered/臂。保留原顺序、prefix/OSL/过滤、匿名范围及父payload SHA，不抽稀/拼长/关闭固定调度。该片只回答这一工作段的组件时间，不认证全300秒、自然语义质量或最大容量。实现前先确认片段包含相关prefill和Graph decode，再冻结选择；若不足以回答问题，改设计而非无效占卡。

独立本地source分支、新commit/bundle、新run；正式服务与质量源码/旧raw不动。记录prefill/decode批次元数据、writer/attention/Graph replay边界与实际CUDA kernel。关键路径wall-time、kernel累积设备时间、异步重叠分别报告，不能把各kernel时间和直接当E2E；torch CPU scope时长也不冒充CUDA时长。observer开销属于诊断，不并入headline性能/不声称可直接上线。

候选整项≤5400秒/1.5GPU-h、3GiB，含两臂JIT/加载/Graph/warm/失败/回收，实际需求以CPU验收后估计。当前第一自限12GPU-h不重置，C128结算后读取真实余量才预留；所有选中卡须精确UUID/归属/无他人占用或已知预约、120秒稳定空闲、容器可见性、CPU/RAM/磁盘/watchdog通过。未知SSH先恢复旧任务，不发副本，不触碰他人。

如果真实clipping/underflow证据支持质量修复，再使用独立heldout输入改校准；若正常域误差主导，不能靠换2次幂global scale宣称修复。最终任何新配置都须真实算子/Graph、必要质量及目标生产workload端到端验证。当前工程方向是明确条件的FP8可选支持，用户质量/SLO界尚未接受，机制诊断不能代替质量或生产默认认证。

## 实现与冻结（2026-10-27，实机采集仍待验证）

独立源码dcbea58cafb007009f5b57aff4c6f8085cb75065，分支production/kv-component-profiling-v1；ceff C++源码/binary保持原SHA。源码CPU87通过、1项CUDA跳过，控制器176项CPU通过。输入从原已验证300秒窗口无抽稀保留首90秒：637请求，399完整前置warm、238测量，全部payload/session顺序/prefix/原OSL不变，新的tokenizer检查数为0，明确继承父全窗口的逐字节前缀验证。输入provenance位于results/prepared/component-toolagent-window-20261027a，不冒充新的模型或GPU结果。

新run提案prod-diag-bf16xqa-toolagent-s0125-r0-20261027a，仍为.125时钟/480+240秒offered。只对本方进程启用CPU+CUDA活动，生产Graph/调度及BF16 FA3+XQA、FP8 FA2+XQA不改。三个预注册窗口分别从观察到客户端后prefix计数增长的粗锚点加warm+15/75/135秒，于其后30秒内下一个prefill开始。锚点不假称精确offered origin，采后与真实AIPerf测量clock独立核对。每窗最多16个forward、在forward完成边界检查6秒停止、trace最多32MiB；记录实际序列数/token数/已有host上下文长度，无逐层.item同步。不采stack/shape/memory，不附着外部PID或采全机计数。

同一受监督单GPU私有容器中，模型加载前须通过真实CUPTI CUDA Graph kernel活动小探针（Graph在profiler前capture，验证实际replay及逻辑设备0），失败即停止本路径。随后服务真实采集需三个窗口均在测量phase内、有prefill writer/materialization/attention及实际Graph活动、trace SHA/大小/进程绑定全部通过，才可标记诊断完整性。新kind与原性能结果分类分开，trace开销不计入headline吞吐。整项5400秒/1.5GPU-h/3GiB，初始12GPU-h未重置。

后续a未prepare、b在模型前探针检查器类别误计而失败，两者保留终态。b原始活动实际可见，不是CUPTI不可用；source82bed5fd只修CPU/GPU同名annotation识别，92CPU+1CUDA跳过/181控制器检查通过。新c=prod-diag-bf16xqa-toolagent-s0125-r2-20261027c重新全部真实门槛，无已完成服务测量被重复。详细失败与当前进度见reports/experiments/production-component-diagnostic-20261027/REPORT.md。
