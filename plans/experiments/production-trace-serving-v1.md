# proposal：原计划到达、前缀和容量的生产服务评估

最新状态：toolagent原速已cleanCOMPLETE、driver释放并独立原始审计通过；synthetic/BurstGPT已COMPLETE待raw审计；conversation末段CONFLICT保留并只读核查是否可恢复两臂，不重跑已完成请求。以下初始协议保留，最新实机优先。最高依据为 `ops/PRODUCTION_DECISION_STANDARD.md`。

先完成当前1719条质量的原始证据验收。若当前FA3/FP8-Q固定scale配置有明确质量损伤，优先改善可部署配置；不为补矩阵铺开受损配置。质量损失与用户容忍度分开报告，1pp只是工程提案。

## 输入和回放

- ShareGPT继续作为真实语义、自然停止的服务主线。已有C128恢复测量需干净复测，随后给正常低负载和持续需求曲线；只按统计或修复需要复测，不重启旧run。
- BurstGPT已逐字节核验完整1,429,737行来源和旧6385条peak10m。旧窗口是从源首时间戳5秒对齐的 `[3063605,3064205)`，保留相同窗口和秒级突发形状；两个原模型的聚合流量须明确映射为单实例需求。
- Mooncake conversation/toolagent/synthetic分别保留原始12031/23608/3993条来源。使用连续时间窗口，先预注册窗口和过滤规则，保留512源token块的前缀关系。240秒测量加前置60秒暖缓存是候选协议；最终预算和窗口取决于真实token总量，不能把稀疏264条替代完整时间线。
- 匿名trace缺少原始文本。若按逐请求原OSL仿真token工作量，则每条独立OSL、明确 `ignore_eos` 和length终止验收，标为“匿名到达/长度/前缀工作量回放”；它不代替自然生成质量，也不是强制统一512。自然EOS版本必须报告未复现原OSL的偏差。
- 过滤输入/输出非正及超出模型32K含模板预算的请求，逐项报告来源覆盖和剔除数。不得把源零输出自动记为新服务器成功或伪造其目标输出长度。
- 时钟缩放在新派生manifest里显式冻结，仅改时间字段。现装AIPerf的 `--replay-speedup` 仅适用于baseten，不能直接套用Mooncake/BurstGPT；synthesis变换可能改hash_ids，必须先验证payload/前缀不变。优先一次生成、封存payload后复用两臂。
- 必须使用真正固定调度并归一化起始时间，记录offered时间、实际HTTP发送时间、服务端响应和最终drain。客户端并发限制产生的延后不能算作服务器排队，不能只报目标RPS。

## 服务和观测

同一H800 UUID顺序配对，必要重复交替顺序；prefix、Graph、chunked prefill、128/10240/200调度及0.8显存比例来自实际可部署配置。现代算子按真实兼容/质量/部署成本选择，明确Q/K/V精度和scale，不限死FA2或FA3。

采集完成/错误/拒绝、actual tokens、TTFT/ITL/E2E分位、队列与drain、SLO曲线及每GPU完成请求成本。SLO未由用户给数字，展示候选阈值，不能声称用户已接受。负载选点覆盖正常、持续压力、突发，以及BF16驻留容量开始限制的区域，不追求OOM。

已确认我方服务可读 `/vars/<name>`。应用内轻量观测记录running/waiting、used/free/prefix blocks与累计prefix token/hit。源码的 `kv_cache_utilization_perc` 是used/(physical_pages−1)比例，used包含预留page0，满载可略高于1；preempted是当前迭代gauge，不能误标为百分数或累计抢占总量。free blocks低可能是可回收prefix占用，需结合有效使用和等待判断。无硬件计数器则不声称测得HBM GB/s。

## 当前CPU实现与边界

`scripts/prepare_trace_package.py`冻结原payload，只按明确倍率改变timestamp，前置60秒逻辑暖缓存与后续流量连续。`controller.trace_serving`核验每个尝试的payload/SSE/真实usage与源OSL，保留429/503/超时/错误，原始AIPerf精确phase epoch、credit及request-start时间分别用于offered lag和client delay；request-start不是服务端实际arrival。每臂同一私有容器内1秒采集bvars。尚未实机验证此trace入口，CPU通过不能当GPU性能。

单个连续toolagent payload约64.9MB，超过原32MiB上传单文件界。采用gzip传输、远端仅在新run展开，展开上限64MiB/精确长度/SHA全部检查，原32MiB单传输文件与64MiB总传输上限不改变；不会切片或损坏时间/前缀关系。local原大文件提案标记未启动、保留，非GPU失败。新包另外复制固定runtime，未来控制器编辑不改变已封存包。

## 执行门槛和停止

先CPU实际loader/tokenizer/wire/固定调度验证，再接入受许可、预算、唯一run和监督器保护的正式入口。每次选用GPU都需120秒稳定空闲、预约/归属/UUID/容器可见性复核；历史瞬时空闲不授权重复启动。原始SHA、Graph capture/replay、真正AIPerf结束和driver释放全部验收。质量或回放语义失败先修复，不用多跑掩盖。

## 较低到达速率与现代BF16参考（2026-10-27已冻结提案）

原速toolagent成功率16.05→31.26%、含drain完成RPS+119.812%，两臂均严重过载，示例低TTFT qualified速率为0。新s=.125将同2088请求连续时钟精确拉长8倍，保持每个session/payload/源OSL/prefix完全不变，每臂offered2400秒、measurement .8796875RPS；399前置warm来源请求仍连续，不单独drain。选择依据是原速实测过载，目标补正常服务区间，不是重跑原速成功点或预断言满足SLO。

业务source1c2be159、BF16 FA3-prefill/XQA-BF16-decode与原FP8 FA2-prefill/XQA-internal-FP8-Q-decode；新BF16参考b的真实模型/Graph/AIPerf单臂已worker验收，完整服务pair/独立审计仍待。新trace必须每臂重新通过数值/Graph/真实模型门槛。CPU154通过、实际frozen policy和全部2088 payload/clock比对通过，64MiB展开/128MiB XZ内存及32/64MiB传输界不变。

run提案prod-trace-bf16xqa-toolagent-s0125-r0-20261027a，GPU4仅候选；整项7200秒/2GPU-h、10GiB输出，含两次JIT/加载/Graph/失败/回收，初始12GPU-h不重置。只有再次S0对账旧run/精确UUID/预约/归属/120秒稳定空闲/CPU-RAM-磁盘/watchdog通过后才唯一派发；此段是提案，不能从CPU冻结宣称已经启动或满足正常SLO。FP8质量损失继续披露，不调评测答案。
