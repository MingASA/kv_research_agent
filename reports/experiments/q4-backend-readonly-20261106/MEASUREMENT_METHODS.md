# 测量方法：受限组件完成，生产Graph BLOCKED

执行源码为自定义MX ba5、冻结driver71bf与部署FlashInfer0.6.18.post1@8bc；current FI4c2/vLLM b209仅作为源码审计。H800 SM90、BF16、Q24/KV4/D256/page128固定，seed20261106，case及执行顺序不变：B1 Prefill8192、B32 Decode2048、B32 Decode16384。Writer新token分别8192/32/32；未加载模型，未测质量或服务。

独立reference采用原atol/rtol=.02/.02。Prefill执行全部8192行，reference固定16个query rows；Decode覆盖32请求的单query。原.001/.001 strict结果完整报告；same-fused matched控制为hard gate，不同native route strict比较自v2起是必报非primary诊断。该修订发生在首Pilot失败及独立CPU复核之后、main之前；原FAILED和原容差保留。

三条non-split pagedPrefill Graph-ready路线明确BLOCKED：mx_prefill_wrapper、nv_prefill_wrapper、matched_nv_prefill_wrapper；其wrapper、Graph plan或capture均不触达。原因固定为pinned_FA2_graph_ready_non_split_padding_mask_defect。Writer Graph、native split Decode Graph与其changed-input/plan guards分别hard资格；成功仍保持production_graph_qualification=false、all_requested_graph_paths_supported=false，完整Prefill Graph N/A。

每个case先完成primary timing，再观测每组件两个实际Eager API调用，各自单独profiler context与raw trace。逐call检查SHA、完整正值kernel事件和独立annotation后，分别求kernel duration和，再取两call均值。19组件共38调用，58 kernel、6 copy。API-active是kernel活动和；并发interval union另算、不用于共同分母。copies/host gap另列。

共同component proxy=同case Writer API GPU-active均值+Attention API GPU-active均值。NV Writer保留scalar-stack/validation GPU工作，原总值不改称纯codec；exact kernel名可另分项。native Attention包括compute与split/merge全部kernel。matched-NV仅为Attention控制，不配造Writer。融合Scale/Unpack独立时间与百分比N/A；route差值无法分离它。

Eager CUDA-event span为warmup10、repeats40、inner5；Writer Graph32/native Graph8分别按burst归一化，保留1160样本，单独报告，不与CUPTI混算share。host plan/stage采用warmup3/repeats20，API-return与async-drain wall分开，一次/model step、不乘FullAttention层数。CUPTI资源字段保留；没有硬件stall/spill/实际occupancy计数器归因。两个CUPTI调用、固定顺序和有限三个case不构成统计显著性或生产排名。

root/supervisor完整镜像与生命周期链核后，冻结标准库读取器独立验证result/acceptance、source/driver/plan身份、所有硬门槛、false flags、blocked routes、rawtrace集合/SHA/事件及16MiB cap，再输出精确CSV。CSV使用LF和Python float round-trip精度；图表只显示适当小数，原CSV不重舍入。Pilot数据不进入main表或图。

旧217普通wrapper capture及6-launch模型窗口保留有限历史观测；生产Graph安全当前BLOCKED。旧质量/工具/服务分数不修改，缺陷构成执行安全混杂，不能独立归因格式数值，也不预设每个旧请求触发。详见[审计更正](GRAPH_AUDIT_FOLLOWUP.md)。
