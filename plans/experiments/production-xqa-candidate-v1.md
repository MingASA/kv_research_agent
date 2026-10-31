# proposal：可部署的BF16-Q prefill与先进XQA decode

**最新状态优先（23:53Z）**：source ede2b9e7，算子/Graph及完整每臂1719质量已独立验收。c双臂80、a恢复FP8的1639、b仅补BF16的1639精确合并，不重复成功预测；a保留CONFLICT。GSM flexible89.2343→89.6892%，LongBench macro50.0885→48.7526%，LCC−5.38pp；不能认证生产默认，正式服务测量将保留此代价。无质量任务仍在运行。ShareGPT服务准备遇到SSH回执超时，正在S0恢复/实际资产传输验收，无XQA服务吞吐成功结果。以下首版提交/首次启动描述为历史设计。

当前FA3/FP8-Q固定scale的完整1719条质量已确认LongBench macro下降2.0075pp，不能直接认证为生产默认。此候选旨在减少prefill的Q近似并使用H800上的先进融合decode，不为补实验矩阵。是否改善仍是假设。

源码独立worktree `/home/mingasa/projects/xllm_quant_kvcache-agent-xqa-20261026`，提交d68f5500；完整质量对应的旧6bbe源码保持原样，可继续独立复算。无C++修改，复用已核验ceff二进制。现装FlashInfer0.6.18.post1，不升级共享依赖、H800无外网。

BF16参考为原生产FA3 BF16-Q/BF16-KV；候选为FA2 paged BF16-Q/FP8-KV prefill，加SM90 XQA decode（内部仍转换FP8 Q），PDL明确开启。K=.125/V=.0625不在评测答案上调参。prefix、Graph、chunked prefill、128/10240/200调度、0.8显存比例和C8质量一致。不同kernel是部署配置差异，不说成仅KV dtype的因果效应。

页表由现有分页元数据转换到固定地址的XQA block table/uint32 lengths，最大上下文32768；无整个BF16 shadow。SM90 wrapper的Q原始指针不携带Q strides，显式contiguous以正确处理fused QKV投影。空Graph行转成初始化过的reserved page0单token读取，其输出舍弃；不把padding当真实请求。转换与query materialization成本属于实际服务路径，后续性能必须计入。

第一项 `prod-quality-xqa-pilot80-r0-20261026a`，先FP8候选、再BF16参考，同一空闲GPU0顺序。整项wall7200秒、最多2GPU-h预留、输出5GiB；这些在已有按需许可的自限额度内。先官方scorer CPU实际调用，再在该受监督隔离容器做真实单层GQA40/8、非连续Q、partial/poison pages、context变化、padded bucket及capture/replay。FP32代表值参考另有relative L2≤5%约束和逐元素误差门槛。任何失败则不加载14B、不继续其他GPU矩阵，保留失败并修具体原因。

算子通过后才进行同冻结GSM64+8×2真实质量pilot，不能以80条声明全1719质量通过。两臂真实模型也必须capture/replay、驻留alias/尺度/后端标记、官方评分及逐请求完整性通过。结果确定下一步：若有改善且无接口风险，补充必要完整质量与原计划BurstGPT/Mooncake生产负载；若仍损伤，优先独立代表性校准或其他有明确部署价值的配置，不无限补受损配置性能格子。

CPU验收为源码91通过、1项CUDA跳过；控制器108通过。实机GPU准入与算子/模型正确性尚未发生。启动前仍由执行器重新核验身份/ownership/UUID、120秒稳定空闲和共享资源，再持久预留与独立watchdog；UNKNOWN不重启、只读恢复。

## 首次实机失败与新版本

a在d68f5500的NaN尾页Graph参考处全NaN，未加载14B，已释放GPU/结算0.045218874GPU-h，50份失败证据逐SHA保留。不是质量结果。下一版本0598400c只在首次绑定时初始化全部新payload，保证未使用区域有限；禁止改绑并保持同地址rebind的有效内容。极大有限尾页参考与独立NaN K/V尾页诊断并存，最后恢复有限态并重验，不降低误差门槛。原因仍需H800控制确认。新b任务复核同一保护流程；绝不重启或改写a。
