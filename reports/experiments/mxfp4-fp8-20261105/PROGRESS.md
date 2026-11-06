# MXFP4 与 FP8 优化：阶段记录

记录时间：2026-11-06 UTC。**研究尚未完成；下表不能作为生产验收结果。**

沿用官方 `261274e5`、实际 BF16 A `721afdabc` 和历史量化实测 `da7c6446`，未升级运行依赖，未重跑既有 A。完整历史结果见[上一轮报告](../upstream-production-kv-20261102/REPORT.md)。

| 工作 | 已有证据 | 尚缺证据 |
|---|---|---|
| OCP MXFP4 | E2M1/block32/E8M0 CPU reference；5 项位级测试；早期 SM90 Writer 与 CPU 字节一致；私有 Attention 模块 SM90a 离线编译成功 | 新 Writer 边界值实机检查；Paged Prefill/Decode/Graph；新 C++ 构建；模型与服务资格；完整质量/性能 |
| FP8 Query | 合并非连续 Query 的读取和 FP8 转换，取消前置 BF16 连续副本 | 与原算子逐字节一致性、实际延迟、服务结果 |
| FP8 混合批次 | Decode 行拆给 XQA，Prefill 行保留 FA3；3 项 CPU 元数据测试 | GPU 算子门槛、临时内存、端到端收益；混合批次仍不宣称模型 Decode Graph 覆盖 |
| FP8 数值范围 | 独立 BF16 诊断实现与 CPU 检查；固定原 136 条独立校准输入及 K=.125/V=.5/Q=1 候选 | 当前执行路径真实截断率、误差；正式质量非劣效性 |
| 资源 | 第一次 MXFP4 作业遇他人占用后停止我方容器；第二次等待超时、未分配 GPU | 全 8 张 H800 目前被其他作业占用；新 C++ 构建所需 CPU 空闲门槛也未满足 |

## 格式与路径

MXFP4 使用[OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)的 E2M1 元素和每 32 元素一个 E8M0 Scale，未将 NVFP4 的 block16/E4M3 路径改名。D256 每个 KV Head 的物理 Token 槽为 128 字节数据加 8 字节 Scale，即 0.53125 字节/元素；Full-KV 理论压缩比为 3.764705882352941。**模型分配器实际可用 Token 数尚未测得。GDN 状态不量化。**

Attention 适配复用已安装 FlashInfer 的 FA2 Paged 加载器、规划器和 BF16 计算，仅将 block32/E8M0 Scale 在片上展开到加载器需要的位置；不创建全历史 BF16 KV 或全局展开 Scale。目标限定 SM90、BF16 Query、Q24/KV4、D256、page128；Elastic VMM 当前显式拒绝。正常生产默认分配器、Chunked Prefill、Prefix Cache 和 GDN 管理未被关闭；兼容性需要模型资格实测。

私有模块由本地 CUDA 13.2.86 编译，SO 为 7,025,360 字节，SHA256 `9508ae4716801261f3c77352738b33edb683d692a542aa50f7a6bceba80ab6dc`。H800 使用 CUDA 13.0.88；本地编译成功不替代其 ABI、运行及性能检查。实际升级依赖仍为零。

## 数值风险与冻结判据

![历史校准范围](cpu-figures-b1/historical-ranges.png)

该图来自独立的历史 BF16 校准，不是当前路径截断率。历史 V 最大值 149，当前 V=.0625 对应 E4M3 恢复范围 28；54/128 个 V Head/Phase 观测的最大值超过 28。原始观测没有保存逐元素截断率，因此不能从这些最大值推导质量损失。

已冻结同一批 136 条独立校准：128 ShareGPT、4 GovReport、4 QMSum；不使用正式 GSM8K/LongBench 正确答案。只有候选 V=.5 实际减少截断、且平方误差不高于当前配方，才进入正式质量判断。质量门槛与有限算子候选见 `reports/operations/mxfp4-fp8-20261105a/FROZEN_SCOPE-a1.json` 和 `FP8_COMPONENT_GATES-a1.json`；失败结果保留，不反向调参。

## Agentic 预筛选

![真实工具轨迹容量预筛选](cpu-figures-b1/agentic-capacity-screen.png)

原 32 项工具任务包含 352 次真实 BF16 模型调用，全部官方 CPU Token 计数与服务器 usage 相符。Prompt 中位数 4,561.5、最大值 6,492 Token，录制输出最大值 459。按每项任务自身历史峰值同时驻留计，需求为 158,967 Token、向上取整 1,256 页；BF16 有效容量为 1,345 页。该固定轨迹未显示必须通过 KV 压缩才能释放的容量，跳过其额外 GPU 压测。该界限仅覆盖已录制输出，不保证新候选会产生同样长度。

已核对实际 AIPerf 0.13.0 的 Agentic Code CLI。固定种子生成的 32 个合成 Session 共 74 次调用，18 个为单轮；默认大段共享 Prefix 与 65K 输入上限形成的分布不能代表真实编码 Agent。生成器名义 Token 数不冒充 Qwen Token 数，也不据此宣传 Prefix 收益。不扩展此合成矩阵。

## 交付状态与数据地址

代码已按 `mingasa` 身份提交用户 fork 的独立分支：

- [MXFP4](https://github.com/MingASA/xllm_quant_kvcache/tree/research/ocp-mxfp4-kv-20261105)：布局、配置、Writer、私有 FlashInfer 适配及实验性入口。
- [FP8 优化](https://github.com/MingASA/xllm_quant_kvcache/tree/research/fp8-service-opt-20261105)：Query 转换与混合 Attention 分派。
- [独立数值诊断](https://github.com/MingASA/xllm_quant_kvcache/tree/research/fp8-numerical-diagnostics-20261105)：仅供未计时的 BF16 校准，未合入性能分支。

本地原始记录位于 `reports/operations/mxfp4-fp8-20261105a/`；CPU 图表及精确 CSV 在 `cpu-figures-b1/`。`MXFP4_CONFLICT_RELEASE-a1.json` 保留首轮冲突释放证明；`MXFP4_OPERATOR_RECONCILE-r1-f1.json` 保留未启动终态；`local-sm90-compile-b1/` 保存原编译源码、日志和产物 SHA；`writer-sm90-compile-b1/` 保存次正规数修复的 PTX/Cubin 与 SHA。私有机器票据和原始服务文本不随此阶段摘要公开。

**当前合入意见：三个新分支均保持研究级，暂不作为生产支持合入。** MXFP4 尚未完成 Attention/模型/完整 Benchmark 验收；FP8 尚无新版本质量和服务收益证据。1720 核心生成、32 项工具及三组服务矩阵仍是未完成交付，原 BF16/FP8/NVFP4 成绩只作冻结对照，不填充新候选成绩。
