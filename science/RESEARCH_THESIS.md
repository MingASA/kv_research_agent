# 科研目标与问题分解

**2026-10-26用户澄清**：最终产出是xLLM完整支持KV量化的生产go/no-go及适用条件，不为论文或矩阵而测。以 `ops/PRODUCTION_DECISION_STANDARD.md` 覆盖下文固定FA2/唯一dtype对照的旧约束；主比较采用可部署的现代优化实现，并明确Q量化、算子差异、质量和实际服务成本。历史同FA2结论保留其范围。

## 终极目标

给出在**真实 LLM 推理服务**中，KV Cache 不同表示方式如何影响 (1) 质量、(2) HBM/KV 容量、(3) 吞吐及延迟、(4) HBM/算子/调度瓶颈，并用证据决定**何时量化、K/V 各量化到什么程度、是否需要运行时自适应**。工程证据应尽量逼近生产的 paged KV、FlashInfer、Cuda Graph、持续 resident 压缩表示，不要仅对 BF16 做 fake QDQ。

## 机制与可检验假设（不是事实结论）

H1 容量：KV cache 基本容量由 layers × tokens × kv_heads × head_dim × 2(K,V) × dtype bytes 决定；低精度在相同预算下提高可容纳 tokens / active sequences，不能把 allocator 预留显存相近误判为未压缩。

H2 访存：decode 每 token 需对已有 KV 做 attention；长上下文、高并发下 KV bandwidth 可重要，压缩有潜在加速，但可能被反量化/额外量化读写/内核兼容和调度抵消。需要实际 HBM 带宽/Kernel profile 来归因。

H3 质量：K 比 V 对量化误差可能更敏感；V-only BF16 K + FP8/INT8/INT4 V 可能取得更好 Pareto 点；但现有 14B Joint FP8 与历史 mixed backend 不能直接跨实现比较，需要同后端控制/单独清晰对照。

H4 精度/Scale：FP8 E4M3 per-tensor fixed scales 在不同层、token 分布、上下文长度、任务上可能有 outlier 饱和或有效动态范围不合；目前 K=0.125,V=0.0625 来自单次 eager 校准，不应以 3 个 prompts 当多场景质量认证。未来探索 per-head/per-block scales、clamp、rotation、outlier-aware、自适应精度；每种均需要额外代价与机制分析。

H5 负载：短 ISL/低 C 可能受 launch/compute/tokenization/调度限制，高并发/长上下文可以体现 KV 优势；Graph capture、paged prefill/decode、prefix cache 策略必须配对，否则归因不成立。

## 生产相关的必要条件

- 真正驻留 FP8 K/V pages（或明确写出使用哪些 V-only、INT8/4 实现），不可 QDQ 回 BF16 称压缩。
- BF16 与 FP8 Joint 采用同一原生 FlashInfer FA2 paged prefill/decode，不把 Triton attention 与 FlashInfer attention 的速度差归因于精度。
- 两臂同 binary/commit/model/tokenizer/GPU/并发/流式接口/输入/Graph/prefix cache 等参数，唯一自变量是 KV 精度与其明确包含的 scale/store 路径。
- 性能：AIPerf 端到端吞吐、request latency、TTFT、ITL、完成/失败、实际 tokens；资源：resident bytes/token、page capacity、GPU HBM 与带宽 profile；质量：paired GSM8K, LongBench, flip 等。
- 过程成本：量化写入、dequant 访问、Graph capture 预算与 startup 成本；将 GPU 占用时间/成本与统计不确定性纳入决定。

## 超越当前工作

性能矩阵及质量验证后：寻找可预测的 KV 特征 → 选择 K/V layer/head/block 精度 → 对收益进行在线 cost-benefit 预测，必要时学习型策略；先进行可解释的离线规则基线，不先验假设 RL/Agent 是最好方法。所有新方向均提出 falsifiable hypothesis、baseline、实验成本与停止条件。

## 2026-10-26 UTC 的证据更新

真实 ceff Graph/驻留压缩及九个可变输出性能条件已确认；质量、满载容量与瓶颈机制仍未验证。具体 H2a/H2b/H2c 与“驻留 scale 字节不等于 attention 读取字节”的源码推断见 `science/hypotheses/h2-decode-pressure-20261025.md`。保持原研究总目标，不把此次测量验收当终点。
