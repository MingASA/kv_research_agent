# proposal：原生 KV 固定 decode 长度对照 v1

提出于 2026-10-26 UTC；尚未启动、没有结果、没有分配 GPU 预算。
依据：`reports/experiments/ceff-takeover-20261026T040426Z-v2/REPORT.md` 和 `pairs.csv`。

## 研究问题与必要性

现有九个条件是有效的自然停止服务测量，实际 OSL 约 211–287，不能当固定 512/1024 decode 工作量。
24K/C2 的 FP8 token/s +5.92%、ITL P50 −10.86%，同时 req/s −9.70%、输出长度 +17.29%、TTFT P95 +15.38%。
本方案改变明确的控制变量，研究输出长度与持续并发变化是否解释这组权衡；不是重复已成功实验。

**hypothesis H2a**：固定 decoded steps 后，24K/C2 的 FP8 token/s/ITL 仍有改善，则单纯输出更长不能完整解释局部收益。
**hypothesis H2b**：固定长度后收益消失，则现有自然停止数据不足以隔离 KV dtype 的内核/访存收益。
二者都不直接证明 HBM 瓶颈；需要后续 app-local 时间分解。

## 对照与输入

- 源码/二进制继续锁定已成功的 ceff SHA 和 `277bd8…b21f3`，不重编译。
- BF16/FP8 同 FA2 paged prefill/decode、同 Graph、prefix=false、同一空闲 GPU UUID。
- 优先两个点：16K/C4 基线、24K/C2 交叉点；OSL=512。先完成一个点的配对和验收再扩展，不增加整个矩阵。
- 从原冻结 canonical 文件派生全新本地输入；messages、session IDs、顺序、模型保持不变。
  新 manifest 记录 parent input SHA、逐 payload SHA、修改字段、派生源码 SHA；原文件和旧 manifest 不改。
- 在新 payload 明确设 `temperature=0`、`top_p=1`、`ignore_eos=true`、`max_tokens=512`；不设 stop 字符串/stop token IDs。
  这些是**待验收的配置**，不能以 JSON 字段存在当服务生效。
- 本地 ceff 的 `xllm/proto/chat.proto:118`、`request_params.cpp` 和 `stopping_checker.cpp` 证明服务源码支持 ignore_eos。
  已只读查看实际安装的 `InputsJsonPayloadLoader`，源码将每 payload 直接交给 `Turn.raw_payload`，支持额外字段保留。
  仍须 CPU 验证 model/transport 链路与最终 HTTP body，不能只以 loader 源码代替端到端验收。
  如丢字段，先开发独立版本 loader/适配器并 CPU 测试；不修改共享安装或冻结历史输入。
- 首个新请求通过 SSE/usage 验收：输出 token count 必须达到要求，finish reason 必须解释长度终止。
  如特殊 token 解码/计数语义导致偏差，先查 raw token/usage，不把假设的 512 写成实测。
- 此负载忽略 EOS，只用于性能/机制控制，**不能作为自然语义质量评估**。

## 顺序、重复与统计

第一轮仅一次配对验证协议。通过后在许可预算内最少三个 paired run replicates，交替或冻结随机种子决定 AB/BA 顺序。
保持每组同 UUID；如改用不同 UUID，采用区组设计，不能将 GPU 差异与 dtype 混在一起。
128 profiling + 8 独立 warmup；phase/session/输入 token counts、逐请求唯一 ID、输出 counts 均审计。
保留自然停止与固定长度两套独立结果，不混算速度比例。

主要量：actual output tok/s、req/s、TTFT/ITL/E2E P50/P90/P95/P99、token-weighted decode 时间、prefill/decode 时间分量。
吞吐按 AIPerf observation window 计算，另保留 first-request/last-response 与 metadata request span；不要混用 `benchmark_duration` 字段。
不把同 run 的 128 个排队相关请求 bootstrap 成 128 次独立 run。
三个重复只能给初步误差范围；如效应落在 run-level 波动内，报告不确定，按信息增益决定是否追加重复。

## 资源与安全

初始规划 envelope：1 GPU，一次配对含加载/捕获最多 2 GPU-hours，新增输出最多 2 GiB；这些不是授权。
固定长度约是历史平均输出的 1.8–2.4 倍，不能直接沿用旧 case wall time；预算预留必须包含两臂 startup/capture/warmup/失败成本。
第一轮按实际占用计量后，自动在用户已签发的总额度内决定重复数；超额不启动。
运行前还必须完成 `plans/controller/EXECUTION_CONTRACT.md` 的执行器及可信许可。

停止：首次长度协议不生效、capture/replay 缺失、数据/源码/binary 漂移、OOM、他人占用/预约、GPU UUID 不符、运行超预算或所有权不明。
这些不是网络 transient；保存日志、修根因，只重试相关失败项。断线后先查旧 run，绝不启动副本。

## 结果如何改变计划

- fixed length 仍有局部收益且 native quality 合格：再做量化写入/prefill/decode 分解与安全 app-local profiling。
- 收益消失或劣化：保留负结果，优先 scale/质量/调度机制，不铺开更多负载。
- native quality 失败：性能作为受损质量配置的描述性上界，转向 scale/分层策略，不发布质量可接受的 Pareto 点。
