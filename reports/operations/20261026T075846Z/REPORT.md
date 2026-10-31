# 原计划正式实验前的本地验收

分类：confirmed 的 CPU/只读环境证据；GPU 结果另行验收。用户最新明确授权按需使用空闲 H800，授权处理由 Agent 记录，不再要求用户手工填写额外许可。原始文本见 USER_AUTHORIZATION.json；本地固定可信文件保存原文、哈希及自限的第一阶段额度，未把这些自限数字称为用户设定的预算。

本地执行器包含持久 SQLite FULL 预留/去重、严格 SSH ED25519 pin、任何写操作丢回执进入 UNKNOWN、只读恢复、全新 run/分臂缓存、非 root/network-none/内存及 CPU 有界容器、连续120秒空闲、切臂复查、PID/start/UID/cgroup 绑定、独立 deadline watchdog、逐臂真实驱动释放计量。共享项目 flock 只协调本项目控制器；不声称对其他用户建立 GPU 排他或 OS 级 SSH 沙箱。其他用户的新占用/预约仅触发停止我方任务。测试见 controller_acceptance.log。

真实 AIPerf0.13.0 源码从 H800 只读镜像。WSL 上实际执行 InputsJsonPayloadLoader、Pydantic Session/Turn/RequestInfo、InferenceClient、AioHttpTransport/AioHttpClient 到 loopback HTTP fixture：272 个冻结 session、544 次 HTTP 请求（字典和预编码字节两路径），max_tokens512/ignore_eos=true 和完整 messages 均未变化。证据 aiperf_wire_cpu.json。fixture 未加载模型、未使用 GPU、未作为性能结果入索引。

16K 与24K 的136条冻结输入均只派生控制字段，父输入未改变；manifest 逐条记录原/新 payload SHA。新增真实服务记录必须每条输出恰好512，包括warmup，否则失败；AIPerf 真完成、真实 Graph capture/replay、resident KV 张量字节/类型必须同时通过。

旧联合质量的72份 shard 全部SHA验证并恢复1719条唯一完整记录；原始数据不重写。实际 Qwen tokenizer 对全体 prompt计数；GSM8K按长度四分位各16条，LongBench八任务各选最短与最长，得到80条 pilot。真实官方 QA-F1/code-sim probes通过。CPU模型环境为WSL，不代替H800容器依赖或native生成验证。工具源码独立 worktree/branch 提交6a3e2463，旧 dirty checkout 保持原状；C++/kernels未变，可复用ceff二进制。

当前正式实验先跑原计划16K/C4与24K/C2的BF16/FP8固定decode对照，再做native fixed-scale质量pilot。成功旧实验不原样重跑。当前实验改变了实际输出长度控制，具备明确研究必要性。初次使用1卡，预算覆盖加载、Graph、warmup、失败与清理。

用户指出后端版本、生产配置、workload适配可能影响整体收益：记录为待验证的解释。既有结论仅适用于测试过的FlashInfer版本/配置/workload，不能据此否定量化整体价值。手头工作完成后据此展开分层控制研究，避免无目的补矩阵。

最新只读 pre_formal_readonly.json：0–3瞬时空闲，4–7被他人任务使用；不会使用或操作这些占用卡。正式run由远端主管理进程再次验证120秒空闲，不把瞬时空闲写成已通过稳定窗口。

## 最新用户生产目标覆盖旧优先级

用户明确本项目为生产支持决策。强制512合成配对已安全停止，仅BF16进入Graph/真实AIPerf阶段，无有效配对；原始远端通用停止状态TIMED_OUT，业务归类STOPPED_PRODUCTION_SCOPE_CHANGE。实际占用 0.146646 GPU-hours，已验证GPU0释放并结算。未触碰4–7他人任务。当前以ops/PRODUCTION_DECISION_STANDARD.md与新的NEXT_ACTIONS为准。
