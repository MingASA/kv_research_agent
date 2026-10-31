# 新模型原生 KV 顺序队列

2026-10-28恢复：旧a队列在FP8 m启动后遇真实foreign UID0 PID660951，安全CONFLICT并结清55.675052秒，未加载模型。旧a保留HALTED；当前计划改为 `plans/queues/nextgen-core-sequential-20261028b/manifest.json`，FP8新o（同experiment_key/replicate1的资源恢复，非独立质量采样）→原未启动NVFP4 n，窗口结束时间不延长。新policy在任何GPU准入前重新核验原终态SHA、真实外来PID、原GPU释放、没有模型产物及相同源码/配置/输入；不自动重发未知任务。304 CPU tests及真实远端只读恢复先决条件通过。

用户2026-10-28授权：GPU忙时通过脚本等待空闲，检查好实验后自动轮流运行。本地入口为 `python3 -B -m scripts.guarded_native_queue`，使用现有受保护 CLI；没有新增远端调度器、系统服务或 Codex。

本轮冻结计划：`plans/queues/nextgen-core-sequential-20261028a/manifest-a2.json`。原manifest a1因一个本地评分器路径使用相对路径而未通过静态检查，保留失败证据，a2只修规范绝对路径，SHA不变。运行状态：`state/queues/nextgen-core-sequential-20261028a/STATE.json`，该目录的 `EVENTS.jsonl` 与 `GPU-S0-*.json` 保存实际观测。运行窗口截至manifest中的UTC时间，轮询间隔300秒。WSL和本地进程必须保持运行；不是跨机器断电自动恢复或OS级隔离。

顺序为新ID FP8 `...20261028m`、NVFP4 `...20261028n`，同一GPU0，最多各2 GPUh；不复用失败k或未启动l的ID。两包使用同一已核验C++二进制dbc1c53、Python提交03134b72、相同服务配置与1720个真实质量请求。FP8 K/V尺度为.0625/.5，NVFP4为.015625/.0625，来源为BF16 i独立校准。BF16 j原FAILED保留，完整raw的本地恢复评分作为参考，不重复GPU生成。

每次候选启动前：

1. 重读固定私有授权、持久化预算和所有既有运行状态。检查包、分析源码、独立标签与基线SHA。
2. pinned SSH只读发现；root vLLM分配仍存在时等待；所选卡任何显存、利用率、compute PID或身份不明都不作为空闲。
3. 空闲候选经过实际CLI新鲜预检、不可重复预留/启动，以及远端至少120秒稳定空闲、预约/UUID/容器/驱动校验和归属watchdog。粗轮询不能替代这些门槛。
4. 每包必须通过小算子数值、Prefill/Decode/changed-input Graph、真实模型20条已知输出与Cache生命周期检查，再生成完整1720条质量数据。FP8新Q/P计算参考只有CPU证据，实机相关性仍是硬门槛；数值容忍度没有放宽。
5. 完成且驱动释放后，分块只读镜像、逐文件SHA独立校验、正式评分并与BF16逐题配对，才进入下一包。分数通过不等于用户接受质量损失；本轮不是AIPerf性能测量。

单实例flock防止第二个runner改状态。持久化启动意图、CLI PID/start/UID/boot和原run票据防止重复；恢复时原CLI仍在则等待，原run仍在则只对账。失败、冲突、UNKNOWN、SSH有限只读恢复耗尽、文件变化或预算不足都停止后续GPU派发，保留原作业与预算。脚本不发送stop/kill，不改他人任务。HALTED状态不自动重置；先核对原票据与远端证据，再由主Agent定位修复并生成明确的新版本。

GPU空闲检查的实际CPU测试、两包frozen policy校验及只读现场检查位于 `reports/operations/nextgen-native-kv-20261028a/`。未派发的包不提前写入实验结果索引，避免历史去重把提案误识别成已运行实验；真实状态变化更新STATE/EVENTS/INDEX/NEXT_ACTIONS。
