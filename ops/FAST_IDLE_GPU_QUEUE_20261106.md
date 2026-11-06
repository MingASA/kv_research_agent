# 5秒GPU观察与有限实验队列

用户2026-11-06明确要求：每5秒观察空闲卡，有空闲即派发；运行中同卡出现外部占用，只退出我方实验并继续观察；主Agent每20分钟常规复核，变化立即交接。5060Ti/SM120复现已被用户取消。

入口是 `python3 -B -m scripts.fast_idle_research_queue --manifest <冻结manifest> --state-dir <新目录>`。当前实例用 `plans/fast-idle-20261106b/manifest.json`。旧实例已退出，旧numeric-0003原票据已完成、释放并结算，不得重启。

脚本通过固定身份、固定主机指纹的只读SSH流，正常每5秒查询GPU UUID、显存、利用率、compute PID；只保留最新帧，不把全量轮询打印到stdout。新鲜度、boot、授权、预算或SSH状态不明时拒绝新准入，并只读恢复。stdout只记录状态变化和至多每20分钟一次常规心跳。

显存为0、无compute PID、利用率不超过5%只是空闲候选。准入仍经过原Controller的授权、预算账本、预约、UUID、容器隔离以及create前新快照检查。仅带精确 `resource_admission` 契约的本轮单卡作业采用0秒稳定等待、5秒复查；没有该字段的旧作业保留120秒稳定等待、30秒复查。

所有新派发仍通过 `scripts.launch_guarded_once`，保留per-run锁、唯一intent/ticket/ledger/history及exclusive日志。SSH丢失后只对账原票据；UNKNOWN、数值/执行失败、人工停止、身份变化不会变成自动重跑。只有真实资源冲突或明确未启动的资源拒绝，在我方释放得到证明后才换新ID重新观察；优先避开刚发生冲突的卡。

远端原supervisor继续每秒检测同卡进程/预约。相同UID不足以证明归属；必须匹配精确进程身份和我方容器。出现外部/未知进程时只停止我方确切容器。若他人仍占卡，按既有两次只读见证流程证明我方已释放，保留原UNKNOWN结果，不改写原始文件。

manifest固定有限截止、阶段包SHA、runner SHA与实际测试过的Controller核心SHA。阶段包由已冻结模板产生，新尝试复用原字节；前置阶段只有COMPLETE才能推进。主Agent只在真实资格通过后用 `NEXT_STAGES-0001.json` 等不可变交接追加后续阶段。

恢复使用原manifest与原state-dir加 `--resume`；不能对状态不明的启动创建副本。退出回执保留原 `EXIT.json`，再次退出另存带时间戳回执。停止本地队列不代表停止远端已运行作业，退出回执明确列出必须接管的原run ID。

监督Agent为用户指定的 `gpt-6.1-sol / max`，只读核对原票据、队列和GPU，变化即时交接。主Agent负责所有恢复、结算、分析与新派发，不能仅留下脚本就结束接管。

CPU验证覆盖：新5秒策略、旧120秒默认、create前竞态、冲突释放、重复派发拒绝、过期/异常流、失败不重试、完整阶段交接、历史索引字节不变及恢复后退出回执不覆盖。实际结果35 tests和15 subtests通过。该软件守护不是OS级独占锁；共享节点状态仍须持续检查。
