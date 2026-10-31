# 用户实际需要做什么？如何一次授权持续使用 GPU？

## 结论：不是每次跑实验都找用户

**一次性、限定范围的授权（standing authorization）**。用户可一次授权「未来 N 天、这个 SSH 主机、这个确定的项目路径、最多 8 张可用 H800、单项与总 GPU-hours 上限、磁盘预算」。**只要未超预算、未过期、没有危险因素，Codex 自主选择 1/2/4/8 张空闲卡并连续推进，不必每组问用户。** 每个作业启动前机器自行核查空闲状态是必须动作，但不是人工审批。GPU 全部空闲不代表无人预约：如能查看共享调度/团队约定，必须尊重；检查不了预约的环境，保守选择不扰人并持续监测。

## 用户第一次启动的实际步骤（只做一次）

1. 在 5060 的 WSL 新建 `~/kv-research-agent`，将包解压至此，从该目录运行 Codex。不要覆盖已有源码仓库。初次从 `MASTER_PROMPT.md` 开始；第一次保持 **S0 read-only**。
2. Codex 自动识别本机 xLLM Git 仓库候选位置、SSH 配置、可信 host key、远端登录身份与项目目录、已运行的作业、8 张 GPU UUID/空闲状况、远端剩余磁盘、Runner 状态。整理成 `state/REMOTE_DISCOVERY.md` 和 `state/reconciliation.json`，向用户给出**一次**简短的范围确认摘要；不要求用户逐条确认 SSH 命令。
3. 用户检查摘要**确认主机身份和工作区归属**，并在 `~/.config/kv-research/approval.json` 填写授权（或认可 Codex 填写的**候选草案**后亲自确认）。本包含 `templates/approval.example.json` 和 `templates/approval.standing_8gpu_EXAMPLE.json`。初始示例都是 `enabled:false`；**不代表已授权**。
4. 将批准文件以**仅本用户可读**权限保存到本机，避免放在公开 repo。Agent 只读批准文件、不得扩大或自行改写其数字/范围。若控制器仍只支持 `state/approval.json`，可以在本地保留一份只读的**引用**，但不能把 Agent 可编辑的状态文件当作真实授权来源。
5. 仅当 Agent 已经实现并测试：从**固定的批准文件**读取额度、run_id 去重、GPU-hour 计量、每卡资源竞争检查、远端归属判断和停机/恢复状态机，才在 S2 范围内启动无人值守 GPU 工作。当前包 `scripts/experiment_worker_stub.py` **不是已经完成的调度器**。若未实现，Codex 先开发并 CPU 测试、继续只读发现；不要谎称 `approval.json` 会自动拦截 SSH 命令。
6. 5060 机器保持联网/不休眠。Codex 控制进程在本地 tmux 中持久运行，远端长实验也各在**我方** tmux/supervisor 中持久运行；Windows 睡眠、重启和 WSL 停机可能让控制进程消失。重启后从本地 state/远端 ledger 恢复，**不能仅靠 `codex --yolo` 保证无限期运行**。

用户完成首次授权以后，通常只需偶尔查看 `reports/` 和 `state/STATE.json`，无需一直在线。一旦预算/权限过期，由 Codex 做安全的 CPU 工作并给出一次续期建议，而不是每天无限要求新批准。

## 批准文件的含义（模板的必要字段）

`ssh_target`：已发现并验证的 SSH alias/host，不猜测。`approved_host_fingerprint`：可信来源确认的 host key 指纹；`approved_workspace`：真实归属的远端项目根目录，真实绝对路径；`allow_remote_user_workspace_writes`：允许在该目录内建立新的独立文件；`allowed_gpu_uuids`：从 H800 实际发现并且共享规则允许使用的 GPU UUID 白名单（不只填序号）；`max_concurrent_gpus`：**同时使用** GPU 数上限，最高可授权为 8；`max_gpu_hours_per_job`：单项的 ∑(GPU 数×运行小时数)；`max_gpu_hours_total`：有效期内所有批准 GPU job 的累计消耗；`max_disk_gib_new_outputs`：新增输出总上限；`valid_until_utc`：UTC 有效期。`enabled` 必须由用户选择为 true。

**数值示例仅用于理解，不是自动填写**：假设同时允许 8 卡、一项运行 2 小时，那么该项消耗约 16 GPU-hours；如果一次授权总额度 96 GPU-hours，就意味着最多约 6 个这样的满 8 卡两小时作业（不计失败重试）。不允许用“空闲很多”自动重置或绕过预算。GPU-hour 按**设备占用实际时间**计费，不仅按成功的输出计；失败/OOM/被动中断都计入已经消耗的时间。

## 哪些行为仍然需要再次审批？

- 身份不符、新 SSH host/fingerprint、工作区归属或路径变化、模型/结果共享文件越权；
- 超出授权的最大 GPU 张数、GPU UUID 白名单、GPU-hours、磁盘额或有效期；
- 安装/升级系统级软件、改驱动/CUDA/Docker daemon、全局环境或共享依赖、修改/停止他人的进程/容器、清理公共数据；
- 改服务器权限、ACL、网络、用户身份，或任何明显破坏性命令（不因 `--yolo` 而自动放行）。

**普通科学/工程决策无需审批**：在我方 5060 git branch 改代码；在批准的 H800 工作区新增隔离 worktree/build/result；重试安全的只读探测；在已批准的预算内自主切换任务、测 C4/C8、使用不同空闲 GPU、开展质量评估、补充分析与报告；只要符合对照与版本规则。

## 不能忽略的安全边界

本文件和 `state/approval.json` 是**代理的行为契约**，但仅靠可编辑的 JSON、`AGENTS.md` 或 `codex --yolo` **不是操作系统层面的安全隔离**。如果 Codex 能直接使用高权限账号的无限制 SSH key 或 Docker socket，则技术上可能绕过任何自写 wrapper。为降低重大损害风险，**推荐在条件允许时由服务器管理员提供**受限 Unix 账号、仅项目目录读写/受限 sudo、共享调度器的配额或带 forced-command/命令白名单的 SSH 凭据；不擅自在 H800 高权限账号下自行设置这些全局功能。在未建立真正的 OS 级隔离前，保持详细审计日志、只在精确白名单目录写、不要让 Agent 对不受信任数据执行任意 shell 命令，并保守设定资源预算。

## 资源占用与安全反应

Agent 可在共享机器上自主使用多张 GPU，前提是每张卡开始前持续空闲、有正确 ownership 与团队使用规则、没有资源预约冲突，且 CPU/RAM/IO/磁盘不致压垮同机其他用户。对于非我方进程、显存占用、温度/电源/主机负载异常，不抢占、不断别人的任务，也不靠修改阈值绕过保护；安全地暂停**新的** GPU 调度，继续本地研究。

工作者自身启动的 GPU 作业如果超预算，只能按唯一 run_id、PID start time、容器名等核对后停止我方任务。SSH 不通后不立即启动同配置副本；先重连并只读 reconciliation。
