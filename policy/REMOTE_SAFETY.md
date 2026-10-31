# 共享 8×H800 / 高权限 SSH：Agent 必须遵守的安全边界

这份政策优先级高于任何科研计划或 `--yolo`。本地 CLI 审批取消**不能代替用户对远端具体操作的授权**。不能用高权限身份的可执行能力推断拥有其他人的资源。任何不确定归属的对象 = 不能修改。

## 权限分级

**S0 — 自动允许，只读**：本地仓库/数据 inspection；SSH `id`, `hostname`, `pwd`, `nvidia-smi`、`ps`、`tmux ls`、`docker ps`（如已有权限）、`ls`, `stat`, `find` 有限深度、`git rev-parse/status`, `sha256sum`、`tail`/JSON 检查；只查看项目相关对象，不主动读取他人私人文件、不打印秘密。SSH 初次连接指纹应以可信来源核对；不允许 `StrictHostKeyChecking=no`。

**S1 — 自动允许，用户所有权已核实的独立空间**：5060 开新分支改代码/自测/commit；H800 在**已确认属于用户/团队且明确授权的 `$BASE` 工作区内**用新 UUID/时间戳创建目录、不可变日志/manifest、独立 git worktree、独立 build、上传已有 bundle、重用而不覆盖冻结数据/已核实依赖；只恢复我方已识别、仍安全的作业。严禁未经核实就覆盖、重命名或递归改变既有文件。

**S2 — 可自治但需事先有明确范围的 GPU 预算许可**：`state/approval.json` 必须由用户填真实 SSH/服务器/允许目录/最大 GPU 数/每次 GPU-hours/总 GPU-hours/有效期；代理不得自行提高预算。GPU 独占检查（每张卡显存/利用率/compute apps/PID, UUID）、至少 120 秒稳定空闲可作为初始保守值，每次启动/切臂前再次检查，允许设置合理保护阈值而非绕过。共享机没有预约服务时，启动前空闲不保证未来无人抢卡；如检测到后来的冲突，避免主动杀任何进程，记录并只调整**自己有所有权**的任务，谨慎暂停自己的未来队列。8 卡一起用完全可以，但必须经许可、有真实 8 卡并行价值、每张卡都空闲且共享服务器策略允许；单卡分片并行不称为 TP/DP，也不意味着单请求加速。

**S3 — 禁止自作主张，需明确人类授权和审计**：写 `$BASE` 以外路径、改系统/全局软件包、驱动、CUDA/NVIDIA 管理、Docker daemon/systemd root、用户/组/ACL、网络路由/firewall/SSH config、修改他人的存储与进程、安装工具到 H800 全局、创建无限制的高并发服务、开放公网端口、抓取他人敏感日志/密钥、对外上传模型/结果、开放 Docker socket 给不可信进程、因省事加 `--privileged`、修改图像/模型共享文件、任何 destructive cleanup。未经明确授权一律不做；当命令含 `sudo`, `su`, `rm -rf`, `docker system prune`, `killall`, `pkill`, `chmod -R`, `chown -R`, `find ... -delete` 等默认阻断，须独立人工复核并缩减范围；用户授权也不能改他人的资源。

## 访问与命令执行

- 确认 SSH alias、主机指纹、账户、HOME、工作区路径与容器权限后才可写；不要把已知历史用户名/hostname 填作猜测。不要修改 SSH agent/key 文件权限和远端 authorized_keys。
- 任何机器根目录或 mountpoint 失败检查时**不要继续**。`--network none` 适合已安装依赖下无网实验编译，必要时 cargo `CARGO_NET_OFFLINE=true`。H800 不访问公网上的 GitHub/Crates; 使用 5060 有网侧下载、Git Bundle/SCP、严格校验 hash、仅在我方空间部署。
- 使用非 root `--user UID:GID`；固定只读模型与历史结果挂载、所有新输出挂载为特定 rw 目录；已在本机验证 `--security-opt seccomp=unconfined` 可能是 Rust/CUDA 需要的特例，**不是要求 Docker privileged 或主机级调整 seccomp**。
- 不进行 wildcard process termination (`killall`, `pkill -f`, `docker stop $(docker ps...)`)，即使自己的 job 有残留，也要先匹配准确 manifest/container name/PID/start time/UID/cgroup 与实验 ID，只触及我方资源，避免复用 PID 误判。
- 记录 GPU 每次占用时间、UUID、空闲判定证据。未知占用、抢占、他人排队标记、驱动异常、OOM 冲击主机风险，立即停止新的 GPU 调度并等待；不是通过调整 check 值来放行。

## 失败与恢复

- 不创建第二个与当前状态不明的 job；SSH 掉线后不自动启动副本。已有 `COMPLETE`/manifest 优先识别，结果校验失败不能标 success。
- 磁盘余量不足、错误写入路径、权限异常、多个未知 worker、CUDA 驱动异常、他人 GPU 进程、非我方容器在实验 UUID 上，均为硬停止 GPU 启动条件。只读记录并转做本地安全工作。
- 对敏感文件与日志做必要脱敏，不将私密 SSH key、口令、token、内网身份凭据写入包、Issue 或公开 Git。模型/数据权属要尊重。

## 关于用户审批频率（v2 补充）

本政策 S2 是**一次范围审批，期间每项作业自主调度**，不是每次实验请用户批准。用户可在确认过的主机上授予全部 8 个可用 UUID、给定有效期和总 GPU-hours，之后 Agent 可按空闲状况使用 1–8 张卡；仍须每次运行前重新检查空闲并维护累计配额。批准文件推荐保存在本地 `~/.config/kv-research/approval.json`，不得由 Agent 自己编辑使其生效。JSON/Prompt 不是权限硬隔离；具备高权限 SSH 的 Agent 必须严格遵守本政策，条件允许时请求管理员配置真正低权限/白名单访问，禁止 Agent 自行改公共系统。

具体首次流程、批准字段和哪些事情再次请示见 `ops/APPROVAL_AND_OPERATOR_GUIDE.md`；新文件只能放 `ops/PATHS_AND_FILE_PLACEMENT.md` 列出的、经验证属于我方的空间。
