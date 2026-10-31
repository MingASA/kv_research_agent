# 持续自主运行手册（高层状态机）

## 生命周期

`DISCOVER_ONLY → RECONCILE → PLAN → PREFLIGHT → WAIT_GPU → RUNNING → VERIFY → ANALYZE → COMPLETE`；失败转入 `BLOCKED_* / RETRYABLE / NEEDS_USER_APPROVAL`。每一跃迁写本地/远端（在可写且属于我方时）事件日志。`RUNNING` 不是成功；退出 0 也要完整性校验；`COMPLETE` 也需匹配 manifest hash, n and verification report。

### 首次运行：恢复比启动更重要

只读：本地找 Git 与已有结果；SSH config（不擅自编辑）、H800 identity/指纹、`$BASE` 核验、`tmux ls`、`docker ps`、GPU `nvidia-smi -L` + 进程列表、实验目录/runner `status`/`COMPLETE`、已运行的过夜脚本和 `ps`。检查已有 `ceff00bb` 的 16K/C4 批次；如果成功就分析；若运行中就追踪；若失败就复用 runner 支持的 `resume`，且**仅在相同 manifest/source/binary 并无运行 worker**时；不明状态只做只读判断。首次核对一切不触发 GPU 模型加载。

### SSH 断连与退避

- `ssh -o BatchMode=yes -o ConnectTimeout=12 -o ConnectionAttempts=1`，配置 ServerAlive；`StrictHostKeyChecking=yes`/系统默认可信机制，不能跳过身份验证。
- 一次命令失败后先判断是否安全重试：**只读/幂等探查**可重试；创建实验、tmux start、写数据等非幂等 SSH 命令因连接断开返回失败时，状态视为 `UNKNOWN`，先重连核查，不自动重复原命令。
- 初始短断线指数退避 [15,30,60,120,300] 秒（附随机抖动）；连续约 10 次失败进入 10 分钟间隔巡查，最长 4 小时/日累计有限尝试，之后本地安全工作并写 BLOCKED。持续自理，不向网络发高频重试风暴；共享服务器维护应尊重维护窗口。
- 网络恢复：先查持久 runner/tmux/containers、GPU、manifest，再决定恢复，而不是从第一步无脑重跑。
- 丢失prepare回执且run始终ABSENT时，单次ABSENT不足以释放UNKNOWN。`scripts/reconcile_absent_dispatch.py`仅做S0双观测：确认本地旧launcher结束，远端精确身份/同boot间隔≥30秒、run路径不存在、没有未结束RPC（包含携带命令的shell）/本run进程/任何本run容器；任一身份缺席证据缺失或变化继续UNKNOWN。通过后仅本地结算NOT_LAUNCHED/0我方占用，明确不是远端result文件或GPU成功；旧run_id与科学replicate不释放去重。2026-10-27按用户委托的安全判断区分任务身份与共享卡可用性：其他/未知计算PID原样保留为busy证据，不证明这个从未创建的run存在，也不要求他人退出才能结清我方零占用。证书只证明本run从未准备，不证明GPU空闲、不授权新的GPU启动。占用/预约/未知GPU仍排除，后续独立新编号必须重新走完整120秒准入。失败观测在判定前持久化，不能通过改许可或SQL清零消除未知状态。

### 远端长任务

2026-10-28传输细化：只对已核验 `/usr/bin/ssh` OpenSSH8.9p1、无ProxyCommand/ProxyJump/LocalCommand、无stdout、退出255且恰为两行已知 `kex_exchange_identification: Connection closed by remote host` 的情况，允许在15/30秒退避内重连（总3次，每次写本地preauth回执）。这是认证/session/exec请求之前的连接拒绝，不能与可能执行过的写操作混同。依据[OpenSSH8.9 ssh_login调用顺序](https://github.com/openssh/openssh-portable/blob/V_8_9_P1/sshconnect.c#L1471-L1481)及[后续session创建](https://github.com/openssh/openssh-portable/blob/V_8_9_P1/ssh.c#L1575-L1607)。任意超时、部分stdout、认证后断开、远端错误、其他客户端/代理或无法精确分类的mutation失败，仍只执行一次并UNKNOWN，绝不重放；三次preauth拒绝后也停止并只读取证。StrictHostKeyChecking/主机pin/预算/所有GPU门槛不变，不声称已修复服务器或网络根因。

- 在 `experiments/<run-id>` 下保存 frozen config, manifest, log, state, job PIDs, tmux session, stdout, hash/argv/env 及结果；单一 `RUN-ID` 的唯一锁与 pid/start time 双校验；日志每 N 秒 checkpoint，控制 Agent 进程重启后读 ledger 恢复。
- 使用 tmux（如果可用）或可信 nohup，**一个队列 supervisor** 按前台子进程管理 GPU job，远端 SSH 退出不杀服务。不要在 shell 内简单 `... &` 后提前写 success；后台 supervisor 必须等 child 真实退出并核对 `COMPLETE`。
- Runner 的 `resume/status/verify` 接口优先于自写猜测。容器名、pid 和日志均记录于 manifest。避免两个 supervisor 启动同一 run，安全锁锁在 user-owned workdir（可以 `flock`）。
- 初始预算建议（需要用户确认）：≤1 GPU，同夜单项 ≤2 GPU-hours，总夜间 ≤8 GPU-hours；资源空闲而且用户在批准文件授权后可提高到 8 GPU（总 ≤相应 GPU-hours），但**GPU idle 不等于已获许可**。每项设置最大 wall time 和磁盘预算，例如 20 GiB/批次；达到阈值停止新的任务，保留结果。

### 智能重试

- 纯网络/临时 GPU 占用：等待与重查，无危险动作；GPU 空闲等待每 30–60 s，稳定 ≥120 s，尝试等待 90 分钟后降级本地分析，下一窗口再查。
- 进程随机 transient 失败：保存错误类型，最大 **2 次**可审计的同配置重试，重试前核查旧进程不存在；不能在同目录重叠写数据；新尝试编号唯一。CUDA Graph capture、显存 OOM、质量错误、binary/ABI 不匹配、Runner 明确的 integrity failure 属**非瞬时**，优先根因修复，不能重复消耗 GPU。
- 代码修复：本地 Git 新 branch/commit/测试，向 H800 仅在我方新 worktree 部署；需改 C++ 时在 H800 独立 build 更新并记录 binary SHA；所有改变了对照内核/配置的性能结果须两臂同版本重测，原结果不可拼接。
- 结果失败：源 SHA 不对/请求不齐/Graph replay 缺失/输出质量异常/服务器热问题，标记 INVALID/INCOMPLETE，不得自动转 success。

### 每天可供接班的机器可读产物

`state/STATE.json` (当前阶段/active run/资源/最终证据时间); `state/EVENTS.jsonl`（append-only 时间、动作、退出码、证据相对路径）；`state/reconciliation.json`（远端真实与本地计划对照）；`results/INDEX.csv`（每个 manifest 唯一一行）；`plans/NEXT_ACTIONS.md`（排序动作/科学价值/风险/预算）；`reports/`（持续报告）。重启后把这些读一遍；发现当前 `RUNNING` 先恢复监测，不重新启动。

## v2：本地控制器实现的落点及授权执行

控制器新增代码放 `scripts/controller/`，小型定向支持脚本放 `scripts/`，实验案列放 `plans/experiments/`，新的 Runner 放 `scripts/runners/` 并以独立版本号标记。Agent 一次从可信（优先本地用户只读的 `~/.config/kv-research/approval.json`）读取批准范围，不得自行创建或扩增许可；控制器在**每次远端启动前自动**核对 ssh 身份、远端目录归属、所需 UUID 属于白名单、全部稳定空闲且未被占用、待用 GPU 数不超过并行上限、每项及总 GPU-hours/磁盘预算、run_id 唯一性。满足时自主安排，无须人类在场。失败只暂停**相关 GPU 作业**。当前 `experiment_worker_stub.py` 未实现这些约束，必须实现可测试调度器才开放 S2 无人值守。

详情见 `ops/PATHS_AND_FILE_PLACEMENT.md`、`ops/APPROVAL_AND_OPERATOR_GUIDE.md`。

## 2026-10-27：退出中的R状态、kernel flag与跨Python版本审计

BF16-XQA a在真实operator全部通过后，PID2594157的原本方UID/start/boot不变但cgroup根/stateR触发CONFLICT；driver已释放，无服务测量。旧事件没有kernel flags，原因保持unverified，不能倒推成功。H800内核5.10；官方do_exit在cgroup_exit之前设置PF_EXITING=0x4，stat可能仍R。本地24个自有CPU退出均观察到R+PF_EXITING，不等于旧H800故障证明。新保护仅对此前本方同PID/UID/start/boot且Z/消失或内核PF_EXITING正证据执行最多3次driver再读/1秒，必须确认真消失并继续核查其他PID；普通R/S迁移、未知/复用/其他UID、driver仍报告均拒绝。150CPU验收通过；新run再验，旧run不重启。完整说明production-bf16-xqa-operator-20261027a/REPORT.md含官方源码链接。

首个toolagent独立audit在本地Python3.10严格比较失败：只涉及31个mean字段的约1e-16至1.5e-10舍入差，其余每请求/clock/count/分位/SHA一致。用与远端相同Python3.12重新执行既有审计后，全部对象逐值严格相等通过；未放宽比较、不修改原测量/旧失败记录。后续trace独立审计使用/home/mingasa/venvs/vllm/bin/python -B，并保存解释器版本。Python3.10失败c1保留，正式c2_py312。

Toolagent HTTP200失败的raw不是JSON SSE error：packets.name为服务端并发上限纯文本，value=null，AIPerf报InvalidInferenceResultError。错误全部保留，不能按HTTP200算成功，也不要把它泛化为所有失败都是结构化SSE error。当前并发上限为200，不据此声称FP8值损坏或将原失败删掉。

## 完整成功trace的大raw只读取证

若已封存raw实际超过128MiB，先S0确认size/归属/对应worker seal。可显式用mirror_readonly --max-file-mib256，仅精确我方run的profile_export_raw.jsonl且ticket/request/成功measurement及raw SHA匹配；不是任意文件增额。全部传输/核验分块，单批总512MiB、XZ解压内存128MiB和展开tar520MiB保持。先CPU再实际取证；新midrate89文件/395MB已通过。保留失败partials，不能因镜像界重跑GPU、删raw或把未验收标成功。详见readonly_mirror_256_actual_H800_acceptance_v1.json和故障库。
