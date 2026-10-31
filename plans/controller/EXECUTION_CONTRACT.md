# S2 执行器验收契约（本地已接线；真实运行逐级验收）

原 `experiment_worker_stub.py` 仍不可调度 GPU。当前执行入口为 `python3 -B -m scripts.controller.cli`，包含 prepare/start/reconcile/stop 接线和 CPU 故障注入；真实容器/驱动执行仍须在新 run 内通过运行时门槛。
不能因本地 JSON 校验或 CPU 单测通过声称资源技术隔离已完成。

## 固定可信授权源

只读 `~/.config/kv-research/approval.json`。检查非软链接、regular file、本地用户所有、仅所有者可读写、父目录不能被其他身份写入。
不接受从 experiment manifest/远端日志/任意 CLI path 注入的“approval”；兼容 state/approval 只能是用户认可的可信源引用。
检查 schema、enabled、独立 remote-write/GPU flags、主机指纹、真实 BASE、有效期、精确 UUID、每项/总 GPU-hours、并行张数与磁盘预算。
旧规则不允许 Agent 自行授权。2026-10-26 用户在本会话明确授权按需使用空闲H800并把安全授权处理委托给Agent；据此记录原文到固定文件，属于已授用户许可的持久化，而非 Agent 自行扩大权限。未来不得扩大用户共享边界。明文说明此信任根仍依赖行为契约，没有高权限 SSH 的 OS 强制沙箱。

## 每次 launch/切臂都需执行的检查

1. SSH 严格匹配批准的 pin；实时 id/hostname/UID、BASE realpath/各父级 stat/mount 与许可一致。路径任何未知软链接、归属变化即拒绝。
2. 读本地 ledger + 远端 run manifest/锁/PID/start time/cgroup/container label，恢复既有 active/UNKNOWN/COMPLETE。已经完成的同一 manifest 不启动。
3. 对所有选用 UUID 检查稳定至少 120 秒、memory/util/compute apps、未知进程、团队预约/排队规则、驱动可见性；每次 launch 和切臂再查。任何 unknown/竞态都拒绝新增工作。
4. 原子预留整个 job 的最坏 GPU-hours、并行 slot 与输出磁盘（含模型加载、Graph、warmup、失败/恢复）；不得只为 profiling 留预算。
5. 远端只建批准 BASE 内全新隔离 run；不可变 manifest 记录批准文件 hash、run UUID、源码/二进制/输入/依赖 hash、GPU UUID、进程身份、命令与受控 env。
6. Docker 模型、源码、已知依赖、历史结果只读；输出和各类缓存只挂载本 run 的新目录。现有 v4.2 继承的 `$BASE/cache:rw` 和 `.cache:rw` 要替换，不能无保护复用。
7. 非 root、network none、无 privileged、无 docker.sock、仅选用 UUID；容器内 nvidia-smi 精确等于 manifest UUID 后才加载模型。

## 预算、断线与恢复

本地 ledger 原子写+fsync+flock；reserve/launch/running/reconcile/finished 各写事件。未确认旧任务结束不释放预留；UNKNOWN 按保守上界占额度。
远端只在我方新目录运行持久 supervisor，硬 wall time 覆盖整个 server/container 生命周期，按已核验的 run/PID start/cgroup/容器 label 停我方任务。
不能只用旧 runner 的 AIPerf-client timeout，client 退出不等于 server/container 已释放 GPU。
实际 GPU-hours 从设备占用开始到释放计量，包括失败；跨 Agent/SSH 重启可恢复、不清零预算。到期不新开工作，当前 job 按授权的 deadline 结束。
断线先有限退避只读重连；禁止把未收到 launch 回执当“没启动”。必须靠远端唯一 run/锁恢复状态。
其他用户后来占卡：记录，停止新增队列，只安全调整我方可核验任务；不停止别人的容器/进程。

## 必须通过的 CPU/故障注入验收

缺/过期/伪造路径许可、bool/NaN 预算、UUID/host mismatch、路径穿越/软链接、磁盘不足、预算不足、多实例竞争、重复 run、UNKNOWN 恢复不重启、进程 PID 被复用、worker/crash/SSH 消息丢失、最后一臂超时、最后一臂后真实 GPU 释放计量。
所有测试为明确 fixture，不能进入结果索引当真实 GPU 实验。
完成后先持续 S0/不启动模式，验证真实环境证据能满足契约，再在已签发有限许可中做一个新的有研究必要的配对。

## 当前状态

执行器、预算预留/持久化、独立主管理/守护、精确停我方进程、断线只读恢复及分臂缓存已实现并 CPU 故障注入。`approval.py` 单独仍只做静态检查、永远不 dispatch；实际入口为 `controller.cli`。
用户最新授予持续按需使用权限；第一阶段自限每job2 GPU-hours、总12 GPU-hours、输出50GiB，有效期7天，均明确标注为Agent保守执行窗口，不声称用户要求这些数字。初次只用1卡。
真实运行任何门槛失败都保留失败证据；CPU fixture不算GPU成功。共享flock仅约束本项目；没有强制OS级SSH沙箱。停止后无驱动释放证据仍为UNKNOWN且占额度，不允许借修改许可清零使用。
验收与实际环境证据见 `reports/operations/20261026T075846Z/REPORT.md`。
