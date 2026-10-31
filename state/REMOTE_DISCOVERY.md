# 首次只读发现（本地日期 2026-10-25）

观测窗口：2026-10-26 04:01–04:04 UTC。原始结构化证据：
`reports/operations/20261026T040426Z/remote_runs_readonly.jsonl`。
交接快照保留在 `STATUS_AS_OF_20261025.md`；本页记录新的实机证据。

- **confirmed**：控制目录 `/home/mingasa/kv-research-agent`，WSL2，UID/GID 1000，当前不是 Git 仓库；本地文件系统可用约 869 GiB。
- **confirmed**：源码 `/home/mingasa/projects/xllm_quant_kvcache`，HEAD `ceff00bbd38054d2b6b4aa44513d773f2bd174b2`，分支 `bugfix/native-fp8-graph-admission`；已有子模块修改和未跟踪结果，均保留。
- **confirmed**：SSH 配置实际存在 `H800`，解析为 `h800-41594a.fkpku.cn:45291`、用户 `yinjinrun`。连接使用 `StrictHostKeyChecking=yes`、ED25519、`UpdateHostKeys=no`；匹配本机既存 known_hosts 指纹 `SHA256:QZwhu4BzgiOmHYABTUK1kXdAYHPun3iNioo69CKCgEI`。这说明与既存 pin 一致，独立的用户签发身份许可仍缺失。
- **confirmed**：远端 hostname `A03-R40-I18-195-B000010.JD.LOCAL`，HOME `/home/yinjinrun`，UID/GID 1005，具有 docker 组；不据此扩展权限。
- **confirmed**：历史 BASE `/export/home/yinjinrun/.borrow/chenjunyu/xllm-h800` 实际存在、非软链接，UID/GID 1005，mode 0755；其 chenjunyu 父目录 mode 0700。挂载为 `/export/home` 的 XFS，约 1.7T 可用。文件系统所有权不能替代用户对团队空间的授权。
- **confirmed**：远端 ceff worktree HEAD 正确、tracked status 干净；binary 大小 562304624 bytes，SHA256 `277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3`。没有重编译。
- **confirmed**：v4.2 runner SHA256 `594ea36c403fbe0fe51d0777c114f5ddd690a8605099ba97c5dc774ed2c5037f`。其 `verify` 会改写 CSV/JSON，首次巡检没有调用它；直接读取 manifest/state/原始文件。
- **confirmed**：未发现项目 runner/queue 进程；默认 tmux socket 没有 server。存在其他长期容器，未对任何容器进行启停。
- **confirmed**：六个 ceff run 根目录及各 arm 有 COMPLETE、state 记载成功，合计九个负载条件、18 个 arm case。夜间日志有 `ALL OVERNIGHT EXPERIMENTS COMPLETE`。原始记录独立审计进行中，暂不升级为科学有效性通过。
- **confirmed**：两个旧 v4.1 manifest 不完整；其中一个 FP8 state 为 failed。它们不是待自动重启的 ceff 配对。
- **confirmed**：旧 joint14b merged-global 有 complete.json、summary、paired 目录；内容和 hash 仍需独立核验。
- **confirmed**：`agent_work` 尚不存在；`~/.config/kv-research/approval.json`、`state/approval.json` 均不存在。此次远端只读，未加载模型、未创建目录、未启动 GPU。

首次 GPU 快照（2026-10-26 04:03:16 UTC）：8×H800，各 81559 MiB；所有卡 memory.used=0、utilization=0，无 compute-app。驱动 535.216.03。以下 UUID 是设备发现，**不是授权白名单**，也不是稳定空闲或预约检查通过：

| index | GPU UUID |
|---|---|
| 0 | GPU-0c628a64-5f31-3501-fa3c-723f5b69254a |
| 1 | GPU-47374a5f-689b-fa8c-49c4-4aa6840ebfb7 |
| 2 | GPU-a6169058-34c8-6edc-e4f5-779bf9a10421 |
| 3 | GPU-0cb28538-7e9f-1b66-750f-1b53ff309195 |
| 4 | GPU-31f08a96-1b44-a19f-2e1c-7b82ef5866e1 |
| 5 | GPU-7da6f763-4e39-dbf0-2b98-a5f7fa2c7348 |
| 6 | GPU-b6136eb1-9dd4-93d3-40fe-7594e5c996c7 |
| 7 | GPU-47211491-ba0d-78bf-5c84-f968c0648213 |

**unverified**：共享预约、其他 socket 中的排队任务、现存长期容器的精确所有权、新 native fixed-scale 正式质量、原始实测记录与全部 Graph 证据。不会据空闲快照启动作业。

后续更新在本页追加观测时间；机器映射另存不公开的 `CONFIG.local.json`。没有 OS 级或调度器配额隔离的证据。

## 后续验收与 04:29 UTC 复查

六个 ceff run 的独立 raw audit 已完成：18 case，2304 profiling + 144 warmup；Graph、allocation、token counts、请求/session 配对、源 SHA、均值和分位数通过。最新报告为 `reports/experiments/ceff-takeover-20261026T040426Z-v2/REPORT.md`，含排除 warmup 的 NVML 抽样统计；这些不是固定 decoded steps 或同质量加速证明。

04:29 UTC 再次实机巡检八卡瞬时 0 MiB/0%、无 compute apps、无带 `cjyu.aiperf.run` 标签的活跃容器、无 project cwd 进程、无默认 tmux server。旧 quality 的 shard_manifest/summary/CSV raw bytes SHA 与 complete 一致，但预测未重新评分。详见 `reports/operations/20261026T040426Z/final_readonly.json`。

已读取的实际 AIPerf inputs_json loader 将 payload 交给 `Turn.raw_payload`；只证明源码数据流，新的 ignore_eos 控制方案仍须 CPU model/transport 与真实 usage 验收。
本地控制目录后来建立独立 Git branch `research/takeover-20261025`（无远端）；源码业务 checkout 保持原状态。本地 33 个 CPU fixture tests 通过，可信授权预检按缺文件拒绝，完整 GPU 预算执行器仍未接通。
