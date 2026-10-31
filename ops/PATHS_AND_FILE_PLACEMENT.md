# 必读：5060 本地与 H800 远端的目录、文件归属及新文件放置规则

这是**文件放在哪里**的权威约定。下文将证据分成 **已知历史路径（必须 SSH 复查）** 与 **推荐新建路径（尚不存在，只有在确认归属/授权后创建）**。不要猜测 SSH 别名、主机名、Windows 用户名、真实本地源码仓库目录，也不要因为本文出现绝对路径就认为可写。

## 1. 本地：5060 Ti WSL（推荐）

控制工作区（**推荐新建，不声称已存在**）：`~/kv-research-agent/`，即本交接包解压后的根目录。**与主源码仓库分离**。用户应从此目录运行 `codex --yolo`；Codex 不应把交接 ZIP 直接解压覆盖旧仓库。

历史本地源码仓库：用户在 5060 的 WSL 中使用 `MingASA/xllm_quant_kvcache`；可能在 `~/projects/xllm_quant_kvcache`，**未由本次交接确认**。第一次只读发现后，将 `local_repo` 写入 `state/REMOTE_DISCOVERY.md` 与机器配置，随后只在这一 Git checkout 的**新工作分支**改真正的 xLLM C++/Python 源码。曾经从 5060 导出提交 `ceff00bb` 的 Bundle 在 `/mnt/e/AI/xllm-storage/reports/native-fp8-admission-20261022/native-fp8-graph-admission.bundle`；这是**历史产物定位证据**，不是永久默认地址。

| 本地文件类型 | 唯一推荐落点 | 具体规则 |
|---|---|---|
| Codex 长期生效的行为规则 | `~/kv-research-agent/AGENTS.md`, `MASTER_PROMPT.md` | 维持根目录；遇到新安全经验先更新 `knowledge`，必要时修改 rules 并记录原因 |
| 科研问题、物理假设、模型架构 | `science/` | `RESEARCH_THESIS.md` 提出总目标；新增研究点可建 `science/hypotheses/<slug>.md` |
| 实验设计、优先级与选点 | `plans/`；新实验 `plans/experiments/<experiment-id>.md` | 明确对照变量、成本、预期收益、质量门槛，不写伪结果 |
| 机器配置/身份/远端映射 | `state/REMOTE_DISCOVERY.md`, `state/reconciliation.json`, `state/CONFIG.local.json`（新建，不入公开仓库） | 首次探查后写实际路径，标记 discovered_at、证据；不得把 SSH 私钥内容写在其中 |
| 用户签发的额度文件 | **建议放工作区外** `~/.config/kv-research/approval.json`（用户创建、最小权限） | 为兼容 v1 规范，`state/approval.json` 可作为指向外部文件的软链接/只读副本，**不得当 Agent 自己生成的授权**；具体见 `ops/APPROVAL_AND_OPERATOR_GUIDE.md` |
| 本地状态、事件、任务对账 | `state/STATE.json`, `state/EVENTS.jsonl`, `state/reconciliation.json` | Agent 自动 append/update；不清空历史，采用原子写入和校验 |
| SSH、排队、恢复逻辑（非 xLLM 业务代码） | `scripts/`，新增 `scripts/controller/`, `scripts/runners/` | 先单测，再接远端；原始 `scripts/experiment_worker_stub.py` **只是占位**，不能直接当调度器 |
| 真正的 xLLM / FlashInfer / KV 源码修改 | **主源码 Git 仓库** `xllm/`, `tools/`, `tests/` | 维护独立 Git branch、commit、patch 和测试。不要把内核实现只放到控制工作区 |
| 结果清单 / 远端结果镜像 | `results/INDEX.csv`, `results/mirrors/<run-id>/` | 大型原始数据默认只留 H800；本地存摘要与 hash；镜像必须记录来源和采集日期 |
| 运行日志与科研报告 | `reports/operations/<date>/`, `reports/experiments/<run-id>/`, `reports/synthesis/` | 日志指向不可变 manifest/远端证据；最终图和 CSV 必须由实测派生 |
| 已交付参考包 | `vendor/` | 视为只读历史版本；修改 Runner 时另存新版 `scripts/runners/...`，不要改原 ZIP 伪装原始文件 |

**推荐本地新增目录**：`science/hypotheses/`, `plans/experiments/`, `scripts/controller/`, `scripts/runners/`, `results/mirrors/`, `reports/operations/`, `reports/experiments/`, `reports/synthesis/`。只在本地创建，不要求这些当前存在。

**不要将本地 Codex 的有状态控制目录直接放在 H800 NFS，也不要把远端原始实验目录整体纳入 Git。** 控制目录可以另建**私有** Git 仓库（忽略审批文件/令牌/秘密/庞大结果）；业务代码提交到 `MingASA/xllm_quant_kvcache` 的单独开发分支。

## 2. H800：真实历史路径（首次通过 SSH 只读验证）

历史路径前缀：

```
BASE=/export/home/yinjinrun/.borrow/chenjunyu/xllm-h800
OLD=$BASE/xllm_quant_kvcache
NATIVE72=$BASE/xllm_quant_kvcache-native-kv-72cb5810
NEW=$BASE/xllm_quant_kvcache-native-fp8-admission-ceff00bb
NEW_BIN=$NEW/build/h800-native-fp8-admission/bin/xllm
OLD_BIN=$OLD/build/lib.linux-x86_64-cpython-312/xllm/xllm
RUNNER=$BASE/inputs/aiperf/aiperf_h800_v4_2_native.py
SYNTH=$BASE/inputs/aiperf/v3_canonical_v32/isl{ISL}_osl{OSL}_n136_seed20261020/canonical_inputs.json
MODEL_HOST=/mnt/cfs/9n-das-admin/llm_models/Qwen2.5-14B-Instruct
MODEL_CONTAINER=/models/Qwen2.5-14B-Instruct
OLD_CMAKE_CACHE=$OLD/build/cmake.linux-x86_64-cpython-312/CMakeCache.txt
```

- **已证实但可能需最新复查的 `NEW_BIN`**：Git commit `ceff00bbd38054d2b6b4aa44513d773f2bd174b2`；binary SHA256 `277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3`，537 MB，Ninja no work；`ldd` 无 `not found`。尚未从本会话证实新二进制真实 Graph replay。
- **已有运行输出**：`$BASE/experiments/aiperf/`，历史旧质量实验 `$BASE/experiments/joint-kv-14b/joint14b-full-quality-v1-8gpu`；已有配对路径可能为 `$BASE/experiments/aiperf/native_flashinfer_14b_16k512_C4_graph_ceff00bb_g2_001`。**必须先检测目录 / COMPLETE / manifest / 运行 PID，不能直接再次启动。**
- **已有 Frozen Input**：`$BASE/inputs/aiperf/`，包含 v3/v4 Runner 和历史 canonical 数据。不得覆盖、重生成、重新抽样已冻结文件。
- **旧已安装依赖**：`/workspace/deps/prefix`, `/workspace/deps/vcpkg`, `/workspace/deps/cargo`, `/workspace/deps/rustup`，这些是历史 **Docker 容器内路径**；宿主机上实际位置要经过挂载检查。`CARGO_NET_OFFLINE=true` 与 `--security-opt seccomp=unconfined` 是已知构建条件。**不触碰全局依赖目录。**

## 3. H800：未来 Codex 的新文件应放在哪里

在验证 `BASE` 的归属及权限之后，推荐为新 Agent 建一个与所有旧结果隔离的远端根：`$BASE/agent_work/`（**当前尚未验证是否存在**），新写入只允许发生在用户明确允许的项目目录内。初次新建采用 `umask 077`，如团队需要共享再明确授权，不自动全盘 chmod。

```
$BASE/agent_work/
├── incoming/            5060 上传的只读核对过的 Git bundle、脚本/清单副本
├── worktrees/<id>/      后续 NEW COMMIT 的独立 Git worktree；不覆盖既有 NEW
├── builds/<id>/         新源码/架构的独立编译产物（也可随各 worktree 建 build）
├── runs/<run-id>/       将来新的实验输出（前提 Runner 允许 --out 在 BASE 下）
├── ledger/              远端持久化任务状态/所有权锁/进程记录
└── scratch/<id>/        有清晰 TTL 的临时中间文件；默认不自动删除
```

**现有 v4.2 路径兼容性**：v4.2 Runner 已硬编码引用 `$BASE/inputs/aiperf` 中的旧脚本/输入，并要求 `--out` 和 `--source` 在 `BASE` 下；**不能直接把 Runner 重定位到 `agent_work/` 就认为功能相同**。默认调用已存在的 `$BASE/inputs/aiperf/aiperf_h800_v4_2_native.py`（不修改），把新的 `--out` 放在经过 Runner `--check-only` 验证的 `$BASE/agent_work/runs/<run-id>`（或继续使用 `$BASE/experiments/aiperf/<run-id>`）；如果必须更新 Runner，创建新版本、保留旧脚本并确认导入路径和 BASE 推导是否不变。

**新文件归属矩阵**：
- **服务/内核代码修改** → 5060 本地 Git 的新提交；H800 只能上传新 Bundle 并创建**新** worktree；需要 C++ 时编译 **新** binary；不在 H800 原仓库直接 hot patch。
- **实验任务** → `plans/experiments/<id>.md`（本地） + 不可变 `manifest.json`（远端对应 `runs/<run-id>/`） + 本地 `results/INDEX.csv` 条目；不得覆盖已有 `run-id`。
- **运行日志/临时编译日志** → 远端新 `runs/<id>` / `builds/<id>`，本地只镜像日志/摘要/哈希；不可隐藏失败码。
- **新 Benchmark 数据** → 首先设计数据、许可与来源；冻结后置入用户空间新 `inputs/` 子目录，记录 SHA；不要偷偷修改旧 canonical 输入。
- **质量/性能研究报告** → 本地 `reports/experiments/` 和 `reports/synthesis/`；报告中链接远端证据、环境 hash 和预期误差。
- **故障速查** → 本地 `knowledge/H800_xLLM_故障速查_完整追加版.md` **追加式维护**，发现新的真实问题记录现象/验证/修复/状态；旧条目仍保留。

任何目录建立前先用 `realpath`, `stat`, `id`, `mount` 判定实际归属与磁盘；不要使用未经核对的软链接作为写权限证明，不跟随可疑 symlink 写到工作区外。用户账户拥有 sudo/Docker 不代表可任意访问共享文件。

## 4. SSH 命令的工作方式（5060 到 H800）

- SSH 别名 `ssh_target` **未知**：Codex 首次从 `~/.ssh/config` 只读查可用条目，并以可信主机指纹核对，不能假设叫 `H800`。
- SSH 连接用于 **远端只读探查/已授权的明确命令**；Agent 自身与 Codex CLI 始终只留在 5060。SSH 断开后应从 `ledger` + tmux/Runner `status` 重新确认远端任务，不盲目重复启动。
- C++ 独立编译/质量评估可在 H800 进行，但源码通过 Git Bundle 从 5060 来；不要求 H800 拉 GitHub/Crates/模型。
- Docker 对 H800 保持既有镜像，历史典型 `--network none`, `--user UID:GID`, `--security-opt seccomp=unconfined`, bind mount 双路径解决 detached Git Worktree，模型和旧目录只读、新 build/results 单独 rw。CUDA 设备限定为使用前实际探查的 UUID，而不是因参数为 `--gpus all` 就拿到整台机器。

## 5. 路径配置更新原则

每次 Agent 启动先读 `state/CONFIG.local.json`（若存在），并核对对照结果 `state/reconciliation.json`；无该文件先仅只读发现。保存 path/source 字段：`local_repo`, `control_root`, `ssh_target`, `remote_identity`, `remote_base`, `remote_model_path`, `remote_runner_path`, `remote_new_binary_path`, `source_commit`, `binary_sha256`, `observed_utc`。这些不是自动获得的授权。

当发现路径变化时不要静默修所有脚本。先确认是哪一机的路径、相应 mount 的映射、Git 版本与数据是否变化。只有证据一致、属我方工作区，才修改机器配置；安全级别由 `policy/REMOTE_SAFETY.md` 和 `ops/APPROVAL_AND_OPERATOR_GUIDE.md` 决定。
