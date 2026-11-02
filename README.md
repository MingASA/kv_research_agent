# xLLM / KV Cache 量化科研：5060 常驻 Codex 接手包

**最新研究（2026-11-02）**：[最新版官方 xLLM 原生 KV 量化 A/C 生产对照](reports/experiments/upstream-production-kv-20261102/REPORT.md)。固定官方`261274e5`、BF16 A=`721afdab`、实测C=`da7c6446`；交付`a5ce9684`只补三个Markdown文档。三格式九场真实服务、1720条核心质量及32原生工具任务均已完成独立核验，所有负结果与限制保留，GPU作业与监控已结清。下方2026-10-30等内容是历史D研究。

**新一代模型集中报告**：[Qwen3.8-27B BF16/原生 FP8/NVFP4 工程与生产结果](reports/synthesis/NEXTGEN_NATIVE_KV_20261030.md)。本轮实现/实机研究、独立审计、归档与资源收尾已完成；[九分区证据目录](results/archive/nextgen-native-kv-20261030/CATALOG-a1.md)和[最终26项范围审计](reports/operations/session-recovery-20261030-01a1765d/COMPLETION_REQUIREMENTS-c1.json)集中列证，默认生产接受与未知边界单列。

**2026-10-30 会话恢复入口**：原会话 `01a1765d-9953-7760-a9fa-102c05c766d7` 的37条用户消息、截断前动作、新远端对账、恢复报告和提示词已归档至 [恢复报告](reports/operations/session-recovery-20261030-01a1765d/RECOVERY_REPORT.md)。原FP8补集ab已完成并独立合成1720条质量评分；后续两配置三格式研究已全部审计，当前无待续GPU作业。实时作业状态先读 `state/STATE.json` 和 `state/NEXTGEN_ACTIVE_RECOVERY.md`，不要根据本页旧快照重复启动。

**2026-10-26 最新用户指令优先**：当前目标是生产 go/no-go，先读 `ops/PRODUCTION_DECISION_STANDARD.md`。用户已明确持续按需授权并委托记录，执行器已有CPU及真实隔离/停止验证。强制512合成任务已按用户生产范围停止，不作有效配对；下一任务为真实ShareGPT上下文/自然停止、现代FA3候选的服务评估。以下“缺授权/未接通”及手动启动步骤属于历史说明。**2026-10-27恢复规则更新**：上下文压缩后查看目录/导航、理解内容范围、恢复状态和当前任务，按需读取相关文档，不再全文重读；远端先只读对账，续接手头工作。

**快照：2026-10-25；按本会话最新可见证据整理。** 工作方向不是修复某一次编译，而是系统性的、生产相关的 KV Cache 精度—质量—容量—吞吐—延迟研究。

**接管后状态（2026-10-26 UTC / 本地 10-25）：** 首次只读巡检与恢复已完成。六个 ceff run、九个配对负载已存在并独立审计；没有重跑。当前入口是 `state/STATE.json`、`plans/NEXT_ACTIONS.md` 和 `reports/experiments/ceff-takeover-20261026T040426Z-v2/REPORT.md`。下文与 STATUS_AS_OF 是历史启动说明；新实机证据优先。可信许可仍缺，完整预算执行器尚未接通，不能启动 GPU。

## 立即使用

1. 在 **5060Ti 的 WSL** 将本包解压到**新的独立工作目录**，不要直接解压覆盖 Git 仓库。打开 `MASTER_PROMPT.md`，在这个工作目录运行 Codex；将该文作为首条指令；在这一级设置 `AGENTS.md`，使重启后自动遵守。
2. 第一次必须 `read-only` 发现本机与远端 SSH、服务器路径、所有运行中的任务、现有实验与结果。**不启动、不暂停任何 GPU 工作负载**；先写 `state/REMOTE_DISCOVERY.md`、`state/reconciliation.json`。
3. 验证可安全恢复后按 `policy/REMOTE_SAFETY.md`、`plans/RESEARCH_ROADMAP.md` 推进。使用 `state/approval.json` 指定有限的实验许可/预算；没有此文件有效许可时，仅能准备本地代码、CPU 分析、远端只读巡检。
4. `scripts/readonly_discovery.sh`、`scripts/summarize_state.py`、`scripts/guarded_ssh.py` 是可用脚手架；`scripts/experiment_worker_stub.py` 仅示意调度模型，不会执行 GPU 工作；核心 runner `vendor/aiperf_h800_v4_2_native_release.zip` 是历史交付，运行前应从 H800 的实际脚本和 README 再核对。

## 文件导航

- `MASTER_PROMPT.md`: 无背景 Codex 的主指令，职责、工作闭环、可持续行动原则。
- `AGENTS.md`: 每次会话生效的强制性安全与证据守则。
- `science/RESEARCH_THESIS.md`: 终极研究问题、假设与物理机制、技术路线。
- `science/ARCHITECTURE_AND_PROVENANCE.md`: 仓库、提交、H800/5060 环境、后台方式、运行时证据。
- `plans/RESEARCH_ROADMAP.md`: 完整实验组合及优先级、成本与有效性门槛。
- `state/STATUS_AS_OF_20261025.md`: 已验证、仅报告、未验证的最新状态和恢复优先级。
- `policy/REMOTE_SAFETY.md`: 高权限 SSH 环境的最小权限与 GPU 资源纪律。
- `ops/OPERATIONS_RUNBOOK.md`: 任务发现、停止条件、重试策略、构建/部署/实验/恢复。
- `ops/RESULTS_AND_METRICS.md`: 源码与机器可读指标契约；实验质量门槛。
- `knowledge/H800_xLLM_故障速查_完整追加版.md`: 原始历史保留，追加截至本轮的构建/runner 问题，**不要用旧状态覆盖新状态**。
- `templates/`: 最小配置、任务/实验 manifest、事件日志与审批文件样板。
- `scripts/`: 安全只读探查、SSH 重试、状态汇总等脚手架。
- `vendor/`: 之前交付的 v4.2 Runner ZIP + 历史阅读材料，仅供审计/复用；原版尚未修改。

## 关键声明

- 用户计划用 `codex --yolo`；**CLI 的自动执行权限不等于对 H800 的授权**。服务器是多人共享且账户权限高；自主权按文档中的白名单范围行使。先验证 SSH 与归属，永不凭猜测执行 destructive commands。
- 这个包不会自动安装 Codex、配置 SSH 或在远端启动服务。`ssh_target`、可用 GPU、使用时长和真实作业均须启动后发现。此包中的 HOST/路径快照是历史证据，不是默认写入权限。
- 如果最初的 16K C4 或后来“过夜队列”已在用户睡前真实启动，必须先恢复与检查；**不能因为交接状态未知就再启动一套**。

## v2（2026-10-25）重要补充：路径和授权 FAQ

如果想知道**本地新代码、新脚本、新实验放哪**，务必先读 `ops/PATHS_AND_FILE_PLACEMENT.md`。如果想知道**用户首次启动要做什么、GPU 需不需要每次人工授权**，先读 `ops/APPROVAL_AND_OPERATOR_GUIDE.md`。结论：用户对确认的机器/路径/白名单 GPU 和周期预算**一次性授权**，Agent 在授权范围内自主启动 1–8 张空闲卡；每次启动 GPU 的空闲/占用检查由 Agent 自动执行，不需要用户逐项批准。越权、到期、超预算或系统级高风险操作才需另行批准。`scripts/experiment_worker_stub.py` 不能被当作已完成的技术强制器，GPU Controller 必须先实现并测试。
