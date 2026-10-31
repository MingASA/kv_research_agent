# 接管记录

控制端日期 2026-10-25；UTC 日期已为 2026-10-26。以下时间均明确使用 UTC。

## 阅读与长期约束

完整阅读 `MASTER_PROMPT.md`、`AGENTS.md`、REMOTE_SAFETY、10-25 快照、RESEARCH_THESIS、ARCHITECTURE_AND_PROVENANCE、ROADMAP/NEXT_ACTIONS、OPERATIONS_RUNBOOK、PATHS_AND_FILE_PLACEMENT、APPROVAL_AND_OPERATOR_GUIDE、RESULTS_AND_METRICS、TRANSITION_PLAN、完整追加故障库、全部 vendor 历史文本与 ZIP 内 README/任务/runner/tests。
本地源码根 AGENTS 与 custom-code-style 也已读；重点读取 ceff admission、resident allocation、backend/Graph 日志实现和请求 EOS/温度处理。
旧 vendor 的“手动一条命令/没有 SSH 权限”等历史工作方式服从本轮明确接管指令与当前安全政策，不移植旧状态。

`handover_package_audit.json` 记录原包 34 个文件的 SHA 对照；核对时 28 个未变，6 个动态 state/plan/index 因此次接管更新而改变。
PACKAGE_MANIFEST 作为历史档案保留，不修改它来伪装未变。后续故障库的新增另有事件记录。

## 对账结果

04:01 起本地只读发现：控制目录原本不是 Git；源码 `/home/mingasa/projects/xllm_quant_kvcache` 为 ceff HEAD，已有未跟踪文件与子模块修改不动。
本机实为 RTX 5060 Ti，16311 MiB；采样时已有 1009 MiB/7% 利用率，不以此机模拟 H800 实测。

04:02–04:04 SSH 只读：发现 H800 alias 后以 known_hosts 中 ED25519 pin 严格认证，UID/GID 1005、hostname、BASE realpath/stat/mount 与历史匹配。
未发现项目 runner/queue 进程，默认 tmux server 不存在；其他长期容器保留。
六个 ceff run 已完成，含初始 16K/C4 及后续夜间矩阵，合计九条件。另有两个旧 v4.1 不完整 manifest，一个有明确 FP8 failed state；不自动恢复旧版本。

04:29 再次只读：八卡瞬时 0 MiB/0%，无 compute apps，无 `cjyu.aiperf.run` 活跃容器、无 project cwd 的进程、默认 tmux 仍不存在。
此事实不等于全机预约/其他 socket 队列已验证；没有宣布获得资源排他性。

## 执行、故障、修复与验证

只读镜像 372 个选择文件；首次本地尝试因远端 Python 3.9 tar stream 参数不兼容失败，保留 transport.json 与 partial archive。
修复采集器、加入流式 roundtrip 与安全边界测试后，新 attempt02 成功。整个传输只在远端内存/SSH stdout 形成 tar，未建立远端文件。

独立 verifier 不 import 历史 runner，不调用其会改写文件的 verify。
已逐 case 核查真实 capture/replay、resident bytes、128 profiling + 8 warmup、ID/session/input length 配对、server usage、summary 均值与 P50/P90/P95/P99、tokens、吞吐时间口径、源码/输入/脚本/binary SHA。
最初 throughput 时间口径与 AIPerf observation window 有小差异，读取已安装源代码后修正并增加回归测试。最终六 run / 18 case 全部测量完整性通过，2304 条 profiling + 144 条 warmup，602269 actual output tokens。

旧 joint14b shard manifest、summary、CSV 的 raw-byte hashes 与 complete 一致；没有重评分全部旧预测。新 native 正式质量仍未验证。

本地新增 31 个 CPU fixture tests 通过，覆盖证据错误拒绝、archive 路径/软链接/hash、Graph 未 replay、配额/许可/UNKNOWN/去重等。
真实批准源预检返回预期 rc=2、can_launch=false：可信许可不存在，完整 runtime budget executor 未实现。

## 研究结论与下一步

详见 `reports/experiments/ceff-takeover-20261026T040426Z/REPORT.md`。
confirmed：resident bytes/token/layer 从 4096 降至 2112，当前分配容量约 1.93946×；Graph 两臂真实执行。
confirmed：当前性能仅是自然停止/可变 OSL 的描述性服务测量；输出 tok/s 变化 −5.90%～+6.12%，没有普遍加速或同质量结论。
unverified：native fixed-scale 质量、满载 admission/HBM 瓶颈与 run-level 可重复性。
proposal：固定 decode 长度对照、原生 paired quality、随后 app-local 机制分解，均有独立实验设计。

安全硬边界未改变：本会话远端写入 0、GPU launch 0。不会代填启用真实许可。审批范围预检有测试，完整 executor 的验收契约在 `plans/controller/EXECUTION_CONTRACT.md`；未完成前不会开放 S2。
所有变更为控制目录的本地工具、状态、研究计划/报告；xLLM 业务源码、历史远端结果、模型、输入、依赖、容器均未修改。

## 补充验收

最新派生报告为 `reports/experiments/ceff-takeover-20261026T040426Z-v2/REPORT.md`，保留 v1。
v2 加入按 raw profiling timestamps 筛选的 NVML 样本（排除 warmup），各 case 采样时间覆盖至少约 99%、最大间隔 2 秒；GPU memory 是采样峰值，不是连续峰值或 KV payload。
本地测试随后增加 profiling-window 及批准文件父目录所有权案例，最终 **33/33 CPU tests** 通过。源码/模型在 GPU 上没有重跑。
控制目录已初始化独立本地 Git branch `research/takeover-20261025`，不配置 remote；授权文件、机器私有映射、原始镜像、Zone.Identifier 不纳入版本控制。Git 未配置人类作者；本地提交使用明确的 `KV Research Agent <kv-research-agent@localhost>` 代理身份，不改系统或全局 Git 身份。
