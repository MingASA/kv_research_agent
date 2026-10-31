# 最新证据冻结表（以此代替旧交接结尾）

本页快照截止 2026-10-25。本会话用户直接贴了 H800 最后一次构建核验输出；**没有贴出** `ceff00bb` 版本的 AIPerf 结果或其 tmux 的最终状态。过去助手曾给过运行和过夜脚本命令，但不能因此推断用户真的执行了。接管后务必只读发现。

| 事件/结论 | 状态 | 依据/限制 |
|---|---|---|
| Qwen2.5-14B 旧 Joint 四臂完整质量实验 | **历史结果已报告** | 8 卡独立分片 1719×4 合并；GSM8K strict auto .6892, INT8 .6884, FP8 .6505, INT4 .7779；LongBench macro auto .4947, INT8 .4983, FP8 .4890, INT4 .4409。各实现/scale 不一定等于新 native FP8，不可移植结论。需要查原 batch manifest 与评分归档；旧 2026-10-20 文档的“尚未完成”已过期。 |
| 7B mixed-V 质量历史 | **已报告** | V-only FP8/INT8 多任务近 BF16、K-only FP8 常严重损害质量；旧 Triton/flashinfer_mixed 结果，不能直接归因 native Joint 性能。历史源与输出需只读定位。 |
| 7B / 14B 旧 ShareGPT/短 AIPerf | **历史结果已报告** | 存在旧 v2.1 四臂结果和报告；例如旧 Joint BF16 2240.8 tok/s, FP8 2047.3, INT8 1959.2, INT4 673.9；这是不同 Attention backend 的比较，非新研究主对照。 |
| `72cb5810` native FlashInfer BF16/FP8 Eager 14B | **确认跑通** | 三条 SSE 中两题双臂都对，代码题两臂同答 25（正确为 34）；原检查因绝对正确率报错，不能认定 FP8 退化。真实 native FA2 和 resident KV/alias 证据。 |
| FP8 驻留容量 | **确认** | BF16 4096 B/token/layer vs FP8 2112 B；BF16 1477 pages/layer vs FP8 2865；allocator 两臂预留显存均约 34.6GiB，不能以总预算相同否定压缩。固定校准 K=.125,V=.0625（不能证明全场景质量）。 |
| v4.1 旧 binary 16K512 C4 BF16 | **确认成功** | Worker rc=0，128 profiling + 8 warmup。旧二进制上的 BF16 Graph 结果可保留，不能与 ceff FP8 拼成正式对照。 |
| v4.1 旧 binary FP8 Graph | **确认失败** | C++ eager-only CHECK 阻止；Python 捕获/重放未进行。v4.1 0MiB/88% GPU util 切臂拒绝后 `resume` 进入 FP8，发现该问题。 |
| `ceff00bb` C++ admission 修复 | **本机测试 + H800 构建确认** | GitHub commit `ceff00bbd38054d2b6b4aa44513d773f2bd174b2`，CPU 7/7；H800 Ninja no work，修复字符串存在，`ldd` 无 not-found，新 binary SHA256 为 `277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3`。 |
| v4.2 Runner | **H800 部署/CPU 检查通过** | test_v4_2_native.py 8/8；连续 5 秒空闲等待，记录 commit、binary hash、capture/replay。还不能当 Graph 真实执行过。 |
| ceff 新 binary BF16/FP8 16K512 C4 Graph 配对 | **状态未核实，不可假定完成或未启动** | 前会话提供启动命令 / tmux session 建议，但本轮没有远端结果。第一次只读检查 `native_flashinfer_14b_16k512_C4_graph_ceff00bb_g2_001`、session `cjyu-v42-ceff-g2`、runner `COMPLETE` 等。 |
| “过夜 8K/16K/24K Synthetic 队列” | **仅有脚本提案，没有启动成功证据** | 先前建议过 `cjyu-native-night`、`run_native_overnight_ceff00bb.sh`；其固定 `COMPLETE` 判断/多 concurrency 命令未得到实机验证。不得只凭建议或日志名字视为既有结果，也不要盲目再建同名队列。 |
| 新 ceff native FP8 Graph 真正 Capture/Replay | **未从会话证实** | 必须看两臂真实服务日志与 `NATIVE_KV_GRAPH_REPLAY`，不能只看 binary、开关或预检。 |
| 新 native fixed scale 的 GSM8K/LongBench 质量 | **未确认做过** | 若 H800 最新文件有实际证据则更新；否则需设计同 binary/score/scale 配对。 |

### 本次最新构建经验（已完成，旧文档已过期）

最初独立 build 缺 Cargo/Rust；随后 `yalantinglibs_DIR` 未带上；再因 `--network none` Cargo 访问 crates.io 失败，使用 `CARGO_NET_OFFLINE=true` 复用缓存；全量 third-party cpprestsdk 测试缺 Boost.Locale，最终只针对 xllm 主目标构建并已成功。**不要为了用新 binary 做实验而重复追第三方测试失败。** 新源码与新 binary 都已到位。

### 恢复首要动作

1. SSH 只读查看 `tmux ls`、我方 output paths、`COMPLETE`、Runner `status`、原始日志、Docker containers 和 GPU 实际状态；将所有 findings 与准确时间记录到本地状态。
2. 优先验证已有 ceff AIPerf 的失败/完整性，若任务仍跑则观察，若已完成则直接分析，若未开始则按明确预算授权排入；一切以当前产物为准。
3. 若缺失 GPU 权限/SSH 位置，继续本地科学/代码/结果分析，不修改远端。
