# H800 xLLM 实验故障速查

> 记录已发生的问题；**现象 → 根因 → 解法**。适用于 5060/Codex 开发、H800/Docker 实验的分离环境。

| 问题 | 现象 / 根因 | 解法 |
|---|---|---|
| GitHub 网络 | `HTTP/2 stream ...`、`Empty reply from server`、SSL timeout；H800 到 GitHub HTTPS 不稳定 | Git Fetch 失败时改用 GitHub API/Raw 下载单个文件，必要时从 5060 SCP；校验 Git Blob SHA，别反复 Clone |
| 远程分支不存在 | `invalid reference: origin/feat/...`；Fetch 未完成，本地无引用 | 确认 Fetch 成功再 `worktree add`；临时方案从旧 HEAD 建独立 worktree，再下载新增文件 |
| worktree 容器内 Git 失效 | `fatal: not a git repository: /export/.../.git/worktrees/...`；`.git` 指向宿主机绝对路径 | 同时挂载 `-v "$BASE:/workspace:ro"` 和 `-v "$BASE:$BASE:ro"` |
| 源码版本不一致 | 新 worktree HEAD 仍是 `e1cea470`，新增文件来自 `e36f04f2`，且 Runner 有本地修补 | 记录**基础 HEAD + Codex commit + 文件 SHA256**；正式实验前让 Codex 同步补丁、锁定版本；不要把旧 HEAD 写成新提交 |
| Docker CUDA 初始化失败 | `cudaGetDeviceCount Error 304`；此前容器启动参数漏了已验证的安全设置 | 恢复 `--security-opt seccomp=unconfined` 等原有 Docker 配置；确认容器内 `torch.cuda.is_available()` 和单卡 UUID |
| OpenBLAS 线程失败 | `pthread_create failed for thread ... of 64`；容器线程创建受限 | `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1` |
| Docker UID 无用户名 | `getpwuid(): uid not found: 1005`；`--user` 的 UID 在容器 `/etc/passwd` 中不存在 | 设置 `USER=yinjinrun`、`LOGNAME=yinjinrun`、`HOME=/tmp`；显式配置可写的 Torch/Triton/HF 缓存路径 |
| FlashInfer 找不到 | 首次 Forward 出现 `ModuleNotFoundError: flashinfer`，HTTP `RemoteDisconnected`；C++ 内嵌 Python 没找到 venv 包 | **不要重新安装**；设置 `PYTHONPATH=/workspace/.venv/lib/python3.12/site-packages:/workspace/xllm_quant_kvcache-joint14b`。已有 FlashInfer `0.6.18.post1` |
| Python / Binary 路径 | Runner 用新 worktree，binary 来自旧构建；混用时容易引用错源码或依赖 | 显式 `--python_model_path`（Runner 已设置）、`PYTHONPATH` 和 `--server-binary`；检查实际加载路径。当前未证实存在 ABI 不兼容 |
| GPU 隔离 | 容器需仅访问空闲 H800，不影响其他用户 | 启动前检查显存；`--gpus device=1` + 对应 `/dev/nvidia*`；容器内比对 `EXPECTED_GPU_UUID` |
| LongBench 假重复 | NarrativeQA 50 条完整记录、50 个 ID，但仅 49 个不同 Prompt；Runner 按 Prompt 去重误拒绝 | 保留冻结样本；LongBench 按**完整记录和 ID**查重，重复 Prompt 单独计数；本地补丁须同步 Codex |
| 数据路径未找到 | `find` 按 `*gsm8k*.jsonl` 搜不到实际 `.../gsm8k/data.jsonl`；搜索范围/文件名错误 | 从已有启动脚本读取 `$BASE/inputs/quality/eval-data` 与 `inputs/longbench-scorer`；冻结数据不重建。GSM8K 1319，LongBench 8×50 |
| Dry-run 连锁报错 | 先 `ValueError`，再 `JSONDecodeError`；前者没有输出 JSON，管道下游继续解析；`set -e` 还可能结束 SSH Shell | 先检查容器进程退出码，再解析 JSON；交互式排错避免全局 `set -e` |
| 实验输出目录 | Runner 要求新输出目录；预先创建并放日志可能触发冲突 | 日志放 `$OUT/console.log`，Runner 输出放全新 `$OUT/run/`；失败记录不覆盖 |
| AIPerf 指标 | 曾遇到预热阶段 ITL=0；不能混入正式数据、不能用短 Prompt 推断高负载加速 | 分离 4 条 Warmup / 20 条 Profiling，启用 `--use-server-token-count`，审计 Streaming、Token 一致性与 ITL。7B 四精度 Smoke 已通过 |

| Docker heredoc 无输出 | `docker run` 后 Python `-` 无打印；Docker 默认未传入标准输入 | `docker run -i ... --entrypoint python IMAGE - <<'PY'`，必须加 `-i` |
| TVM FFI 双库冲突 | 14B 首次 Forward 后 `ffi.GetRegisteredTypeKeys is already registered`，`RemoteDisconnected`；xLLM 链接 `/usr/local/lib/libtvm_ffi.so`（26.5 MB），Python venv 自带另一份（2 MB）；CPU 双库加载复现同样崩溃 | 容器内 `export LD_LIBRARY_PATH=/workspace/.venv/lib/python3.12/site-packages/tvm_ffi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}`；`ldd -r`、两库先后导入以及 14B 真实 Smoke 均已通过，不覆盖系统库 |
| HTTP RemoteDisconnected 误判 | API 客户端仅提示连接断开，可能是 worker 动态库崩溃 | 先查对应 `arms/<arm>/attempt-*/server.log` 的末尾和 C++ fatal；不要当成单纯网络故障 |
| 八卡脚本保存路径错 | `[Errno 20] Not a directory`，`$BASE/scripts` 实际是一个 42 KB 文件 | `namei -l` 检查每一级；将文件安全改名为 `$BASE/joint14b_8gpu.py`；修复内部 merge 路径为 `/workspace/joint14b_8gpu.py` |
| LongBench 评分依赖缺失 | 八卡容器全部 `ModuleNotFoundError: jieba`，直接发生于导入官方 `LongBench/metrics.py`，尚未启动 GPU 模型；检查还发现 `fuzzywuzzy`、`rouge`、`nltk` 缺失 | 使用 Python 3.12 容器**独立只读挂载**的依赖目录安装 `jieba/fuzzywuzzy/rouge/nltk` 及递归依赖；在无 GPU 容器中导入官方 metrics 并做必要函数调用，全部通过**后**再改八卡 Docker `PYTHONPATH` 运行，不修改全局镜像和已批准的数据 |
| 正式实验预检缺位 | GSM8K 单题 Smoke 无需 LongBench metrics；`--dry-run` 也不导入它，导致八卡启动后才发现缺包 | **正式运行前必须在生产 Docker 环境无 GPU 导入真实评分模块及检查全套依赖**，不能仅以 dry-run 通过当成运行时就绪 |
| 八卡结果与分片 | `prepare` 八份数据审计通过（GSM8K 1319、LongBench 8×50）；首次 `run` 八卡都因缺包退出 | 不重复分片、不覆盖已审批文件；修好依赖并预检后对相同 batch 重跑 `run`，由独立输出与 `--resume` 恢复；最后按原始索引合并并全量评分 |

**已验证环境（2026-10）**：H800 SM90；`torch 2.10.0+cu130`、`triton 3.6.0`、`transformers 5.3.0`、`flashinfer-python 0.6.18.post1`；AIPerf `0.13.0`。**固定版本，勿盲目升级依赖。**

**当前状态（2026-10-20）**：14B BF16、K/V INT8、FP8 E4M3、INT4 单题真实 Forward 和存储格式/字节数审计已全部通过。1719 条 × 4 臂的八卡正式实验 `prepare` 已通过，但首次 `run` 因 LongBench 缺包全部在模型启动前退出；**正式质量实验尚未完成**。
