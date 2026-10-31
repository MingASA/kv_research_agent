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


---

## 2026-10-21～24 新增：AIPerf、原生 FlashInfer 与 H800 独立编译

> **适用范围和机器边界：** RTX 5060（Codex）用于源码开发、基本单测和 Git 提交；8×H800 是**另一台共享服务器**，由用户通过 SSH 部署、编译、测量。5060 的 SM120 单测不等于 H800 的 SM90a 完整服务验证。**不要让 5060 Codex 在本机操作 `$BASE=/export/home/yinjinrun/.borrow/chenjunyu/xllm-h800`；它是 H800 上的目录。** 两机通过 Git Bundle / SCP 或正常 Git 同步；H800 的 GitHub 网络可能不稳定。切勿覆盖旧 Worktree、旧 binary、旧实验批次或已冻结输入。

### A. 最近发生的具体故障：现象 → 根因 → 已证实的处理方式

| 故障 / 日期 | 实际现象与根因 | 处理方式 / 结论与状态 |
|---|---|---|
| **CMake 找不到 Rust（2026-10-24）** | `cmake/FindRust.cmake:69: Could not find Rust!`。新 Docker 构建没有传递已存在的 `CARGO_HOME=/workspace/deps/cargo`，因此即使镜像中有编译工具也找不到项目指定的 Rust。 | **已定位、Rust 单独验证通过**。Docker 传 `-e CARGO_HOME=/workspace/deps/cargo`，并在容器 Shell 中执行 `export PATH="$CARGO_HOME/bin:$PATH"`。不要重新安装 Rust、重装镜像或改源码。 |
| **Rust 无权创建线程（2026-10-24）** | `OS can't spawn worker thread: Operation not permitted (os error 1)`；独立排查容器漏掉 `--security-opt seccomp=unconfined`。 | **补回 `--security-opt seccomp=unconfined` 后，未再出现相同错误**。此前 Docker CUDA/Rust 运行也因安全配置遇过故障；不能只给编译器配置加 `--network none` 就当作沿用了生产 Docker 参数。 |
| **Rustup 默认落到 `/root`（2026-10-24）** | `could not create home directory: '/root/.rustup': Permission denied (os error 13)`。Docker 镜像已有 `RUSTUP_HOME=/root/.rustup`，但容器用非 root UID 运行。 | **已验证**显式设置 `-e RUSTUP_HOME=/workspace/deps/rustup`，同时保留 `CARGO_HOME` 与 `PATH` 设置。`cargo/rustc/rustdoc` 均报告 `1.99.1`，`RUST TOOLCHAIN OK`；只说明工具链可用，**不能声称新二进制编译已成功**。 |
| **OpenBLAS 尝试创建 64 个线程（2026-10-23）** | 在检查 Docker Torch 时出现 63 条 `pthread_create failed ... Operation not permitted`，尽管随后仍打印 `Torch: 2.10.0+cu130`。前述工具检查命令忘了限制 BLAS 线程。 | 在**每个新 Docker 调用**统一传 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`，不能因为前一条命令曾设置过就默认下一容器继承。 |
| **CMake 首次配置失败后 build 目录已存在（2026-10-24）** | `cmake -S ... -B "$BUILD"` 配置中途失败，但会创建独立 build 目录和部分缓存。先前启动脚本使用 `test ! -e "$BUILD"`，原样重跑会在这个检查处被拦住。 | **复用已创建的独立 build 目录继续 CMake 配置**；核对 `CMakeCache.txt` 中 `CMAKE_HOME_DIRECTORY` 对应 `ceff00bb` 新 Worktree，沿用相同的 `-S/-B`、vcpkg、ABI 和 CUDA 配置。不要 `rm -rf` 重建、不要覆盖旧 binary；若缓存确实不一致，再定位具体冲突处理。**当前尚未进入 C++/CUDA 编译**。 |
| **H800 v3.3 Runner 将容器路径当宿主机路径（2026-10-22）** | `'/workspace/.../executor.py' is not in the subpath of '/export/home/.../xllm-h800'`；`Path.relative_to(BASE)` 不能将 Docker `/workspace` 映射到宿主机 `$BASE`；恢复时的源码 SHA 校验也有同类问题。 | **v4.1/v4.2 Runner 已修改对应映射并通过 CPU 单测**；路径先按明确 bind mount 映射，再计算宿主机 SHA。不能直接跳过源码完整性校验。旧 v3.3 不再用于正式原生配对实验。 |
| **H800 v4.1 切臂时误判 GPU 忙（2026-10-23）** | BF16 AIPerf 返回 `rc=0`；随后的 FP8 因 GPU2 `memory.used=0 MiB`、`util=88%` 被拒绝启动。稍后连续检查显示 0 MiB、0% 且没有计算进程；**推测是退出后的利用率采样滞后，而非已最终证明原因**。 | 通过 `resume` **已实际跳过完成的 BF16、进入 FP8 Worker**。v4.2 修改为连续 5 秒空闲、最长 90 秒等待，同时保留显存、利用率、计算进程/UUID 保护。绝不能以 0 MiB 为由绕开共享 GPU 的忙碌检查。 |
| **C++ 禁止所有量化 KV CUDA Graph（2026-10-23）** | FP8 Graph 服务在 KV 分配阶段 `CHECK` 崩溃；`xllm/core/runtime/worker_impl.cpp`（旧 `72cb5810`）对所有非 auto 的 KV 强制 eager，与 Python 原生 FlashInfer FP8 Graph 支持冲突；FP8 还未进入真实 Graph Capture。 | 5060 Codex **已提交源码修复** `ceff00bbd38054d2b6b4aa44513d773f2bd174b2`：仅明确放行 `fp8_e4m3` + Python Model + `XLLM_FLASHINFER_NATIVE_KV=1` + `XLLM_QUANTIZED_BACKEND=flashinfer` + `cudagraphs`，保留其他量化后端限制。H800 新 Worktree 已导入；**必须独立编译新的 C++ binary**，旧 binary 仍有拦截。当前编译尚未成功，不得写成 Graph 已验证。 |
| **CUDA Graph Runner 存在，但并未真的捕获（2026-10-22）** | Codex 复核发现 `DecodeCudaGraphRunner.warmup()` 未接入真实 ModelExecutor；旧证据仅根据 Graph Runner 对象存在就声称 enabled，真实服务一直 eager。 | 在 `72cb5810` 已加入绑定 KV Cache 时的 Graph warmup/capture、捕获 bucket 与首次真实 replay 标记 `NATIVE_KV_GRAPH_REPLAY`。必须同时有**已捕获 bucket**和**实际 replay**才能认定 Graph 生效；只有 `--graph-mode cudagraphs`、C++ admission INFO 或 `NATIVE_KV_BACKEND` 不够。SM120 单测通过，H800 FP8 Graph 尚未验证。 |
| **BF16 与 FP8 都答错了示例代码题（2026-10-22）** | 原生 eager 配对 3 题中，前两题语义均正确；第三题 `sum(x*x for x in [2,3,5] if x%2)` 正确应为 `34`，BF16/FP8 均输出 `25`，检查器绝对正确率要求触发 `Semantic check failed`。 | **两臂相同错误不等于 FP8 后端失效，也不能证明模型质量完全等价**。服务正确性应按真实 backend、paged KV 驻留、SSE/usage、Scale、运行证据判定；绝对答题正确率和两臂逐 token/质量差异独立报告，正式精度评估另用 GSM8K/LongBench。**eager 两臂真实服务已运行成功**。 |
| **FP8 KV 总预留显存不降（2026-10-22）** | BF16 KV 每 token `4096 B`，FP8 KV+scale 每 token `2112 B`，但两臂总预留显存都约 `34.6 GiB`。BF16 每层约 1477 pages、FP8 约 2865 pages。 | 这是 allocator 在同样显存预算下分配更多可用 KV pages，**不能用总预留显存接近否认压缩**；对比 resident bytes/token、page count、capacity、实际 dtype、float8 alias。H800 eager 已得到固定 Scale：`K=0.125`、`V=0.0625`，无需为继续本次实验重复校准；其跨任务质量仍待验证。 |
| **原生后端与 Triton 比较口径混淆（2026-10-22）** | v3 Joint BF16/INT8/FP8/INT4 走自定义 Triton Attention，V-only 是 `flashinfer_mixed`，不是原生 FlashInfer BF16/FP8 对照；大量 AIPerf 负载准备不能替代后端适配。 | 正式生产相关核心对照锁定 **原生 FlashInfer BF16 vs 原生 FlashInfer FP8 Joint KV**，两臂相同 FA2 paged prefill/decode 与调度模式。Triton 量化仍是独立实现实验，不能把不同后端的性能差异归因于 KV 压缩。 |
| **5060/SM120 测试误当 H800 验证（过程教训）** | Codex 在 RTX 5060 SM120 机器写代码、做轻量测试，H800 是另一台设备；本机通过不证明 H800 的 C++/Python/FlashInfer/SM90a 兼容。 | 5060 提交 → 增量 Git Bundle/SCP → H800 独立 Worktree → H800 **独立 binary** → 再做真实服务/性能；每步核对 commit 和 SHA。不要将 `~/projects/xllm_quant_kvcache` 等 5060 路径当 H800 路径。 |

### B. AIPerf v3→v4 维护中已解决的配置陷阱

| 陷阱 | 原因及正确行为 |
|---|---|
| AIPerf 0.13.0 的 `--warmup-request-count 0` 无效 | CLI 要求 `warmup_request_count > 0`。**0 预热时省略这个 flag**，不强行改为 1。v3.2 已针对这一点修复。 |
| AIPerf 0.13.0 离线 Tokenizer 解析本地模型出错 | 旧版将已存在的绝对本地模型路径送进 HuggingFace `snapshot_download()`，在 `HF_HUB_OFFLINE=1` 时抛 `HFValidationError` 或多进程错误。v3.1 引入**进程局部** `sitecustomize.py` 兼容层；勿修改全局 transformers/huggingface 包。 |
| `mooncake_trace` / `inputs_json` 的 `inputs.json` 不自动导出 | AIPerf dataset manager 对这些 replay 源明确跳过 `inputs.json` 导出；不能把“没有导出文件”当失败，或把没有记录的文件当审计证据。v3.3 改用真实请求记录、源 SHA、响应/统计审计。 |
| 固定请求 ≠ 完整生产 Trace | 将 Mooncake Trace 抽样成 264 条与 `--no-fixed-schedule` 改变了原来的到达节奏/可能的前缀复用关系；可用于长度分布受控比较，但不能称“完整复刻生产调度”。 |
| 正式 AIPerf 配对不能续拼不同 binary | v4.1 BF16 在旧 binary 下完成（16K+512、C4、128 请求、Graph），但 FP8 因旧 C++ gate 崩溃。**新二进制成功后应在全新结果目录重跑 BF16/FP8**；保留旧成功 BF16 数据作历史记录，不与新 FP8 拼为一组结果。v4.2 Runner 记录源码 commit、binary 哈希和 Graph 证据。 |

### C. 本轮 H800 独立编译：已验证的固定参数与继续方式

实际源码与工具状态（截至 2026-10-24）：

```text
H800 基址:    /export/home/yinjinrun/.borrow/chenjunyu/xllm-h800
新源码:      $BASE/xllm_quant_kvcache-native-fp8-admission-ceff00bb
源码 commit: ceff00bbd38054d2b6b4aa44513d773f2bd174b2
旧 binary:    $BASE/xllm_quant_kvcache/build/lib.linux-x86_64-cpython-312/xllm/xllm
新 build:     $NEW/build/h800-native-fp8-admission  （目录已创建）
Docker 镜像:  quay.io/jd_xllm/xllm-ai:xllm-dev-cuda130-x86
VCPKG:       /workspace/deps/vcpkg
Cargo home:  /workspace/deps/cargo
Rustup home: /workspace/deps/rustup
SM 架构:     90a
Torch:       2.10.0+cu130，CXX11 ABI ON
CUDA:        13.0
CMake:       3.27.9
FA2 ABI:     FLASHINFER_FA2_0_6_18_ABI=OFF （继承旧 CMakeCache）
```

**下一次 Docker 编译必须显式带上**（将这些参数补到此前已有的、挂载及编译方式不变的命令，不要重建全部环境）：

```bash
--security-opt seccomp=unconfined \
-e HOME=/tmp \
-e CARGO_HOME=/workspace/deps/cargo \
-e RUSTUP_HOME=/workspace/deps/rustup \
-e OPENBLAS_NUM_THREADS=1 \
-e OMP_NUM_THREADS=1 \
-e MKL_NUM_THREADS=1 \
-e NUMEXPR_NUM_THREADS=1
```

容器入口运行 CMake **之前**：

```bash
export PATH="$CARGO_HOME/bin:$PATH"
command -v cargo rustc rustdoc
cargo --version && rustc --version && rustdoc --version
# 继续先前相同的 cmake -S "$SRC" -B "$BUILD" 参数，不要删除已创建的 BUILD
# 配置成功后才 cmake --build "$BUILD" --target xllm -j 4
```

**注意构建隔离：** 新目录 RW，只读挂载旧 Worktree、模型、已通过的 binary、冻结输入；不修改老二进制/实验结果。已实际验证 Cargo/Rustc/Rustdoc 版本 1.99.1，**尚未验证这套修复后的完整 CMake 配置、C++/CUDA 编译、链接或 FP8 Graph Capture/Replay**。报错时保存 `build.log` 与准确退出码，不以最后的 `echo PASS` 掩盖失败。

### D. 当前状态快照（2026-10-24）

- 5060 Codex 已提交 `72cb5810`（原生 FlashInfer BF16/FP8 + Python Graph）与 `ceff00bb`（只放行原生 FP8 Graph 的 C++ admission）。
- H800 已导入 `ceff00bb` 到独立 Worktree，14 个子模块已由本地已有 Checkout 复用，并匹配固定提交；v4.2 Runner 的 CPU 单元测试 **8/8 通过**。
- 旧版本上，原生 BF16/FP8 **eager** 14B 服务与实际 FP8 resident KV / Scale 已验证；v4.1 的 BF16 16K+512/C4/128 条 AIPerf 成功；FP8 CUDA Graph 在旧 C++ binary 上**启动失败**。
- 新 `ceff00bb` C++ binary **尚未编译成功**：目前失败停留在 CMake 配置，Rust 环境三个问题已查明，工具链独立验证通过。下一步是在既有独立 build 中修正环境变量并继续配置、编译。
- H800 新 binary 的 CUDA Graph 捕获与重放、同一新 binary 上 BF16/FP8 正式 AIPerf 对照、GSM8K/LongBench 的新固定 Scale 质量验证**尚未完成**。


---

## 2026-10-25 补充（严格仅追加：最终编译成功与遗留陷阱）

> 此章比前面 “截至 2026-10-24，尚未编译” 状态**更新**；历史文字不删，供回溯。新进展只证实主程序构建，不代表 FP8 Graph 性能/质量已通过。最新可见 H800 结果截止 2026-10-25；有时间戳为 10-26 的产物属 H800 机器本地时间/日志显示，交接时以真实 SHA 和检查输出为准。

| 问题 | 现象/根因 | 已采用措施与准确状态 |
|---|---|---|
| 新独立 CMake 缺 `yalantinglibs` | `third_party/Mooncake/mooncake-common/common.cmake:525`: 找不到 `yalantinglibsConfig.cmake`；旧 CMakeCache `yalantinglibs_DIR=/workspace/deps/prefix/lib/cmake/yalantinglibs`，新缓存 `-NOTFOUND`，库实际已安装 | 在**同一新 Build** 配置加 `-Dyalantinglibs_DIR=/workspace/deps/prefix/lib/cmake/yalantinglibs` 和/或 `-DCMAKE_PREFIX_PATH=/workspace/deps/prefix`；无需升级库或删除 CMakeCache。已越过配置阶段。 |
| Cargo 在无外网 Docker 访问 crates.io | 编译 `librust_safetensors.a` 时 `Could not resolve host: index.crates.io`；容器 `--network none`，Cargo 仍默认解析远端索引 | 已有本地 safetensors 0.6.2 等缓存，增加 `CARGO_NET_OFFLINE=true`，复用 `/workspace/deps/cargo` 与 `/workspace/deps/rustup`；**无需 H800 外网**。此项后编译推进。 |
| 全量第三方测试缺 `boost/locale/encoding_utf.hpp` | cpprestsdk 的第三方测试代码头缺失：vcpkg 基础 Boost 1.84 已在，但 Boost.Locale 和 libiconv 等依赖不齐 | 避免为核心实验无谓安装全量第三方测试依赖；**只执行 `cmake --build "$BUILD" --target xllm --parallel 4`**，不运行所有默认 targets 或 cpprestsdk 测试；符合最小增量修复，主目标已成功。若未来研究确需第三方 test，再在隔离依赖环境补齐。 |
| 编译产物存在但没有 RC 证据 | `ls -lh bin/xllm` 537MB、具有执行位只说明文件存在，可能是未完成/早期链接的残留 | 随后已补跑**目标增量构建**：`ninja: no work to do.`，`CPP GRAPH ADMISSION FIX PRESENT`，`ldd` 无 `not found`，`NEW XLLM BUILD VERIFIED`；binary SHA256=`277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3`。编译确认完成，不必再编译。 |
| 构建依赖隔离的隐藏风险 | 旧 CMakeCache 中 `VCPKG_INSTALLED_DIR` 指向旧构建位置，独立 CMake 可复用该安装树，但编译时不会保证第三方测试全部就绪 | 已成功的 ceff 主目标可用作**待 runtime 验证的候选**；对 vcpkg 安装目录只读复用，不删除/重建/升级共享 package，记录 CMakeCache、ABI、依赖 SHA，后续只在真的发生链接/运行问题时排查。 |
| 配对输出目录/Runner 参数与旧夜间脚本未经证实 | 先前给用户的 ceff 16K512 C4 tmux 命令、`run_native_overnight_ceff00bb.sh`、`cjyu-native-night` 只是计划与可复制命令；**本会话没有收到运行终态**。Runner v4.2 真正 job 完成标记为 job 目录 `COMPLETE`，root `COMPLETE` 仅在所有验证完成写入。 | 新 Codex 第一次连上 H800 前只能只读恢复：检查 tmux、docker、PS、runner `status`、root/job `COMPLETE` 与 manifests；不可误以为睡前脚本一定已启动，也不能因还没收到回执就双重启动。`--concurrency` 是 nargs '+' 支持多个值，但要核对按臂预算和结果目录语义。 |
| 高权限自动 SSH 操作安全 | 新控制端计划 5060Ti WSL Codex 经 SSH 自主操作 H800；共享 8 卡、多用户，高权限账户不等于可改他人资源 | 采用先只读巡检/主机身份及权限验证、只写我方新隔离 worktree/结果目录、按用户明确预算用空闲 GPU、tmux 持久化/唯一 job ID/恢复、断连时先只读对账、无全局依赖安装/没有清理他人容器/无 `docker --privileged`。不可把 `codex --yolo` 当无限远端授权。 |

**最新修正状态（2026-10-25）：** 新候选 `$BASE/xllm_quant_kvcache-native-fp8-admission-ceff00bb/build/h800-native-fp8-admission/bin/xllm` 已成功编译并基本链接/修复标记验证，大小约 537MB，SHA 见上。旧独立构建 14 子模块已匹配 pinned gitlinks。v4.2 Runner H800 CPU 8/8 已通过，BF16/FP8 Eager 真实验证已有，固定 FP8 K=0.125 / V=0.0625。**尚无对话中的实测证据**证明此**新 binary**的 FP8 Graph Capture/Replay、16K512 C4 新配对、后续过夜队列与新原生 FP8 全量质量已完成。首先做只读恢复，再决定下一实验。

---

## 2026-10-26 UTC / 5060 本地 10-25 接管新增：实机只读对账与独立验收

本节比上面的交接快照更新；保留旧文，不把当时未知误写成当时已经知道。
证据：`reports/operations/20261026T040426Z/`、`reports/experiments/ceff-takeover-20261026T040426Z/REPORT.md`。

| 现象/问题 | 新证据与处理 | 状态与限制 |
|---|---|---|
| 交接认为 ceff 新配对/夜间队列状态未知 | SSH 严格匹配已存 ED25519 pin；发现六个 ceff root/arm COMPLETE，夜间日志 `ALL OVERNIGHT EXPERIMENTS COMPLETE`。九个配对负载、18 case；没有项目 runner 进程、没有默认 tmux server。 | **confirmed**：任务已结束，不重复启动。其余长期容器未启停；队列/预约全局状态仍不明。 |
| ceff FP8 Graph 是否真实执行 | 将 372 个选择文件（20,580,006 bytes）只读流式镜像，逐 SHA 验证；raw server logs 有双臂 captured batches 与 `NATIVE_KV_GRAPH_REPLAY`，FP8 alias 指向 resident pages。 | **confirmed**：真实 capture + 至少一次 replay。每 server 仅首次 replay 打标，不能断言每个计时 decode step 都走 Graph。 |
| 当前 allocator 与旧 eager 页数不同 | 新 AIPerf BF16 1916 pages/layer、245248 tokens；FP8 3716、475648。48 层 resident 总量均约 44.9 GiB；4096 vs 2112 B/token/layer 仍成立。 | **confirmed**：每 token 含 scale 降 48.4375%、分配容量约 1.93946 倍。旧 eager 约 34.6 GiB 是不同分配预算，不能直接拼总量；最大服务并发尚未满载测量。 |
| OSL=512/1024 被误读成真实固定输出 | 18 case 各有 128 profiling + 8 warmup，实际平均 OSL 211–287。24K/C2 token/s +5.92%，但 req/s −9.70%、输出长度 +17.29%、TTFT P95 +15.38%。 | **confirmed**：是自然停止/可变输出服务测量，不能称固定 decoded steps 加速。具体提前停止原因未逐请求保存，EOS 解释仍为推断。新 fixed-decode 是改变控制变量的新 proposal。 |
| AIPerf throughput 重算出现约 0.007% 小差异 | 初次独立校验用 `benchmark_duration` 作分母不吻合；读取实际安装的 AIPerf metric 源码，throughput 使用 `observation_duration()`，有分析窗口时用 window bounds；导出 start/end 与之匹配。`benchmark_duration` 另由最早 request/最晚 response 算出。 | **confirmed**：修正本地 verifier 的时间口径，18 case 吞吐与分位数均从 raw records 复算通过；不通过放宽阈值掩盖差异。保存 installed source 及回归测试。 |
| 旧 runner 不能直接当符合新政策的执行器 | v4.2 继承 v2.1 的 `$BASE/cache:rw`、`inputs/aiperf/.cache:rw`，切臂 stable idle 为 5 秒，无可信总预算强制监督。其 `verify` 会重写 CSV/JSON。 | **confirmed（源码/argv）**：首次未调用 verify，也未重新运行。未来必须 run 独立缓存、政策规定的稳定空闲、许可/预算/UNKNOWN 恢复；本地只读预检不是完整 runtime 配额执行器。 |
| Python 3.9 流式 tar 参数兼容 | 初次 S0 下载在 `tarfile.open(mode='w|gz', compresslevel=1)` 因不支持参数失败，未读取结果内容；本地保留 partial archive/transport rc。 | **confirmed**：移除该参数，补流式 roundtrip/路径穿越/软链接/hash 失败测试，新本地 attempt02 下载成功。远端无 tar 文件、无任何项目写入，更没有重试 GPU。 |
| 历史 joint14b 质量是否已完成 | 远端 merged-global complete 与 summary/CSV 原始字节 SHA、shard_manifest SHA 一致，支持 1719×4 的历史汇总档案。 | **confirmed（归档一致性）**；分数仍是旧实现报告，未重读/重评分全部 predictions，不升级为新 native fixed-scale 质量通过。 |
| 文本 CSV 的 SHA 与原文件不符 | Python `read_text()` 对 CRLF 做 universal-newline 转换，重新编码后的 CSV SHA 不等于历史 raw file SHA；远端直接对原始 bytes 计算则匹配 complete。 | **confirmed**：审计 hash 必须用 bytes；文本规范化可以比较内容，不能替代原始字节完整性。没有修改旧 CSV。 |
| 空闲但缺许可 | 04:03 与 04:29 UTC 八卡均 0 MiB / 0%，无 compute apps；可信 `~/.config/kv-research/approval.json` 与兼容 state 文件都不存在。 | **confirmed**：空闲不授予 GPU 权限，未验证连续 120 秒稳定或团队预约。此会话 remote writes=0、GPU launches=0；继续本地研究与实现。 |

新本地工具：`scripts/mirror_readonly.py`、`scripts/audit_native_results.py`、`scripts/controller/approval.py`。
31 个 CPU fixture tests 通过。批准预检即使静态范围有效仍 `can_launch=false`；完整预算 supervisor、原子预留、精确进程归属与恢复尚未接通，不宣称已经 OS 隔离。

**本轮后续补充：** 最新报告 `reports/experiments/ceff-takeover-20261026T040426Z-v2/REPORT.md` 加入 profiling-only NVML 抽样；BF16 device memory 采样峰值 76225–76253 MiB，FP8 76419–76445 MiB。不要把 MiB/1000 误标成 GiB；MiB 转 GiB 应除 1024。采样约覆盖 99% profiling request span，不能认证间隙瞬时峰值。新增两个回归案例后，当前 CPU suite 为 **33/33**。`science/hypotheses/h2-decode-pressure-20261025.md` 根据源码区分 resident scale storage 与 FlashInfer payload 读取；尚无硬件带宽证明。

## 2026-10-26 生产候选新增：FA3、真实请求与退出竞态

用户最新生产目标和持续空闲按需授权优先于旧FA2固定/合成矩阵及“必须手动SSH”的规则；共享安全与证据要求不变。当前生产标准为 `ops/PRODUCTION_DECISION_STANDARD.md`。

| 现象 | 已验证的处理与证据 | 限制 |
|---|---|---|
| 只改FA3 planner会拒绝原BF16-Q/FP8-KV组合 | 已安装0.6.18.post1实际能力判断要求该FA3路径Q也FP8；独立Python源码 `7831a874` 显式Q dtype/静态scale1.0及单次Triton转换。H800两臂FP32-reference、partial-page/replan与真实Graph算子检查通过，14B真实模型也capture/replay。 | FP8-Q是新增数值近似，不能沿用旧FA2/Triton质量认证；不静默fallback。 |
| 将上游6月SM120 FP8 FA2问题当作本机缺补丁 | 实际H800安装头文件已有shared-memory repack；上游修复不是本机H800归因证据。image env `FLASHINFER_VERSION=0.6.2` 也不是实际import版本，backend记录为0.6.18.post1。 | 真实瓶颈仍须生产观测，NVML activity不等于GB/s。 |
| CPU prompt长度全部2 | transformers5.17的模板tokenize返回BatchEncoding，`len`数的是字段；改显式render+encode。旧派生包标INVALID保留，新v2请求1007条，服务端usage再核验。 | 无效包从未派发GPU；不能用CPU全池估计替代实际执行子集counts。 |
| 原Mooncake264子集与manifest索引对不上 | `origin_line_indices`实际为从1起的源行号；三source及subset SHA、全部映射已核验。 | 随机稀疏子集会破坏完整到达/复用，生产回放用完整来源或连续窗口。 |
| 两臂AIPerf完成但root CONFLICT | ShareGPT FA3任务在FP8最后profiling响应9.00034秒后触发归属检查；只停我方容器并释放驱动。之后只读发现两臂measurement都已封存，56证据SHA/79镜像文件及1024profiling+32warmup raw payload/SSE/usage/Graph全部独立通过。 | 保留root CONFLICT/无COMPLETE，分类恢复测量，不伪造正常完整运行。具体冲突PID未记录，不能回溯确认为误报；需干净复测与质量。 |
| 已退出的GPU PID被直接归为foreign | CPU故障注入复现旧检查FileNotFound立即冲突。后续只对**此前验证为我方容器**的PID退出作一次新GPU只读复核，确认不再列出并再查其余进程；未知PID/复核仍忙/新foreign/启动前竞态继续拒绝，新增完整冲突观测事件。99控制器CPU测试通过。 | 这是收窄退出竞态，不绕过空闲或他人占用；不是该次真实冲突原因已证实。 |
| 新质量只验证hash而没检验官方依赖 | FA3 pilot在同生产容器先隐藏CUDA，实际import官方LongBench scorer并调用QA-F1/code-sim均1.0，成功后才加载14B；独立旧deps只读加入PYTHONPATH，不安装/修改共享依赖。 | 当前pilot在运行；80题不构成全量1719质量通过。 |

## 2026-10-26 完整FA3质量、原trace来源与只读镜像补充

- Full质量：pilot80与completion1639精确合并每臂1719，原评分器复算GSM flexible89.31→88.63%，LongBench macro50.22→48.22%；比80条pilot风险估计更准确，当前配置不作生产默认认证。报告production-fa3-quality-full1719-20261026。
- XQA API的BF16 Q不代表内部BF16计算。实际安装data/csrc/xqa/mha_sm90.cu的F16QToF8Converter::store把InputElem直接转CacheElem，FP8 KV时仍为FP8 Q。证据flashinfer_xqa_sm90_source_v1.json；未进行XQA GPU测量。
- BurstGPT旧peak桶以源首时间戳5秒为原点，并非0秒。6385条窗口为[3063605,3064205)，按0对齐会漏尾端101条。完整source SHA与manifest匹配，顺序/重复记录均保留。
- 65MB质量raw证据只读下载超过240秒，首残片保留且不标镜像成功。新增显式有界30–900秒选项，超时也写transport journal；新c2目录900秒下载152文件成功、逐SHA通过，没有GPU重跑。

## XQA接入首个实际失败（2026-10-26）

d68f5500在NaN-poisoned unused KV末页的Graph参考处10240输出全NaN；未加载模型、未启动BF16，GPU释放，失败证据production-xqa-operator-20261026a。假设为无效V lanes的0×NaN传播，尚需控制。候选0598400c采用首次绑定全payload零初始化、同地址rebind不清有效数据、禁止替换storage，并增加极大有限尾页参考与单独NaN K/V诊断。不能把该修复候选或CPU通过写成H800通过；原失败不覆盖。

XQA b的paddedGraph误差已CPU定位为新增测试夹具alias：benchmark _attention_metadata的pages.reshape(-1).to(int32)在dtype已为int32时不复制；重规划写indices将pages改成[[2,0,3],[1,1,5]]，后续row1把poison尾页当整页。d60812fd只为测试planner clone indices、保持reference页表不可变，93 CPU通过。GPU复验待c，不把b误差当upstream kernel缺陷。a的NaN尾页问题独立保留。


## XQA c真实验收及生产trace接入（2026-10-26）

ede2b9e7修正夹具页表alias并按生产预留page0布局后，H800全部operator数值/partial/replan/非连续Q/padded Graph通过。保持同Graph与PDL的控制：unused K NaN仍10240输出有限、unused V NaN则10240全NaN；恢复有限padding后重验通过。首次payload初始化是部署要求，无整份BF16 shadow。c两臂80预测已独立重评分，GSM57→58/64，tiny LongBench macro43.4274→39.2495%，无全量质量认证；仅剩余1639补测派发，原a/b失败不覆盖。

源码发现容量观测边界：BlockManager构造函数allocate预留page0也计入num_used_blocks，但kv_cache_utilization的分母num_total_blocks排除page0，故比例满载可略高于1。采集器不能以严格≤1拒绝满载样本；记录physical_pages/reserved0，比例不是百分数。prefix-cache空闲块可回收，preempted仍为迭代gauge。

原连续toolagent payload64.9MB超过32MiB上传单文件界，在GPU启动前CPU检查发现；原包标记未启动并保留。新方案gzip传输，新run内有界展开、精确长度与原SHA验证；32MiB单文件/64MiB总传输界不改变，不切断到达或前缀。新trace CLI最初因sys.path缺控制root发生CPU import失败，修正独立入口后准备通过，无远端动作或GPU失败。


## 完整XQA质量的退出冲突与单臂恢复

completion1639 a最后预测22:41:15.895059Z，归属保护22:41:20.496Z触发；rootCONFLICT/driver释放，FP8 measurement59文件与1639预测独立验收，BF16从未开始。不能重跑已完成FP8；按精确原结果/audit/源码/输入绑定仅补BF16。旧触发PID1438223、UID1005但cgroup为根，当时未记录stateZ，不倒推确认为误报。新CPU保护仅对已有本方身份且Z/消失进程做有限driver再查，PID仍在或其他foreign仍拒绝。127项控制器CPU验收包含缺失臂和退出边界，GPU验证仍待新run。


## SSH prepare回执丢失且目录不存在的恢复（2026-10-26）

ShareGPT XQA r0 a的SSH255、UNKNOWN和2GPU-h预留均保留；多次S0返回ABSENT，不据单次缺目录重发。追加同boot双观测87.128秒、确认原本地launcher结束、无远端RPC/run进程/任何run容器/选定卡compute app后，仅在本地结算NOT_LAUNCHED/0GPU-h，没有创建远端result/成功标记，也不释放旧run/replicate去重。新r1 b是独立恢复提案，重新完整准入。早先双观测判定拒绝时未保存具体快照，原因不能回溯确认；采集器已改为判定前先保存观测。新transport保留实际SSH退出码及有界stderr，避免旧CalledProcessError仅倾倒整个RPC源码却丢失stderr；原255根因仍unverified。134项控制器CPU通过，无新GPU测量成功声明。证据sharegpt_xqa_absence_resolved_a5.json。


## SSH间歇失败与有界传输（2026-10-27 UTC）

第二次ShareGPT prepare也出现server-not-responding；随后小S0发生banner/kex失败，所以未确认仅大包导致。三组有界S0重试失败后退避60/120秒，小连接恢复；b再以同boot46.003秒双观测无root/RPC/进程/容器/选定卡计算进程证明NOT_LAUNCHED，仅本地结算0GPUh，旧编号不重用。139项CPU验证覆盖有界gzip输入（96MiB JSON封装界）、截断不执行原程序、解压界和原数据一致；每资产32MiB/总64MiB界不变。SSH单连接keepalive容忍延长为10秒×6，仍受RPC90/prepare600秒硬超时，不改服务器网络/SSH配置。实际S0将5867789bytes资产压成2561590bytes传输、14项SHA全匹配、远端零写入/零GPU；这证明传输路径可用，不证明网络根因或永久修复。下一次唯一新r2仍完整准入，不把两次未落地当GPU重复。


## 原连续trace的传输界与无损XZ（2026-10-27 UTC）

toolagent2088完整payload约64.9MB，gzip22.4MB使prepare触及600秒超时；旧d没有prepare/start，双观测验收NOT_LAUNCHED/0GPUh，不重启旧ID。新的CPUprepare硬窗口1200秒，不增GPU7200秒/额度。XZ将同一payload减到4.86MB，保持原datasetSHA、到达/OSL/prefix/请求数和scientific key；旧r1a仅本地gzip提案未启动，退役后另冻r1b。单资产32MiB/总64MiB传输及64MiB展开界不变，XZ额外128MiB解压内存界、必须完整单stream、拒截断/尾随/多stream。142CPU测试通过，H800 Python3.9.9自带lzma只读fixture通过，无安装/全局修改。未据CPU压缩或fixture声称实际GPUtrace已成功。

代码模式分析教训：官方code_sim找不到合格行时返回默认空串，不等于原输出有空白行或全whitespace。v1/v2模式字段合计两者，v3已区分并查原始输出，主质量分数不变；不要从零分猜测纯评分器问题，解释性输出也会违反代码请求契约。


## 冻结运行时必须随协议一致验收（2026-10-27 UTC）

XZtoolagentr1b已完整prepare/监督器启动，但在GPU模型前由旧policy拒绝，root正确NOT_LAUNCHED/driver释放/0GPU-h。根因prepare_trace_package只刷新trace模块，继承ShareGPT冻结包的旧gzip-only policy。已改为冻结整套当前controller；CLI在任何SSH写入/额度预留前用实际包内policy、干净CPU子进程对完整request/可信approval执行validate_request，不再只相信当前控制端policy。144项CPU通过，真实旧XZ包CPU可复现拒绝、同数据新c的实际frozenpolicy通过。旧b结果不改成功，不重启；新run/replicate明确仅补尚未发生的GPUtrace。

## 2026-10-27：退出中的R状态、kernel flag与跨Python版本审计

BF16-XQA a在真实operator全部通过后，PID2594157的原本方UID/start/boot不变但cgroup根/stateR触发CONFLICT；driver已释放，无服务测量。旧事件没有kernel flags，原因保持unverified，不能倒推成功。H800内核5.10；官方do_exit在cgroup_exit之前设置PF_EXITING=0x4，stat可能仍R。本地24个自有CPU退出均观察到R+PF_EXITING，不等于旧H800故障证明。新保护仅对此前本方同PID/UID/start/boot且Z/消失或内核PF_EXITING正证据执行最多3次driver再读/1秒，必须确认真消失并继续核查其他PID；普通R/S迁移、未知/复用/其他UID、driver仍报告均拒绝。150CPU验收通过；新run再验，旧run不重启。完整说明production-bf16-xqa-operator-20261027a/REPORT.md含官方源码链接。

首个toolagent独立audit在本地Python3.10严格比较失败：只涉及31个mean字段的约1e-16至1.5e-10舍入差，其余每请求/clock/count/分位/SHA一致。用与远端相同Python3.12重新执行既有审计后，全部对象逐值严格相等通过；未放宽比较、不修改原测量/旧失败记录。后续trace独立审计使用/home/mingasa/venvs/vllm/bin/python -B，并保存解释器版本。Python3.10失败c1保留，正式c2_py312。

Toolagent HTTP200失败的raw不是JSON SSE error：packets.name为服务端并发上限纯文本，value=null，AIPerf报InvalidInferenceResultError。错误全部保留，不能按HTTP200算成功，也不要把它泛化为所有失败都是结构化SSE error。当前并发上限为200，不据此声称FP8值损坏或将原失败删掉。

## 2026-10-27：最后可见内容与完整SSE完成的计时边界

低/中速BF16 worker汇总中，offered E2E均值减request_latency不只等于client start lag。读取已镜像的实际AIPerf源码：request_latency取最后非空content，request_end_ns取最后raw SSE包；并用原toolagent所有799个成功测量请求逐条复算，两种时钟和官方ITL严格一致。原速BF16/FP8的无内容尾段P95分别10.932/8.718秒，最大41.483/52.624秒；不能将最后内容发送当整个请求已完成。无可见内容的真实生成token、SSE结束/调度等原因不能仅从现有raw隔离，具体token IDs未导出，不把EOS假说写成事实。

原官方ITL P50 85.054/84.656ms；以first-content至last-SSE除(server输出count−1)的新增完成时间代理P50为88.016/86.947ms。代理非GPU kernel/硬件step计时，也非每token stall分布。保留官方raw/summary/冻结runtime及历史数值，报告中区分content latency与最后SSE完成、匿名OSL仿真与自然EOS语义请求；新增代理和示例qualified曲线单列，不热改正在跑的实验。脚本analyze_stream_completion_gap.py、证据toolagent_original_content_terminal_gap_CPU_v3.json；2个CPU时间边界测试通过，无新增GPU/模型预测。

## 2026-10-27：完整成功trace的raw超过128MiB取证界

两个新ToolAgent压力点的已封存BF16 raw约150MB，超过旧单文件128MiB。不是GPU失败，不重跑。新增显式256MiB仅允许精确我方run/output/(bf16|fp8_e4m3)/artifacts/profile_export_raw.jsonl，需本UID/匹配ticket+measurement_success+requestSHA和worker封存raw SHA；其他大文件拒绝。远端tar和本地核验/写入改为1MiB以内分块，验证源size/inode/mtime不变。原16/64/128默认、单批总512MiB、XZ解码128MiB、展开520MiB、路径/软链接/全部SHA/新目录及无远端临时文件保持。

169CPU验收后实际H800中速trace89文件395316425bytes（两个>128MiB raw）完整逐SHA与独立请求/时钟/usage/Graph重算通过，256MiB只读路径已实证。新完整控制器171CPU通过，旧archive和正在跑的runtime不改。证据readonly_mirror_256_actual_H800_acceptance_v1.json；报告production-bf16-xqa-toolagent-s025-20261027。不能为取证限制去重跑已完成GPU或静默删raw。

## 2026-10-27：新诊断kind未接入传输白名单

组件诊断a在prepare返回Compressed dataset is only a trace input；policy/实际frozenpolicy已接受新kind，但remote_rpc.decode_dataset_asset仍只接受两个旧trace类型。拒绝发生于任何mkdir/start之前；保持UNKNOWN预留并先S0，随后同boot间隔至少30秒的双观测确认无目录/RPC/run进程/任何run容器/选用卡计算进程，本地结算NOT_LAUNCHED/0GPU-h，不创建远端成功文件、不重用a的ID或replicate。证据component_diagnostic_prepare_absence_resolved_a2.json。修复只明确加入新诊断kind，并在CLI任何SSH动作前对真实包执行同一stdlib解码函数；64MiB展开/128MiB XZ内存/原SHA及32/64MiB传输界保持。CPU修复不是GPU采集成功，新b重新全部准入。

## 组件探针同名CPU/GPU annotation的误计

b实际BF16算子通过，但小Graph活动检查失败，未加载14B/运行AIPerf；rootFAILED与0.052761936GPU-h全占用计量/driver释放保留。48文件21,747,368bytes只读镜像逐SHA通过。真实probe.trace有6个CUDA kernel、2次cudaGraphLaunch、逻辑设备0/单deviceProperties，以及CPU user_annotation2个、GPU gpu_user_annotation2个。旧探针只按name计数得到4，要求2而拒绝，原因确认为检查器类别错误，不是CUPTI不可用。source82bed5fd按CPU类别检查，真实CUDA kernel/launch/设备门槛不降低；CPU92+skip1与控制器181通过。旧b没有执行到数值输出assert，不能回溯标探针完整通过；新c仍须重新真实数值/Graph/活动准入。只读活动时长分析同样按CPU类别归属，不把gpu_user_annotation当CPU耗时。证据component_diagnostic_b_raw_failure_and_CPU_repair_v1.json。

## 组件诊断中的元数据语义与窗边界

实际c双臂采集/Graph/原始审计通过。kv_context_tokens字段直接保存kv_seq_lens_host_values；C++ getter返回kv_seq_lens_vec，而CUDA batch builder累计seq_len，故实际0起/sequence+1的累计向量，相邻差是各sequence含当前query的KV长度。全部96forward布局核对，原始字段不改，不将其当逐序列长度。约24ms/臂设备活动在被记录forward之外，原始轨迹保留并显式报告outside_recorded_forward，不能为凑100%强配。CPU scopes、kernel累计和busy union不互换；3窗口/17vs18prefill实际批次不同，不把累计差直接当dtype因果。

## 原生工具任务的公开参考与CPU夹具

固定BFCL commit6ea57973公开live_simple258中，两条live_simple_106-63-0/112-68-0共7个required array参数候选为[]，原checker无法匹配。保留全部原题/分母/官方函数，不通过改答案或剔除记录消除；报告单列参考限制。首版CPU canonical夹具误把optional timespan的空串sentinel作为实际整数，原checker正确拒绝，修夹具为按原optional规则省略后其余256例通过。原v1失败保留、v3仅CPU契约通过，不能当模型质量。工具输入冻结先于答案下载，模型或实际外部API均未在此CPU验证执行。

## 原生工具流式参数后缀重复与完整JSON验证

流式a首warm请求84output/892prompt的原始SSE包含重复参数后缀，标准delta拼接后JSON无效；55文件875600bytes逐SHA独立核验。算子/真实模型Graph通过但无AIPerf/FP8，rootFAILED/driver释放与实际0.096657004GPU-h保留。内部model/parser责任未复现，不把线上坏参数静默去重修好。新的完整JSON协议是执行前选择已有生产路径，原274输入仅stream=false/移除options，不认证旧流式成功。

实际AIPerf TextResponse是dataclass、无model_dump；CPU验证脚本最初误用，v5改用.text并验证实际RawRecordInfo.model_dump(exclude_none=True,mode=json)的responses[].text/perf_ns原样保留，274sessions/548HTTP全部通过。导入路径错误和该API误用均在GPU前修复，无服务结果或GPU消耗；不能把CPU验证失败当网络故障。buffered只报告完整JSON完成延迟，不把该延迟叫流式TTFT或填ITL。自定义S0响应缺ok封装被安全拒绝也不是SSH掉线，改用已验证只读镜像。

后续CPU v2已确认可复现根因：实际buffered warm参数包含height1080/image_formatPNG，若合法模型JSON按prompt,width,height,image_format顺序输出，nlohmann.dump的键排序会改变已经发送的参数前缀；旧sent偏移继续用于新序列化字符串，越界prefix.substr异常又被吞。原C++核心/partial parser/API flush函数体不改，chunk1/2精确复现原坏参数字节；其他多种分块可静默得到合法但错误的参数。完整排序键输入9种分块全对作控制。原始模型token流未导出，历史实际键次序仍unverified，不声称逐token历史回放；已证明有效模型输出足以触发该接口缺陷。不是KV量化原因，不热改当前run或客户端修复。证据native_tool_stream_CPU_late_key_defect_v2.json，完整JSON BF16当前已真实API/AIPerf worker验收。


2026-10-27T11:05:39.094936+00:00 AIPerf0.13 CLIConfig.warmup_request_count为Optional但Field(gt=0)：不做warmup应省略选项，不能传0。wire loader/transport CPU通过不验证CLI参数。range probe a小Graph/计数器/14B admission/reset通过后在CLI拒绝、无128正式请求/真实activation导出，FAILED/driver释放保留。实际完整argv应在CPU以同Cyclopts/CLIConfig解析再派发；原合成control_probe导出不当模型分布。


2026-10-27T11:33:03.782463+00:00 真实ShareGPT JSONL含合法U+2028：str.splitlines会拆字符串，raw128LF被误拆136，抛Unterminated string(char96881)。b83文件逐SHA及128原payload/SSE/usage/完成可按物理LF UTF8流式读取完整恢复，原raw和FAILED终态不改。JSONL逐行应按实际LF，不能把Unicode行分隔当record delimiter；真正坏JSON仍拒绝。设备统计必须client完成后先导出再做任何postprocessing，避免审计错误触发模型清理后计数不可恢复。合成probe导出不当真实模型数据。


## 2026-10-27：conversation现代参考pre-model未知PID可见性竞态

原run prod-trace-bf16xqa-conversation-s1-r0-20261027a在小operator准入时触发CONFLICT；未加载14B/未启动AIPerf/无arm measurement。原事件只记GPU0两个PID4170450/4179970与unresolved_gpu_pid_visibility_race，没有具体缺失PID或旧identity，不能证明是本方短命进程，也不能认定他人作业。只释放本方container，fresh S0确认8卡空闲；57文件40,435,677B逐SHA核实，原终态不改。后续仅增unresolved_gpu_pid/previous_process_proof/missing_proc_file诊断字段，未知PID继续fail-closed；不把此前已知owned退出的窄grace套到未知PID。允许独立新编号在另一张卡完整准入后重试相同未运行的服务实验，不增加独立用户。证据modern_conversation_a_independent_conflict_audit_v1.json。


## 2026-10-27：已完成服务实验的低速只读取证与冻结资产复用

conversation现代参考b原COMPLETE/两臂worker封存/driver释放，原始输出仍在H800不改。只读gzip下载观测低速，Agent按精确local PID/UID/start/boot/父子关系及pidfd，仅停止本方SSH传输；transport rc255是此主动中止的结果，不能误写为GPU或推理失败，也不推断共享网络根因。合并XZ同样保留partial。新S0对34个已知本地input/runtime逐文件核验远端SHA/size/UID/无软链接/读取前后stat，以及原ticket/request/rootCOMPLETE SHA；仅复用这些已证明字节相同的文件，不复用任何raw输出。70个输出/root文件分臂只读完整传输，原大小/路径/总量/单文件界及900秒不放宽；新组装目录再次逐SHA，并保留原下载、子snapshot与cache attestation。当前输出审计未完成前不标独立通过、不为传输重跑GPU。工具reuse_verified_mirror_assets.py，5项CPU故障注入包含cache/raw改变、路径逃逸与分片合并。


## 2026-10-27 新原生KV构建：common Options漏字段与增量source生成头

confirmed：CPU d(source59fc32ac)在distributed_runtime/master.cpp编译失败，runtime::Options已有KV backend/K/V scale字段，而common::Options缺失；未产生binary。862c29ca补齐统一传播，并加入typed prefill/decode选择；C++编译与服务验证分开记录。

confirmed：第一次增量E把新私有source映射到原D virtual source路径、复用自己的D/build；brpc CMakeLists.txt:111需要向source/third_party/brpc/src/butil/config.h.tmp写入，source只读导致configure失败。保留E源码/manifest/result/log和原D源码/失败证据。新F仅把新私有副本brpc/src/butil作为RW overlay，其余source和既有依赖保持RO；F configure完成，进入145个增量target编译。此时尚无new binary成功声明。

一次E git fetch返回1后，新commit对象实际已导入；经过只读确认source仍clean parent、目标commit存在、无容器/PREPARED，再以新不可变resume intent完成一次no-refetch checkout/create，未重复未知prepare。原fetch错误的具体底层原因未证实。后续本地bundle/clone显式禁用递归submodule fetch并将Git允许协议限制为file，避免H800外网依赖；不改全局Git配置。

证据：reports/operations/nextgen-native-kv-20261027a/CPU_build_d_failure_complete_luna_a1、CPU_build_e_failure_complete_luna_a1、CPU_build_f_readonly_luna_a1/a2/a3。旧失败不能重写为成功，CPU-only不等于模型/Graph/性能通过。


### 2026-10-27 CMake实际binary目录与CPU启动配置

confirmed：新F完成145/145和36/36 CXX CPU tests(26+8+2，无skip)，实际产物D/build/xllm/xllm，frozen driver误用了旧build/bin/xllm，因此原F result=FAILED。保留该结果，G仅把既有binary独立封存，没有重编译。新driver明确区分两种CMake产物布局，缺失/多份/软链接均拒绝，增量复用也拒绝覆盖任意已链接布局。

confirmed：第一轮CPU metadata容器遗漏了项目已有seccomp=unconfined profile，xllm在main前的brpc sampler线程创建报Operation not permitted并SIGABRT；ldd已通过。G/verify-v2使用现有服务Docker配置(非root/capDropALL/无GPU/断网/只读/1CPU8GiB，seccomp=unconfined及4个BLAS线程变量=1)，--help实际返回0且5个typed KV flags齐全。首轮文件/容器与F失败不覆盖，不把容器配置差异称源码修复。新binary已CPU核验，模型/量化质量/性能仍待GPU。证据CPU_binary_g_main_independent_metadata_a1.json及完整G metadata镜像。


## 2026-10-28：统计输出目录与atomic progress重命名的竞态

FP8FA2 t真实模型/20known/Graph通过，1720生成到约918durable时Supervisor报FileNotFound(.quality_progress.json.<temp>)，停止自身容器137、OOMfalse。错误直接来自活跃目录tree_bytes的walk/lstat窗口；同原函数CPU故障注入可确定复现文件列出后被atomic rename而缺失。修复为最多3次完整扫描重试，不跳过文件/不返回0/不取消磁盘额度；持续churn或其他I/O错误仍拒绝。新增4目标测试和绑定原始FAILED/释放/相同部署/精确已测试supervisor SHA的有限恢复，336控制器CPU通过；原结果/部分raw不改成功。证据 progress_atomic_scan_race_actual_reproduction_a1.json、FP8_FA2_t_main_reconcile_failure_a1.json。另只对自有atomic_json独特临时文件清理的ENOENT作安全忽略，避免分布式FS过期exists造成同类假失败；不清理他人数据。


## 2026-10-30：本地只读SSH复用误入确认模式

confirmed：session恢复时readOnlyMaster先设置ControlMaster=yes又传-M，本机OpenSSH8.9实际ssh-G解析为controlmaster ask，复用触发ssh_askpass缺失；a的原错误/partial保留。删除重复-M，保留-N、原host pin/BatchMode/只读mutation拒绝和600秒TTL；实际parser变为true，8项目标测试通过，新独立b socket已建立。此修复只在本地只读传输，未修改SSH配置/服务器/实验runtime或GPU准入。另整包xz的SSH server-not-responding不等于已完成ab的GPU失败，不能为取证重跑模型。证据reports/operations/session-recovery-20261030-01a1765d/readonly-master-effective-options-reproduction-a1.json。
