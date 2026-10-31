# 技术架构、代码、路径、版本和 provenance

## 两台机器，不可混同

**本地**：5060 Ti 16GB，WSL2，常驻 Codex + 稳定网络，负责 Git 分支、代码变更、CPU 测试、小模型研究、报告、SSH orchestration。SM120 不等于 H800 SM90a；禁止把 5060 CUDA 测试宣称 H800 性能/兼容通过。

**远端**：共享 8×H800，用户高权限账号通过 SSH 登录；H800 不能安装 Codex、不能依赖外网。远端 `BASE` 历史：`/export/home/yinjinrun/.borrow/chenjunyu/xllm-h800`；**必须只读验证远端主机、真实 HOME、真实 BASE、文件 ownership/权限与当前空闲 GPU**。Docker 镜像历史：`quay.io/jd_xllm/xllm-ai:xllm-dev-cuda130-x86`；Qwen14B 模型宿主机历史路径 `/mnt/cfs/9n-das-admin/llm_models/Qwen2.5-14B-Instruct`（容器 `/models/...`）。模型仅只读。

## 代码

- GitHub：`https://github.com/MingASA/xllm_quant_kvcache`。
- 初期 `feat/mixed-v-kv-cache`：BF16 K + V INT8/FP8/INT4 等 mixed-V；传统自定义 Triton、`flashinfer_mixed` 等历史路径不要混入原生性能对照。
- `feat/native-flashinfer-joint-kv` / `72cb5810adb124b87042c71511129ca65a77980a`：native BF16/FP8 Joint, FlashInfer FA2 paged prefill/decode, FP8 resident K/V, Python CUDA Graph warmup/bucket/replay evidence。
- `bugfix/native-fp8-graph-admission` / **`ceff00bbd38054d2b6b4aa44513d773f2bd174b2`**：C++ `xllm/core/runtime/worker_impl.cpp` 的 gate 只允许经过显式 opt-in 的 native FlashInfer FP8 E4M3 Graph；已有 CPU 7/7、syntax 检查；H800 新 binary 主目标最终已确认成功。
- H800 旧仓库：`$BASE/xllm_quant_kvcache`; 原生 72cb worktree：`$BASE/xllm_quant_kvcache-native-kv-72cb5810`; **当前新 Worktree**：`$BASE/xllm_quant_kvcache-native-fp8-admission-ceff00bb`；独立 build：`$NEW/build/h800-native-fp8-admission/bin/xllm`。不要覆盖 `$BASE/xllm_quant_kvcache/build/lib.linux-x86_64-cpython-312/xllm/xllm`。
- 最新新 binary 已最终验证：**537 MB**, SHA256 `277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3`；Ninja `no work to do`、grep C++ admission 字符串存在、`ldd` 没有 `not found`。这**不是 FP8 Graph replay 的验证**。

## 运行时

主程序 C++ 嵌入 Python ModelRunner；`--model_impl=python`，`--python_model_path` 必须指当前 worktree。`XLLM_FLASHINFER_NATIVE_KV=1`、`XLLM_KV_QUALITY_MODE=none`；BF16 `kv_cache_dtype=auto` 且不设量化 backend；FP8 `kv_cache_dtype=fp8_e4m3`, `XLLM_QUANTIZED_BACKEND=flashinfer`, `XLLM_FP8_K_SCALE=0.125`, `XLLM_FP8_V_SCALE=0.0625`。两臂相同 FA2 planner/Attention, disable prefix cache，Graph `--enable_graph=true --enable_prefill_piecewise_graph=false --python_graph_backend=cudagraphs`。C++ INFO 仅 admission，不是 replay 证据。Python 记录 `NATIVE_KV_BACKEND`（两类 `FlashInferBackend` / `FlashInferFP8Backend`，真实 graph buckets）、`KV_ALLOCATION`（48 层、payload dtype、pages）、`NATIVE_KV_GRAPH_REPLAY`。必须同时核对真实 capture 和 replay。

Docker 常用：bind mount `$BASE:/workspace:ro` 和 `$BASE:$BASE:ro`，新 worktree 独立 rw 以用于 build，模型只读，**仅显式允许 GPU UUID**；`--security-opt seccomp=unconfined` 在已验证环境中有必要，但这扩大容器调用权限，必须满足共享机授权、最小化 mounts、绝无 Docker privileged。`PYTHONPATH=/workspace/.venv/lib/python3.12/site-packages:<CURRENT_REPO>`，`LD_LIBRARY_PATH=/workspace/.venv/lib/python3.12/site-packages/tvm_ffi/lib:/workspace/deps/prefix/lib:...`；4 类线程 env=1。Torch `2.10.0+cu130`、Triton `3.6.0`、Transform `5.3.0`、flashinfer `0.6.18.post1`、AIPerf `0.13.0`（历史验证版本，以运行时读取为准）。源码/二进制变更不得默默混用。

## Runner 与实验数据

- `inputs/aiperf/aiperf_h800_v4_2_native.py` 与 `vendor/aiperf_h800_v4_2_native_release.zip`（本地交接复制）。v4.2 有 `run`、`resume`、`status`、`verify`，具体 args 以 `--help`、当前脚本源码为准；`--expected-commit`、`--server-binary`、`--repo` 必须指 ceff 源码/新 binary；`--arms bf16 fp8_e4m3 --group 14b-native --graph-mode cudagraphs --concurrency C --requests 128 --warmup 8`；Runner 有 `--check-only`。Runner 自动验证真实 Replay，不可因 Graph 失败而静默改 eager。
- 冻结 Synthetic 数据：`$BASE/inputs/aiperf/v3_canonical_v32/isl{ISL}_osl{OSL}_n136_seed20261020/canonical_inputs.json`。历史 15 组：ISL 512, 4096, 8192, 16384, 24576 × OSL 256, 512, 1024；每组 136 行，128 measured + 8 warmup（实际脚本选择需检查 SHA；16K+512 样本 SHA `f8f91b44be5336cda88a7bde29cbe2878c8350892649ad1695e4e62ecf06be53` 是诊断报告记录）。
- Mooncake FAST'25 conversation/toolagent/synthetic 冻结 trace 各 264 条；历史 ShareGPT、BurstGPT 等准备状态需以文件查证。`inputs_json`、`mooncake_trace` 与本地 tokenizer 兼容 shim 存于 `$BASE/inputs/aiperf`；不重新生成已冻结内容。
- 质量冻结结果：`$BASE/experiments/joint-kv-14b/joint14b-full-quality-v1-8gpu`；Qwen14B 1,319 GSM8K + LongBench 8×50 = 1719 prompts，各四 arm 分片，历史并行是 8 个单 GPU 独立工作，不是 DP/TP。该旧质量结果和新 native FlashInfer FP8 fixed scale 应独立标记。
