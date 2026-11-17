# FP8 PR 代码审查与修复

按仓库 `.agents/skills/code-review/SKILL.md`、共享规范、Python/C++ 规范执行。审查固定官方基线 `261274e5` 至 FP8 PR `a00beb76` 的完整差异，并审查修复后的增量；没有自动 rebase、升级依赖、合入 Q4 或执行 GPU 作业。

生产 PR 分支：[pr/native-fp8-kv-cache](https://github.com/MingASA/xllm_quant_kvcache/tree/pr/native-fp8-kv-cache)。

| 提交 | 内容 |
| --- | --- |
| [1458605a](https://github.com/MingASA/xllm_quant_kvcache/commit/1458605ab2d6ac61bbf3a0cdac3201bb2dd0550b) | 修复缓存／XQA 页契约，增加准入和回归测试，整理 Backend 可读性 |
| [a945df9f](https://github.com/MingASA/xllm_quant_kvcache/commit/a945df9f402232e72748155383e209786b7bef88) | 仅整理 Writer、Query Kernel 和 Executor 调用格式，AST 不变 |

最新提交：`ba87794dad1a8c6f7244b6168dbc8b43d8450645`。另以 ba87794d 加入可选 Triton 缺失时的显式测试跳过，避免非 CUDA 平台收集测试失败；缺失依赖探针验证通过。

作者和提交者均为 MingASA。NVFP4／MXFP4 研发分支保留原有独立提交，没有把其实现或实验材料带进 FP8 主线。

## Strengths

- [真实字节分配](https://github.com/MingASA/xllm_quant_kvcache/blob/ba87794dad1a8c6f7244b6168dbc8b43d8450645/xllm/core/kv_cache/storage/kv_cache_utils.cpp#L258)、容量核算、不可变 Scale 配置与 Python Executor 透传确有必要，予以保留。它们保证 KV 真实压缩常驻，GDN 状态保持原 dtype；不是临时实验基础设施。
- [执行路径](https://github.com/MingASA/xllm_quant_kvcache/blob/ba87794dad1a8c6f7244b6168dbc8b43d8450645/xllm/python/attention/flashinfer_native_kv.py#L225)保留完整 packed FA3 Prefill／普通混合批次和纯 Decode XQA，没有重启已否决的混合拆分，也没有失败驱动的 fallback。
- Writer、Query 模块、CSR 转换 Kernel、XQA `run()` 和 Executor factory 的计算 AST 与 a00beb76 一致；变化集中于输入契约、状态检查和可读性。AST 一致不是新 GPU 编译或性能证明。
- 官方 BF16 Backend、Scheduler、Graph runners、Qwen GDN 文件仍与固定官方基线一致。没有增加公共执行优化或变更冻结 Scale／质量规则。

## Issues

### Critical (Must Fix)

本轮没有确认 Critical 级代码缺陷；这不代替完整 native build 和设备验证。

### Important (Should Fix)

**1. 已修复：各层缓存页几何可以不一致。**

- 位置：[原绑定入口](https://github.com/MingASA/xllm_quant_kvcache/blob/a00beb76a8a4b7aabc36ae34ea212371e8518418/xllm/python/attention/flashinfer_native_kv.py#L87)；[修复后](https://github.com/MingASA/xllm_quant_kvcache/blob/ba87794dad1a8c6f7244b6168dbc8b43d8450645/xllm/python/attention/flashinfer_native_kv.py#L104)。
- 问题：每层只按自己的前两维验证 K/V，没有验证所有 full-attention 层共享 block count/page size，也允许空页存储。
- 影响：Backend 只有一套 page size、slot mapping 和 planner；混合几何可能使 Writer 跳过越界槽或使缓存与 Attention 页计划不匹配。这里只确认准入缺陷，没有声称观察到 GPU 越界故障。
- 修复：在 bind 边界一次性验证非空、共享页几何，并在建立别名前拒绝不兼容布局；保留 GDN 空 K/V 槽。

**2. 已修复：XQA 原生页表转换没有验证连续的一维输入。**

- 位置：[原 prepare](https://github.com/MingASA/xllm_quant_kvcache/blob/a00beb76a8a4b7aabc36ae34ea212371e8518418/xllm/python/attention/xqa_decode.py#L68)；[修复后](https://github.com/MingASA/xllm_quant_kvcache/blob/ba87794dad1a8c6f7244b6168dbc8b43d8450645/xllm/python/attention/xqa_decode.py#L79)。
- 问题：只检查 dtype/device/元素数，stride=2 的 index 向量仍能到达按裸指针连续读数的 Kernel。
- 影响：例如逻辑 `[0,1]` 的 stride=2 视图底层为 `[0,77,1,77]` 时，裸指针会读到不同页号。真实 CPU Tensor 加 Kernel 入口 spy 已确认旧代码放行；没有在共享 GPU 上制造故障。
- 修复：转换前校验三个设备 CSR 向量均为连续一维 int32；CPU host metadata 要求一维，保留已有 host 值域与 Graph 地址稳定检查。不读取设备数据、不添加 D2H 同步，检查只发生在每步 prepare 而非每层执行中。

**3. 已修复：XQA 不兼容页大小发现过晚。**

- 位置：[原构造器](https://github.com/MingASA/xllm_quant_kvcache/blob/a00beb76a8a4b7aabc36ae34ea212371e8518418/xllm/python/attention/xqa_decode.py#L59)；[新构造器](https://github.com/MingASA/xllm_quant_kvcache/blob/ba87794dad1a8c6f7244b6168dbc8b43d8450645/xllm/python/attention/xqa_decode.py#L66)；[服务配置检查](https://github.com/MingASA/xllm_quant_kvcache/blob/ba87794dad1a8c6f7244b6168dbc8b43d8450645/xllm/core/framework/config/kv_cache_config.cpp#L249)。
- 问题：page size 限制到首次 Decode 的长度校验才触发，非法几何已经可能分配设备资源；page size=0 还可能先触发除零。
- 修复：正 batch、支持页大小和 page-aligned context 在设备查询／分配前检查。服务 FP8 `auto|xqa` 配置和缓存 bind 同样提前限制 page16/32/64/128；显式 FlashInfer 与原 BF16/int8 配置不受该 XQA 限制影响。CLI 中英文参考同步更新。

**4. 已修复：缺少 Decode Plan 时可能先写入 KV 再报错。**

- 位置：[execute](https://github.com/MingASA/xllm_quant_kvcache/blob/ba87794dad1a8c6f7244b6168dbc8b43d8450645/xllm/python/attention/flashinfer_native_kv.py#L225)。
- 原入口仅检查 metadata，随后 Writer 运行，再访问可能为空的 XQA Plan。现在先确认所选 Decode Plan 已准备，错误时在 Writer 前失败；新增回归检查 Writer 未调用。正常 Prefill／Decode 的算子参数和 Scale 路由不变。

**5. 未解决的合并验证门槛：完整 native build 与 SM90 集成。**

- 位置：[新 C++ 配置测试](https://github.com/MingASA/xllm_quant_kvcache/blob/ba87794dad1a8c6f7244b6168dbc8b43d8450645/tests/core/framework/config/native_fp8_config_test.cpp#L29)、[现有设备测试](https://github.com/MingASA/xllm_quant_kvcache/blob/ba87794dad1a8c6f7244b6168dbc8b43d8450645/tests/python/test_native_fp8_cache.py#L174)。
- 本机缺少 glog/gflags/gtest 与完整 native 构建环境，3 项新增 C++ 配置测试未执行；SM90 Writer/Query/Replay 测试仍跳过 3 项。需要构建并运行 `native_fp8_config_test`，以及一次支持环境下真实 Executor／Graph／Prefix 集成回归，才能建议直接合并。
- [默认 Docker](https://github.com/MingASA/xllm_quant_kvcache/blob/ba87794dad1a8c6f7244b6168dbc8b43d8450645/docker/Dockerfile.cuda#L12)使用 Torch2.7.1、Python FlashInfer0.6.14，历史实测则为 Torch2.10/cu130、FlashInfer0.6.18.post1。兼容性尚未证明；没有据此断言 Docker 必然不兼容，也没有擅自升级依赖。

### Minor (Nice to Have)

已完成：

- CUDA 特性测试在缺少可选 Triton 时显式跳过，不迫使其他平台安装 CUDA 依赖；独立 Host dispatch 测试保持可运行。

- 统一新 Python 运行代码／测试的格式，展开紧凑签名和调用，增加 Plan 容器、可空 Plan 与 Scale kwargs 的类型。
- 将文件局部长度校验私有化；不可变几何只在构造／绑定边界验证，不再逐步重复验证。
- 复用已经读取的 host indptr 列表，去掉额外 CPU 标量 `.item()`；page size 只在建立新 bucket 时获取。没有据此宣称实测延迟收益。
- 移除永远为 False 的 VMM 成员和重复字节求和；运行信息保持既有字段，并以实际 FP8 页 `nbytes` 计数。

## Validation

- 旧实现上的新增回归：**18 failed**，覆盖跨层／空页／不支持的页大小、六种 CSR 布局和七种非法 Plan 几何。
- 最终：**39 passed、18 subtests passed、3 SM90 skipped**。真实 CPU Tensor 验证布局、别名、字节数、Graph metadata 地址契约；CUDA Kernel 入口被替换的测试不声称设备算子正确性。
- 新增 3 项 C++ 配置测试：XQA 页大小、其他后端兼容性、Scale／Backend JSON round trip。已注册 CPU test target，但未执行，不计入通过数。
- C++ clang-format20.1.6、Python Ruff check/format、语法和 whitespace 检查通过。主线扫描 3,263 个非第三方文本文件，本机／私有服务器路径零命中；无临时脚本、实验输入／记录或环境信息进入实现分支。
- CPU 环境版本和原始日志 SHA 见 `VALIDATION.json`；完整失败／成功日志保留在本地及 E 盘。数值源码等价证据见 `NUMERICAL_SOURCE_EQUIVALENCE.json`。

## Recommendations

保持本轮边界修复，不扩展新的 Attention 算子、配置搜索或完整 Benchmark 矩阵。先完成上述有限的构建／设备回归门槛，再决定合并；不把历史模型成绩记作新提交已经执行的成绩。研发分支及原始失败继续保留。

## Assessment

**Ready to merge? No — 代码问题已修复，完整 native build、SM90 集成及默认依赖兼容性仍待验证。**

可继续以此分支进行正式 PR 审查。主线保留必要的 FP8 运行实现、容量／存储／Scale 基础设施及正式测试，Q4 与研究资料继续隔离。本轮没有更改数值公式、冻结 Scale、正式评价输入或结果，也没有新增性能收益结论。
