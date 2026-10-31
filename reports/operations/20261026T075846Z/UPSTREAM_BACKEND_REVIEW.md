# H800生产后端版本复核（2026-10-26）

confirmed_source：上游最新稳定发布为2026-10-23的0.7.2.post1；本项目现装0.6.18.post1。已从本地有网侧下载固定tag的8个源文件并记录SHA，没有H800安装或外网操作。[官方发布](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.2.post1)

`csrc/xqa/mha_sm90.cu`与实际H800已镜像文件逐字节一致，SHA1edd5708fafae8667c9b6ad3119c00d373ad53104bb4f696cb5409423cc7b068。0.7.2.post1的`is_fa3_backend_supported`仍拒绝BF16-Q/FP8-KV，因此仅升级包版本不会让这一FA3 dtype组合自动可用。[固定版本源码](https://github.com/flashinfer-ai/flashinfer/blob/v0.7.2.post1/flashinfer/utils.py)

unverified：核心translation unit相同不证明整个kernel二进制或性能相同；传递头文件、wrapper、JIT flags、workspace和调度需进一步核验。不能据此声称升级无收益，也不把Blackwell/Rubin/MoE/MLA的改进直接移植到H800/Qwen14B GQA。本次已冻结的质量批次不改环境；后续升级只在存在相关生产改善证据时采用全新隔离依赖和run，经过相同真实算子、Graph与质量门槛，不覆盖共享安装。

当前XQA候选已真实通过SM90算子/Graph及完整每臂1719质量完整性，GSM flexible+0.4549pp、LongBench macro−1.3358pp；质量代价未被接受，无新AIPerf性能结果。完整补测及三路恢复已结束，报告production-xqa-quality-full1719-20261026。版本问题作为部署变量保留，既不盲升，也不把旧版本锁成永久研究结论。

机器可读对照：flashinfer_upstream_072post1_source_audit_v1.json。这里只做源码比较，无新增GPU实验。

补充只读比较：SM90核心直接包含的13个项目文件及xqa_wrapper.cu共14份，上游v0.7.2.post1与H800现装版本的SHA全部一致。没有远端写入或GPU工作。证据flashinfer_sm90_direct_headers_comparison_v1.json。传递依赖闭包、Python/JIT dispatch和编译工具链仍未完全等价验证，不能将14份一致扩大为全后端/性能完全相同。
