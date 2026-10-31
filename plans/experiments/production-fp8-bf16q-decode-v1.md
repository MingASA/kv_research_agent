# 可部署候选：resident FP8 KV，保留 BF16 Q 解码

2026-10-27。目标是改善现有部署的质量/服务取舍，非固定 FA2 论文对照。当前先进 FP8 XQA 在内部量化 Q；KV range 探针没有测 Q，也不能据此归因质量损失。安装版本 FA3 的 FP8 KV 要求 FP8 Q，故选择实际支持 BF16 Q 的 FA2 Tensor Core 路径作一个明确候选。主生产参考仍为 BF16 FA3 prefill + XQA decode，不降低参考后端。

## CPU 已确认与限制

- C++ cuda_kv_cache_admission.h 只约束执行模式，无 batch16 硬上限；Python Graph bucket 支持128。旧质量工具16限制不能凭改planner字符串绕过，须先实机证明。
- 安装版 decode.py 与上游 v0.6.18.post1 原文件 SHA 相同。上游 FA2 compute_qk 将 FP8 K 转为 DTypeQ，Q 以原类型从 shared memory 装载，BF16 MMA/FP32累加；不是 FP8 Q Tensor Core。H800 实际 header SHA 仍待新算子门槛核对，不能从 Python dtype 推断所有已安装内核行为。
- 新 source78004505（完整SHA见冻结manifest），独立 opt-in XLLM_FLASHINFER_FP8_BF16Q_DECODE=1，明确 FA2/flashinfer/BF16 model Q/resident FP8。全体payload只在首次bind初始化为有限值；同storage重绑定保留prefix，替换storage拒绝。无完整 BF16 cache shadow，无全模型 BF16 workspace。
- 78项 CPU、576种旧配置结果/错误逐项相同、控制器203项通过。CPU不代表 CUDA 运行成功。

## 唯一初始 GPU 门槛

计划 run prod-bf16q-decode-op-r0-20261027a；H800/单空闲UUID，1800秒/.5 GPU-h/3GiB，仍计入既有12 GPU-h自限，当前已用9.4507125811。固定 binary277bd8...、模型config/镜像/源码/input SHA，原资源准入/120秒空闲/归属与已知预约/容器隔离/watchdog不变。尚未launch；新结果不得预写成功。

先核对实际安装 prefill.cuh、mma.cuh、decode.py 与本地下载的对应官方tag SHA，再执行原生 prefill/decode/partial-page/noncontiguous-Q/小Graph数值门槛。对 BF16 Q 的 FP32 dense reference 相对L2≤.01，不能用FP8 Q参考掩盖内部转换。随后实际同一Graph128包含 writer+attention，以128/126/1/127 live rows、257/130上下文、空indptr padding、非连续QKV、动态metadata和固定resident指针测试；page0不被live使用，padding writer只写零，输出只消费live rows。重复bind不能清空prefix。

这是与生产 envelope 相符的数值/生命周期验证，合成tensor不称真实请求或端到端收益；本步不加载14B，不运行AIPerf。任一kernel/Graph/精度/源码门槛失败，保留失败与原始日志，只针对具体证据修复或停止，不自动退回eager/小batch，不重新启动状态未知的run。

## 后续取舍

仅实际门槛通过后，才扩展明确的质量manifest/Graph128工具contract：同1719原题及原官方scorer、自然EOS，复用已审计先进BF16参考而不是无理由重跑。保留每题原响应/生成长度/正确率与LongBench/代码损失；一个变化配置不证明Q因果，也不把减少损失当已接受质量。

质量有实际改善、部署价值成立时才选择必要的目标服务验证：真实ShareGPT饱和服务和能回答低压/队列问题的原连续ToolAgent请求时钟，保留生产Graph/prefix/token budget、原自然停止语义与现代BF16参考。报告成功goodput、尾延迟、实际生成量、容量与成本；BF16-Q FA2可能比XQA慢，须公开取舍。若质量或系统取舍没有改善，停止该候选，更新生产支持结论和下一具体修复；不扩受损配置矩阵。
