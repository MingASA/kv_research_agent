最新2026-10-27T11:59:38.595966+00:00：实际观测已完整独立恢复，原c FAILED保留；无clipping，scale不变，停止本global调参方向。正式结论/局限/原始失败/复算见reports/experiments/production-kv-range-probe-20261027/REPORT.md。以下为历史设计状态。

最新2026-10-27T11:33:03.782463+00:00：b完整128原始服务结果已独立恢复，但8个合法U+2028暴露JSONL分行错误，且旧顺序导致真实GPU统计未导出。原FAILED保留，83文件逐SHA。c只修复物理LF读取和先导出后审计，已冻结/CPU，尚无新scale。以下历史状态以本段和STATE为准。

最新恢复（2026-10-27T11:11:47.789991+00:00）：a实际小Graph/数值/模型admission/reset通过，但显式zero warmup被CLI拒绝，未发128正式请求；FAILED及73文件/driver释放保留。b只修复CLI构造，28资产/196CPU/实际完整CLI+远端入口SHA通过，fresh S0待完成后派发。source5e2c99f9不变，无真实activation或新scale。以下保留初始设计。

# 生产质量风险诊断：Graph内有界K/V分布聚合

目的：当前现代BF16/XQA参考下，FP8生产C128完整质量LongBench−1.7705pp，code/multihop风险尚未接受；不能盲调global scale。先检查固定K=.125/V=.0625对真实BF16 activations是否造成饱和/下溢，区分范围问题和FP8固有舍入误差。诊断不证明FP8真实生成质量，不认证新scale。

独立source分支production/kv-range-probe-v1，parent11e7cddb，目录/home/mingasa/projects/xllm_quant_kvcache-agent-distribution-20261027。默认关闭；仅untransformed native BF16 FA3+XQA/Graph生效，不改checkpoint/原算子/default paths/API/C++binary。单GPU TP1/48层/8KV heads/D128；每次实际writer前按层/phase/KV/head Triton聚合finite/nonfinite/clipped/normal/subnormal/zero/nonzero→zero和absmax/原值平方和/QDQ误差平方和。fixedscale QDQ为clamp±448→E4M3→原单位，保留整数计数和FP32 moments，共105216B，不保留activation/cache shadow。

CUDA Graph capture/replay内只写GPU计数；无.item/.cpu/文件输出或动态扩容。slots<=0（含reserved page0和padding）排除，非连续K/V stride支持。先在小GPU探针逐项对照Torch QDQ计数/矩，故意padding NaN/Inf、真实有效NaN/Inf、层/phase分离、真实Graph更新输入/重放、reset清warm且不换storage、真实控制线程reset/export握手。任何失败不加载14B，不放宽容差。

模型仍使用生产Graph/prefix/chunked128seq/10240token/200admission/.8，BF16 FA3+XQA。model readiness/原source32token admission/真实Graph后，控制器在无请求边界RESET并收到ACK，随后通过真正AIPerf发送已有独立heldout128原始自然聊天，C128/warmup0/cap1024/自然EOS/greedy保持；所有请求结束后EXPORT并ACK，再停止自己的模型。只统计执行过的有效KV写入，prefix复用的旧缓存不重复计数；admission后的prefix历史明确保留，逻辑prompt usage不等于writer token总数。

输入results/prepared/calibration-sharegpt-heldout128-20261027a2，来自未进入既有528服务样本的479条候选，长度四分位各32、SHA seed20261027；CPU估计143037prompt tokens、32–2980。消息、采样和自然停止不改，不读评测答案。只覆盖短聊天，不外推32K长文、代码、所有用户或FP8错误反馈的激活。数据为CPU冻结输入，GPU结果另存新run，不改旧manifest。

仅单BF16诊断新kind heldout_kv_range_probe，候选唯一run prod-calib-bf16xqa-heldout128-r0-20261027a；整项≤1800秒/.5GPU-h/3GiB，包括GPU探针/JIT/模型加载/Graph/请求/导出/回收。第一12GPU-h累计9.1379426631且无active/UNKNOWN，预留前再只读核实。必须每次fresh UUID/全部120s空闲/无他人占用或已知预约/容器隔离/CPU-RAM-disk/watchdog。未知SSH只读恢复同run，不发副本。

真实AIPerf完整raw和usage/Graph/backend/binary/config身份按原规则保留，既有完整质量的weights身份作历史参照，不冒称本诊断重新hash全部权重；计数完整1536group、48层相同实际call数/valid token、phase/KV/head覆盖、finite partition、sum-of-squares/L2与reset epoch/命令SHA全核对。诊断kernel开销存在，AIPerf时延/吞吐仅诊断完整性，不作为新headline服务收益。没有用户质量/SLO接受声明。

目前本地source99 CPU通过，控制器196 CPU通过；离线SM90 codegen通过但本地Triton3.7.1不同于H8003.6。后续仅按真实分布决定是否有价值生成候选scale；normal区2次幂scale的QDQ数学等价已CPU确认，不以改scale名字冒充精度修复。prefill工作区候选已按低成本证据停止，旧run与失败均保留。

2026-10-27T10:49:21.867649+00:00 source5e2c99f9已提交且干净；27冻结资产SHA/实际frozen policy/fresh S0通过，累计9.1379426631GPU-h/预留0。下一唯一受保护a，尚未派发，不预写GPU门槛或激活成功。
