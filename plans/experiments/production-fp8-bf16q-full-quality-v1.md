# BF16 Q / resident FP8 KV：完整生产质量验证

2026-10-27。前置算子prod-bf16q-decode-op-r0-20261027a实际H800数值/Graph128/driver释放及55文件独立审计均已通过；audit_a2补充原始推理源码、binary/image/配置绑定，未重跑GPU或改写a1。

新run拟为prod-quality-bf16q-full1719-c128-r0-20261027a，仅新FP8候选一臂，source becc8dd7b19ae5632428c8e6abe54b6e28fed369，原算子78004505保持干净。控制器与GPU worker在模型加载之前校验完整operator audit、原source SHA map、binary/image/精度配置，推理和算子代码逐字节相同才复用门槛，不重复GPU算子job。当前仅冻结/CPU/只读预检，不预写模型结果。

样本、顺序、prompt构造、generation上限、自然EOS、原官方scorer及模型保持原完整1719（GSM8K1319+八LongBench各50），生产Graph/prefix/clientC128/maxseq128/tokenbudget10240/admission200/memory.8；固定K=.125/V=.0625。BF16Q FA2 prefill+decode、resident FP8无完整BF16shadow。新质量profile/实际日志必须明确BF16Q，不准借FA3/XQA字符串绕过旧16上限或污染历史manifest。

现有现代BF16 FA3+XQA C128参考由prod-quality-bf16xqa-full1719-c128-r0-20261027b原始独立完整审计复用；不重新生成BF16、不合并成更多独立样本。新FP8和保留参考的GPU时间、调度/batch/cache历史可能不同，报告每题配对变化和统计范围，不声称隔离Q因果。此前FP8-XQA结果作为另一部署候选对比，不能把FP8算子fixture误差作为模型任务分数。

资源上限单空闲H800、5400秒/1.5GPU-h/5GiB，既有累计9.4930775161/第一12GPU-h不重置；派发前与运行时仍须fresh120秒空闲/共享预约归属/UUID/非root私有容器/独立watchdog。新资产45项SHA/实际frozen policy、完整源码84与控制器205 CPU检查，真实candidate命令的argparse/env/prefix/C128已CPU验收；CPU不是GPU/model通过。

验收须原封存终态、实际模型weights/config/scorer/输入和source SHA、真实模型Graph capture/replay、每题原响应/usage/自然停止/超时及官方重评分。报告GSM严格/宽松、LongBench逐任务/宏平均、每题变化/代码相似度局限、生成长度与cap命中；不得把quality worker complete提前当质量接受。完整模型数值/并发仍由这次真实运行检验。

依据实际质量取舍决定是否继续必要的真实ShareGPT/ToolAgent目标服务验证；质量/代码风险无实际改善则停止该候选服务矩阵，更新支持结论与下一具体修复。用户没有接受数字质量门槛，1pp只作已有工程提案，不捏造已通过门槛，不为填矩阵重复旧GPU实验。
