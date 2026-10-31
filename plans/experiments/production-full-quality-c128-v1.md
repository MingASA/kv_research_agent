# 完整1719的生产C128配对质量验证

proposal：当前服务压测是C128，完整质量是C8；新的BF16-XQA参考对保留FP8为GSMflex+.4549pp、LongBench−1.3726pp、LCC−5.38pp，并有不同分批/缓存历史。为决定生产支持，需在实际服务的客户端C128包络做一次同完整输入的双臂质量验证。它既补真实batch数值边界，也避免把跨run的C8质量直接认证为C128。不是新增独立题目或重跑同条件。

source11e7cddb，推理引擎全文件与服务1c2字节相同；只更新控制器显式协议。BF16FA3-prefill+XQA-BF16-decode，对FP8FA2 BF16-Q-prefill+XQA内部FP8-Q-decode，K.125/V.0625。原完整1719记录/原顺序/5shot/官方评分器/自然EOS/greedy/任务cap、prefix/Graph/chunked-prefill/128seq/10240token/200admission/.8相同；客户端concurrency128不等于每个真实GPU batch固定128。按真实backend/Graph日志报告实际路径，不能悄悄降回C8。

run prod-quality-bf16xqa-full1719-c128-r0-20261027b，GPU3仅候选、顺序BF16→FP8。新的full_production_batch显式kind只允许完整1719、C128和双臂；旧C8 pilot/completion和仅BF16参考边界保持。先CUDA隐藏真实import/调用官方scorer，再各臂真实数值/partial/replan/非连续Q/paddedGraph与有限新cache初始化，门槛失败不加载模型。完整权重identity/源码/binary/data/scorer/prompt/usage/finish/Graph/封存/driver释放及官方重评分全部验收，不能预写质量成功。

仅因客户端batch及共同完整cohort改变而新增两臂预测；旧C8成功预测不计为被替代/新样本，不按已见答案调参。正式报告并排给C8与C128分数、逐题变化/任务macro和样本条件，保留实际source与cohort/cache历史；1pp工程界未被用户接受，质量损失不能用吞吐覆盖。

整项7200秒/2GPU-h、5GiB，含两次加载/权重读取/JIT/Graph/失败/回收。当前已结算6.102708744GPU-h、低速trace预留2，本项再预留2后最坏10.102708744≤初始12，不重置或扩大。固定许可仍有效；GPU3仍需fresh身份/归属/预约/120秒稳定空闲/容器可见性、CPU/RAM/磁盘和独立watchdog。未知SSH只读恢复同run，不发副本。

CPU171项控制器通过，实际冻结policy/30资产SHA/两臂命令C128/full/FA3-XQA与FA2-XQA验证通过。CPU不是H800质量成功；无GPU可用就继续现有结果验收/本地分析。

2026-10-27T04:36:14.616184+00:00 completed: cleanCOMPLETE/driverreleased,168files67729166bytes fullSHA/1719perarm independently rescored. GSMflex+.530705pp, LongBench−1.770520pp (pairedbootstrap95%[−3.4893,−.1667]pp), LCC−3.92pp. Qualitynotaccepted; formalreportproduction-bf16-xqa-quality-c128-20261027, no duplicateGPUlaunch.
