# 当前最高研究任务：最新版官方xLLM的原生KV增量价值（2026-11-02）

用户最新完整指令覆盖旧优化BF16主基线与旧阶段终点。固定官方xLLM-AI/xllm main为261274e5f45449e98d7740144c3a958a29f1fd3e，未修改A0工作树保留。实际A使用721afdabc1de1af8e594b4da3e6fa926def50821：只为官方源码中仅NPU定义的helper调用补两行USE_NPU编译保护，原CUDA构建失败保留；不含KV扩展，不改普通BF16执行。C从此共同基线选择性迁移原生FP8/NVFP4，不整体merge旧分支。A/C采用相同Qwen3.8-27B只读权重、GPU、真实请求/采样和可比资源配置。官方正常Python Executor/FlashInfer BF16为A；旧FA3+XQA为历史D，不跨版本算加速比。仅公共Scheduler/Graph/Prefix/VMM变更需要隔离提交并做最小B回归。

优先真实ShareGPT原512测量+16独立warmup文本，27B官方tokenizer重核，合理有限C16/32/64/128档位由pilot容量选择。再补一个中长context、自然decode充足/真实容量敏感负载，不以旧41K输入160输出单独代表方案。质量复用GSM8K/LongBench/工具基础设施，补必要同版本证据。正式前核混合批次真实dispatch/Graph覆盖、跨请求prefix reuse与GDN恢复、Elastic VMM/page-copy/scale/Graph地址。后端参考当前FlashInfer/vLLM/SGLang的实际成熟路径；SM90无Blackwell FP4MMA，不能以整cache BF16 shadow或隐藏fallback冒充native。

最终代码、A/C实机数据/必要B、算子/容量/硬件结论、公共机制兼容、集中精简Markdown图表与合入建议全部完成后才结束接管。负收益可作为完成结果；不调到正数、不无限grid。沿用共享安全/一次性派发/原票据对账/同累计账本，GPU启动后Luna300秒有界只读监控，main持续接管。私有授权不进公开Git；GPU工时/到期自限若需新有限窗口，须记录当前直接授权、旧累计账本和冻结执行范围，不重置budget_id。
