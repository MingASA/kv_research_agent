# BF16-XQA完整质量参考：生产基线改变后的必要补证

proposal：ShareGPT先进BF16参考的服务配对已独立验收，FP8增量请求吞吐+32.9647%、输出+30.5593%。新的BF16 FA3 prefill+XQA BF16 decode不能沿用旧FA3 decode的质量认证；只生成新BF16配置的原冻结1719参考，不重复已验收FP8预测。不是旧缺失片恢复，也不修改旧得分、终态或评测答案。

source11e7cddbfe33eb21d209c7f48f5e461ac56545eb，只改质量工具/证据和测试；全部XQA_SOURCES与已测1c2be159逐字节相同，整个xllm推理树未改。新bundle单独命名，性能bundle和正在运行的两条曲线不变。现装FlashInfer0.6.18.post1/ceff binary/只读Qwen14B保持真实来源。

完整恢复原72份shard逐SHA与全局索引，GSM1319+八LongBench各50；NarrativeQA50完整记录/49prompt不删。原顺序/5shot/官方评分器/自然EOS/greedy/原任务生成上限、C8/prefix/Graph/chunked-prefill/128seq/10240tokens/200admission/.8保持。每条源记录、prompt/tokenizer/usage/finish和完整权重identity独立审计。新BF16质量与原FP8质量分开记录实际源码和配置，比较不声称整个runtime相同或C8认证全部生产batch。

先在受监督私有容器CUDA隐藏实际import并调用官方scorer，后真实BF16 GQA40/8、非连续Q/partial/replan/有限尾页/reserved page0 padded Graph门槛，误差不降。失败不加载模型。通过后才加载14B、真实Graph capture/replay、所有1719预测和封存。全量质量文件与driver释放均需独立只读SHA审计，不预写通过。

run prod-quality-bf16xqa-full1719-r0-20261027a，GPU3精确UUID仅候选。7200秒/2GPU-h整项硬窗口含JIT/加载/Graph/失败/回收，输出5GiB。当前已结算4.678642933GPU-h、两曲线预留4，本项再预留2后最坏10.678642933≤初始12，不重置/扩大。仅freshS0+全部选中UUID120秒稳定空闲/归属/预约/CPU-RAM-磁盘/独立watchdog通过才唯一启动；未知SSH先只读恢复。

CPU：源码76passed/1CUDA skipped，控制器162passed，实际冻结policy和全部30资产SHA通过。CPU不是H800质量成功。FP8原LongBench损失/用户质量界未接受继续披露，最终按真实服务压力/质量/维护成本决定支持范围。

2026-10-27 completed: root cleanCOMPLETE/driverreleased,108files43709322bytes independently SHA-verified; original1719/reference officialrescoring and retainedFP8 comparison passed. NewBF16 macro50.125239%, retainedFP848.752646% (−1.372593pp), GSMflex+.454890pp, LCC−5.38pp. This is crossrun/differentcachehistory, not new pairedFP8 generation or user quality acceptance. Reportproduction-bf16-xqa-quality-reference-20261027.
