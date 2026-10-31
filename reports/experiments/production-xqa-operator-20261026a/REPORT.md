# XQA首次接入：算子失败，模型未加载

confirmed：`prod-quality-xqa-pilot80-r0-20261026a` 在H800/GPU0的单层算子Graph参考处失败，10240/10240输出为NaN。真实源码d68f5500，FA2 BF16-Q/FP8-KV prefill与SM90 XQA decode、PDL开启；测试把末页未使用区域写成FP8 NaN。此任务无14B模型加载、无80条预测、未进入BF16 arm，不能作为质量或性能结果。

50份原始证据、不可变ticket/request、终态和日志已只读镜像并逐SHA验证。supervisor只回收本方容器，driver确认释放，实际占用0.045218875 GPU-h计入额度。新出现的GPU4他人占用未触碰。

hypothesis：SM90 XQA对未使用V lanes的NaN不安全，可能在概率mask后仍由0×NaN传播；PDL等因素尚未由控制排除。此前FA3/FA2路径通过NaN尾页测试，不能据接口相同就认定所有kernel契约相同。

新修复候选会在第一次绑定时清零全部新分配payload，拒绝替换存储、同指针rebind保留已写入的有效/前缀KV。真实生产的后续回收页残值为有限值，正式算子参考将用极大有限残值验证mask，同时保留同一Graph/PDL下单独改变K或V尾页为NaN的诊断，再恢复有限值并复验。误差门槛不降低。只有这一生产初始化契约与真实算子检查通过，才允许加载模型。

这些修复和原因验证尚未H800通过。原失败终态及源码/缓存/日志保留，不覆盖、不重启本run。下一次必须是新提交和新run。
