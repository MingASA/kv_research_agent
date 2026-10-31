# 后续候选规则：先看真实范围，不按评测答案选scale

冻结2026-10-27T10:58:27.992960+00:00，尚未查看heldout模型activation导出；父diagnostic a正在模型初始化。规则机器可读文件reports/operations/20261026T075846Z/range_probe_candidate_scale_rule_before_export_v1.json。

对K/V各自：若真实有限值饱和数为0，维持baseline、不重跑未改变配置；若存在饱和，以全部记录layer/head/phase的最大有限absmax加10%余量，取2次幂覆盖scale，并只允许不低于当前K=.125/V=.0625。发现非有限激活/不完整证据/不安全scale立即拒绝，不用质量分数挑候选或往返搜索。

这是简单可维护global scalar的下一候选规则，不是校准效果结论。normal共同域内2次幂缩放不能改进相对精度；更大scale还会增加很小值的下溢。短聊天不覆盖32K/code或FP8生成反馈。更精细的per-layer/head只有实际kernel与部署收益支持才另案，不暗换attention路径。

只有配置实际改变且有范围依据才新manifest：按真实scales做数值/partial/padding/Graph门槛，必要的完整1719 C128/同模型同顺序/自然停止质量；再按目标真实业务端到端验收益。诊断在正式性能中关闭，所有负结果/flip/长输出保留。未设用户质量/SLO接受界，不把proxy或总分接近当上线认证。
