当前最新结果（2026-10-27T10:32:51.879329+00:00）：算子/Graph/页生命周期完整通过并独立52文件验收；额外1539MiB+4B，仅该记录形状CUDA中位-.5967%/wall-.5545%，停止完整模型扩展。报告reports/experiments/production-prefill-workspace-20261027/REPORT.md。以下保留执行前冻结设计。

# 生产prefill修复：有界BF16工作区与FA3，保留resident FP8和XQA Graph

当前状态：本地已实现、提交与CPU验收；H800算子/数值/Graph尚未验证，不是服务性能或质量结果。

已独立验收的组件诊断c显示FP8 writer设备时间约占.40%，prefill attention约39.60%；一组相同记录元数据（window index2/forward0，1seq/510query/1022KV）的attention累计BF16 1.749→FP8 3.534ms。17/18个真实prefill批次组成不同，不能从aggregate比例推算FP8因果损失。该诊断只支持优先验证可部署的prefill修复，不铺微基准矩阵。

独立业务分支production/fp8-fa3-prefill-workspace-v1，source8af6e702b6686b1402b1a42104b0b84eec98222b，位于/home/mingasa/projects/xllm_quant_kvcache-agent-prefill-workspace-20261027。每层先写resident FP8，再只对本批唯一活跃page解量化到一份共享BF16 K/V工作区，以FA3 BF16-Q/BF16-KV prefill执行。decode继续原XQA/内部FP8 Q/真实CUDA Graph。没有新增外部Q量化，不维护48层BF16 shadow；固定K=.125/V=.0625、prefix/chunked/128seq/10240token/200admission/.8保持。

显式opt-in XLLM_FP8_PREFILL_WORKSPACE_MIB=1536；预分配在backend构造、resident KV绑定和C++容量估算之前，元数据也一起预留。H8/D128总保留1539MiB+4B，使用一组稳定view，physical-page-ID不变；页形状超过上限在服务前拒绝，不扩容/回退。每批device原子收集去重活跃页，每层仅转换活跃页，无服务路径GPU→CPU计数同步。真实14B实例必须核实容量估算日志顺序、实际page数/全量额外显存/峰值，不能继续拿旧1.94倍容量当此新配置净收益。

本地CPU：88passed/1CUDA skipped；576个旧profile组合逐字等价。2个Triton kernel离线SM90 codegen通过，但本地Triton3.7.1不同于H8003.6，不能当实机通过。首次masked FP8 load类型失败已保留并修正为byte alias+bitcast。控制器192项通过，冻结24资产SHA和实际frozen policy通过。首次封包窗口索引错误被CPU拒绝；第二次Git bundle未使用命名引用被拒绝、geometry-only目录保留；最终新v2包成功，均无远端写/GPU预留。

下一唯一operator-only候选prod-prefill-workspace-op-r0-20261027a，包results/prepared/packages/prod-prefill-workspace-op-r0-20261027a-v2/package.json。仅单GPU0/.5GPU-h/1800秒/3GiB，先全部120秒空闲/无他人占用或已知预约/UUID/容器可见性/共享CPU-RAM-disk/watchdog门槛。实际dense FP32 reference、partial/replan/非连续Q、unused-page NaN/finite/padded bucket、CUDA Graph capture+replay；额外共享页去重/FP8 key stride2/未活跃页不转换/跨层稳定复用/下批active替换均须通过。失败不加载14B/AIPerf，不以fallback过门槛。

只测上述原生产记录形状，随机fixture显式不是真实activation。两profile在执行前固定，同resident pages/Q，8次反序交替；计时含planner/active收集/48层materialize+attention，另报wall/CUDA原样本。不是完整model forward，不是HBM测量，不是服务收益。若额外数据流量/launch成本不抵FA3收益，保留负结果并停止候选；通过且有价值后才按新manifest做真实质量与目标生产低压端到端，不能拿算子通过推断质量相等。

原生tool b已完成并独立验收，未修改旧source/API或GPU结果。流式API已有CPU复现的参数排序增量缺陷，完整JSON实验不能认证流式。两臂182/258官方参数匹配，FP8尾部重复到1024上限保留；有限短cohort无整体服务收益，不能将此说成稳态峰值。正式报告reports/experiments/production-native-tools-20261027/REPORT.md。

初始12GPU-h未重置。任何新run先只读旧票据/账本/远端；SSH丢回复不重发launch，按原run恢复。无GPU继续CPU研究。用户质量/SLO阈值仍未被假定接受。
