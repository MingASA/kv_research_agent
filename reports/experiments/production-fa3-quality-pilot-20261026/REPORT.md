# 同生产FA3候选的80条质量pilot

**原始预测独立验收与重评分通过；候选质量尚未通过生产验收。** 两臂各80条（GSM64、LongBench八任务各2），root COMPLETE及两臂封存SHA一致，已验证源记录、prompt SHA、实际usage、FP8-Q/KV backend、真实Graph、模型权重和源码。151个镜像文件逐SHA一致。此批用于检查接口和明显退化，不能当完整1719评估。

| 指标 | BF16 | FP8候选 |
|---|---:|---:|
| GSM8K strict EM（64） | 41/64，64.06% | 42/64，65.63% |
| GSM8K flexible EM（64） | 57/64，89.06% | 57/64，89.06% |
| LongBench八任务macro（每任务2） | 0.43427 | 0.35085 |

数学总分一致不代表逐题完全一致。独立paired flips及原始源索引在audit中，strict与flexible分别报告。LongBench macro差约−8.34个百分点，有长文和代码风险信号；每任务仅最短和最长两条，不能外推为全任务平均退化。NarrativeQA约.231→.133、HotpotQA .625→.286、RepoBench-P .485→.225，其他任务有持平或上升，须查看完整样本与长度分组。

LCC和RepoBench两臂均触及冻结benchmark的生成上限，官方代码相似度仍按原协议评分，不能把length终止单独当模型错误或强行改上限。

模型权重实际逐shard读取后的identity为 `cf67b8e03cf159dd6017e36da84c28e143e59ce766261db2f6c97aabca7aae73`，两臂一致；同ceff binary、Python运行时 `68fedda1`，同FA3、FP8 Q=1/K=.125/V=.0625、prefix=true、Graph、显存比例0.8和128/10240/200调度，质量并发8。未触碰他人资源。

下一步已冻结剩余1639条（GSM1255、每个LongBench48），与此80条精确不重叠，合计覆盖原1319+8×50。先验证此父批COMPLETE、封存、audit及完全相同的实际引擎配置，再生成剩余部分，最后按原始源索引合并并统一评分。fragment本身不能宣称全量完成。该配置的性能扩展先等待质量证据。

证据：`reports/operations/20261026T075846Z/quality_independent_audit_b1.json`；原始镜像 `results/mirrors/prod-quality-fa3-pilot80-r0-20261026b-c1/files/SNAPSHOT.json`；复算工具 `scripts/audit_quality_run.py`。pilot不建立完整非劣性，也不代表用户已经接受该损失。
