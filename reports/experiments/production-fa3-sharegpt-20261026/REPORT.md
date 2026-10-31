# FA3生产候选：真实ShareGPT C128 首次服务测量

**结论：发现值得继续验证的服务收益；尚不能据此决定完整生产支持。** 两臂各512测量+16预热，原始HTTP payload、完整SSE/usage、actual token、Graph及resident KV已独立验收。FP8请求吞吐约+18.7%，实际平均输出几乎相同；质量、干净复测与原项目其他生产负载仍待补足。

**运行状态必须保留：root为CONFLICT，未写COMPLETE。** 安全监控在FP8最后一个profiling响应约9.000秒后触发归属/预约检查，随后只停止我方容器、确认驱动释放。事后只读发现两臂measurement.json都已完整封存，全部56个封存证据文件SHA一致，完整79文件镜像逐SHA一致。独立复算请求与统计通过，因此保留为“measurement integrity verified / runtime conflict retained”的恢复数据。旧监控对已退出PID直接报foreign是可复现的竞态缺口，但该次触发的具体PID没有记录，不能倒推确认为误报或宣称完全无共享干扰。后续复测仍有必要；不重复启动本run。

## 部署与请求

Qwen2.5-14B-Instruct，单张H800/SM90/GPU0，既有ceff C++ binary SHA `277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3`；Python生产源码 `7831a87402256dff2ed813ccb326b11be316d6f1`。同FA3：BF16-Q/BF16-KV vs FP8-Q/FP8-KV，Q=1.0、K=.125、V=.0625；没有完成这些scales的生产质量认证，不能称纯KV dtype因果对照。

启用prefix cache、chunked prefill和真实CUDA Graph capture/replay，max_seqs128/token_batch10240/admission200/memory_utilization0.8。两臂独立缓存、同UUID、独立私有容器与资源上限，切臂前再次稳定空闲120秒。GPU4–7有其他用户任务，未使用、未干预。

来源94145条完整ShareGPT记录，冻结1007个最终真实user请求及其已有上下文，按相同顺序执行唯一528请求。自然EOS、greedy、生成上限1024，未强制输出长度。到达模式明确是closed-loop C128，不能称原生产时间线。实际测量输入均值1120.586 tokens；池的CPU估计与实际执行子集分别记录。数据SHA `810415584dc2c751f0cf0e895cd7c4f1418ad338c03f70781d9abfc4e0df44a1`。

## 实测

| 指标 | BF16 | FP8候选 | FP8相对变化 |
|---|---:|---:|---:|
| 请求吞吐 | 2.515 req/s | 2.986 req/s | +18.71% |
| 实际输出吞吐 | 1352.763 tok/s | 1603.916 tok/s | +18.57% |
| 平均实际输出 | 537.809 tok | 537.156 tok | -0.12% |
| TTFT P50 | 8564.207 ms | 237.254 ms | -97.23% |
| TTFT P95 | 14514.743 ms | 5167.681 ms | -64.40% |
| ITL P50 | 74.002 ms | 73.880 ms | -0.16% |
| ITL P95 | 82.477 ms | 81.975 ms | -0.61% |
| E2E P50 | 44539.937 ms | 37225.787 ms | -16.42% |
| E2E P95 | 80735.507 ms | 77885.800 ms | -3.53% |
| E2E P99 | 87536.615 ms | 79891.321 ms | -8.73% |

两臂测量成功率均512/512，实际输出总数275358 vs 275024；stop/length为455/57 vs460/52。唯一配对session及服务端prompt counts摘要完全一致。TTFT包含排队；ITL为逐请求平均间隔，不能据其P99推断每个token的最坏stall。

同约34.62GiB resident KV预算，bytes/token/layer4096→2112，pages/layer1477→2865，分配容量189056→366720 tokens，约1.9397倍。不是最大真实承载并发或无拒绝容量已证实。profiling NVML显存采样峰值66825→67063MiB，更多页使预留总量接近；这不否认压缩。

吞吐按真实observation窗口203.552→171.470秒复算，未用benchmark_duration替代。稳态每GPU完成请求/s改善18.7%；只按该完成请求率估算，每请求GPU时间降低约15.76%，尚未加入质量门槛。整项实际占卡0.313035GPU-hours，包含算子诊断JIT、模型加载、Graph及回收，不能当标准生产cold-start。

## 含义与下一步

当前差异主要表现在TTFT和整体完成速率，ITL中位数几乎相同。容量/排队、prefill、先进FP8算子均可能参与；没有scheduler/KV occupancy或硬件HBM计数器，不能宣称已经定位原因。NVML memory-controller utilization不等于测得GB/s。

需要同生产FA3/FP8-Q配置的GSM8K与LongBench质量检查。已准备GSM64+八任务×2的分层pilot；只能找严重退化，不能当完整1719质量通过。之后补干净repeat、低压力服务、BurstGPT与Mooncake连续到达/前缀窗口，触及BF16容量开始限制的生产区间，再判断收益、维护代价与适用范围。

示例延迟门槛曲线已在机器可读audit中保存；如TTFT≤5秒且逐请求平均ITL≤100ms，满足条件的完成请求率约0.712→2.782req/s。这是工程展示阈值，非用户已批准的SLO，也不含质量，不称完整生产goodput。

证据：`reports/operations/20261026T075846Z/production_sharegpt_independent_audit_v1.json`；可复算工具 `scripts/audit_production_run.py`；原始只读镜像 `results/mirrors/prod-sharegpt-fa3-c128-r0-20261026b-c1/files/SNAPSHOT.json`。失败终态与封存原件均保留；没有重写远端成功标记。
