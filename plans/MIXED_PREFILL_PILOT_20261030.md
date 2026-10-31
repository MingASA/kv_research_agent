# 完成状态 2026-10-31T01:39:01.482665+00:00

单候选既定9个正式条件已得到原始结果并独立审计：质量T/U/V、压力WC/XC/YC、工具ZC/ZA/ZB；YC为真实FAILED31/32，不称9项全通过。连同真实S pilot与W协议失败共11次实际GPU尝试，原失败保留。全部资源/账本/监控已收尾，最终报告和目录见reports/synthesis/NEXTGEN_NATIVE_KV_20261030.md与results/archive/nextgen-native-kv-20261030/CATALOG-a1.md。

## 原有限计划与历史记录

Latest 2026-10-30T23:58:26.535950+00:00: YC原同容器处于native operator冷编译（Luna23:57:04Z GPU7 1047MiB/0，PID249780为算子，不是模型），其他卡0。按已观测NV4冷启动及剩余YC+三组工具/独立分析归档，将Luna300秒只读截止有限延至2026-10-31T02:00Z；主Agent持续、原累计24GPUh/同budget_id，未新增case。

# 单一混合调度候选的有限验证

更新时间 2026-10-30T19:48:45.238244+00:00。原研究目标保持 active；这是实际发现配置限制后的必要部署验证，不替换既有结果。

## 已有证据与候选

原配置在实际C++日志中为batch token budget2048、max_tokens_per_chunk_for_prefill2048、CP1、MTP0。DecodeFirstPolicy先扣除decode token，hybrid新chunk须按chunk stride对齐；剩余2047可能无法提交下一个长文chunk。F/H/P/Q应用采样长文running峰值1、waiting7或15，和该来源一致，但仅源码推断不作为已证实性能因果。

仅冻结一个候选 `mixed8192_chunk2048_v1`：batch8192、显式chunk配置2048、maxbatch16、GDN state slots32、max memory0.9、page64、Graph/prefix开启。源码模型仍03134b72，C++binary仍862c29ca/SHA dbc1c53a...f09；不修改内核或做参数网格。配置chunk是调度上限/对齐stride，redistribution行为以C++实现和实际证据为准。

原三格式1720、三种场景共9条服务观察、BF16工具R以及进行中的M/N全部保留，属于原2048配置。新profile不得沿用旧profile的质量快照作为正式性能资格。

## Pilot与技术门槛

S=`nextgen-bf16-qwen38-mixed8192-calibration-r0-20261030s`，47资产已冻结CPU policy/隔离导入/SHA通过，package SHA d30f9c3a178f51e0d407779f1596a49de3ff9118b4c7c00c1b60f9e9b35da6e2。仅BF16，单GPU7，1小时run上限；先完成20条已知答案/A-B-A/实际模型Graph，再同一冻结136真实校准请求（128ShareGPT+8heldout长文）、client concurrency4、原自然caps。GPU KV范围observer开启，仅用于无计时的候选资格和scale统计，不是正式性能或质量接受。

只有实际完成/release/SETTLED/完整SHA和独立校准审计后才可进入主测。新raw校准请求记录实际单调时钟；附加原容器/vars读者只读，不发GPU/模型请求、不写远端。独立 `audit_nextgen_headroom_pilot` 必须在至少2个长文client请求在途时找到同一采样：server running>=2，active FullKV pages超过任何一个请求在原cap下的最大page数（含padding）。不把client并发或不同时刻的两个峰值拼成通过。样本不能证明原子峰值、最大capacity或因果速度。

原pilot失败/未取到并发证据时保留原件，不启动主测、不重复旧ID；先分析并做有根据的最小修复。正式新profile request需绑定pilot proof SHA；执行器重新核原pilot票据/result/validation SHA、COMPLETE/释放、实际136/20检查、同一模型/binary/配置/设备和resident计数界限。CPU proof、FAILED原run或空洞accepted标志不能代替实际pilot。

## Pilot通过后的有限顺序

这些只是proposal，尚未冻结、没有ticket/INDEX。均同GPU7，按原安全门槛一次性启动；质量与工具评分保持原分母。

| 顺序 | 拟定run | 内容 |
|---|---|---|
| T | nextgen-bf16-qwen38-mixed8192-quality1720-r0-20261030t | 新profile完整1720 BF16 |
| U | nextgen-fp8fa2-qwen38-mixed8192-quality1720-r0-20261030u | 新profile完整1720 FP8/FA2 |
| V | nextgen-nvfp4-qwen38-mixed8192-quality1720-r0-20261030v | 新profile完整1720 NVFP4/FA2 |
| W | nextgen-bf16-qwen38-mixed8192-prefix32-r0-20261030w | 同冻结32真实重复文档/C16 |
| X | nextgen-fp8fa2-qwen38-mixed8192-prefix32-r0-20261030x | 同case FP8 |
| Y | nextgen-nvfp4-qwen38-mixed8192-prefix32-r0-20261030y | 同case NVFP4 |
| Z | nextgen-bf16-qwen38-mixed8192-tools32-r0-20261030zc | 同32原始工具任务/112轮 |
| ZA | nextgen-fp8fa2-qwen38-mixed8192-tools32-r0-20261030za | 同工具FP8 |
| ZB | nextgen-nvfp4-qwen38-mixed8192-tools32-r0-20261030zb | 同工具NVFP4 |

低精度scale需来源于通过的BF16真实校准并匹配后续完整质量；若数值未变仍保留来源。每个正式性能/工具包仍需其新profile完整1720独立评分，不减样本或抹掉失败。仅一个新服务压力case，不追加新聊天/长度/并发网格，不重试尚未修复的FP8XQA模型路径。若实际证据要求修复，保留失败并明确新版本/控制变化。

## 调度、预算和交付

主Agent负责启动/恢复/写状态；Luna每300秒严格只读、变化立即报，当前明确截止2026-10-31T01:00UTC，需要延长时由main先记录有限新截止与原边界。保持主Agent活动。所有新GPU只经 launch_guarded_once；proposal只STATE/NEXT，实际原ACK+ticket+ledger后才首插INDEX，禁止复现L的history竞态。未知SSH只读旧run。

同一固定私有授权/同budget_id/累计账本，Agent24GPUh自限不清零。原M/N/S均已终态释放并独立闭环，S同采样驻留资格通过；T原once ACK/RUNNING，U/V已冻结未派发。新profiling全部单卡7；GPU-hours按实际账本与当前预留检查，不能把旧SETTLED行reserved求和当当前预留。其余用户资源、环境、路径及预约不扩大。

最终集中报告应分别列原配置与新候选的质量、服务、工具、物理/活动容量和代价，不合并成同配置结果。所有旧原始失败与benchmark口径问题保留；源码/bundle/图表/本地SHA归档和最终资源收尾后再做完整目标审计。未验项目明确标未完成，不能提前关闭goal。

更新：T已独立完整1720评分，U真实质量生成中。W52asset/ZC83asset冻结未GPU；原Z只在CPU准备时因 copied tool_source manifest无files子树失败，已核无ticket/once/ledger/history，INVALID原件保留，使用新ZC，GPU有限9项数量未变。实际tool source为results/prepared/bfcl-multiturn-native-source-20261028a，全30文件SHA已核。

最新：T/U已独立1720并配对新baseline；V原900秒operator在约11分钟冷编译后通过，实际模型20known/A-B-A/Graph已过且quality推进。后续只剩本计划W/X/Y/ZC/ZA/ZB六项，不增加网格；根据已观测冷编译及最终分析/归档/资源收尾，Luna只读有限截止延至2026-10-31T01:00Z，卡数/预算/授权原样。


更新 2026-10-30T23:18:18.864123+00:00：S实际资格、T/U/V各完整1720评分均已通过独立原始审计；新profile总5160。W原FAILED并释放/SETTLED（109SHA），实际CLI32行只有27唯一session-turn，5重复首轮/5缺失后轮，不作为该压力case性能。CPU负对照复现，修复在原request-count之外指定准确--num-sessions；5冻结cohort194请求/C16异质延迟/元数据与SSE原始记录全部核对，1005个SDK源码SHA与远端一致，完整442CPU通过。

原W永久保留FAILED，不再次启动。原X/Y只有CPU冻结、无once/ticket/ledger，已加INVALID；新WC=`nextgen-bf16-qwen38-mixed8192-prefix32-r1-20261030wc`以typed session_protocol_recovery绑定原W失败票据/原result/raw/cohort audit和修复CPU proof，GPU配置/质量/模型/源输入相同；先用WC实际验证恢复，再继续新XC=`nextgen-fp8fa2-qwen38-mixed8192-prefix32-r0-20261030xc`和YC=`nextgen-nvfp4-qwen38-mixed8192-prefix32-r0-20261030yc`，保持有限单case。ZC/ZA/ZB原生工具流程不使用AIPerf CLI，会沿用已冻结工具包。有限9个有效主条件不变，W这一项实际失败GPU开销另列，不把10次物理执行说成9次。WC原once现进行中，ACK/ticket/ledger之前不能首插INDEX或重发。
