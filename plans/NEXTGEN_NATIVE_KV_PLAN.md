# 新一代模型原生 KV 工程：当前执行计划

遵循 `ops/NEXTGEN_NATIVE_KV_STANDARD.md` 与生产决策标准。用户要求实际完成新模型 BF16/FP8/NVFP4 代码、部署、质量与生产服务实验；旧 Qwen2.5 分支、失败及原始结果保持，旧 project-briefing 封存 run46。不是规模容忍度论文，不开发新的 INT4 Attention，不向官方仓库 push/PR。

最新确认（2026-10-27）：新binary BF16 i已完成20真实服务检查/实际模型Graph与136真实校准，79文件6121048B独立SHA及所有tokenizer/usage/层统计通过，原COMPLETE/释放/账本结清。观测Kmax23/Vmax149，FP8候选K=.0625/V=.5，NVFP4 K=.015625/V=.0625；候选不是质量证书。正式质量与核心服务统一采用maxbatch16/GDNslots32/memoryutil.90/page64/chunk2048/Graph+prefix，观察器关闭；质量客户端C16+逻辑160k token预算用于避免容量压力污染质量，正式性能另测真实容量压力。1720质量协议已固定（官方NarrativeQA为QA-F1，来源eval.py固定SHA），还没有新模型完整质量或性能结果。

当前采用 Qwen3.8-27B 文本路径，官方 revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。18/18 权重正文、tokenizer/vocab/merges SHA 匹配；1199 个 tensor 中 851 文本、333 视觉、15 MTP，TP1/TP2 meta loader 映射已检查。实际单卡文本服务已运行，不需要为本任务重构视觉或下载权重。64 层为 16 Full Attention 与 48 GDN，Q24/KV4/head_dim256。

目前已经确认：

- 阶段 A 的 BF16 E：65 文件独立 SHA 核验、20 次真实 SSE/已知答案、4 组 A/B/A 重复、并发互不串扰、模型实际 Graph 回放、最长 8794 prompt tokens 与新 tokenizer 一致。Full KV 14,436,794,368 B、GDN 2,770,993,152 B 分开常驻；220288 是物理 token slots，不是实测最大成功容量。此运行仍用旧 ceff 二进制及新 Python 绑定，不能代替新分配器/正式配置的验证。不是质量 benchmark 或性能结果。
- NVFP4 h：H800/SM90、FlashInfer0.6.18.post1、D256/GQA6、page64，直接原生压缩 pages + block16 E4M3 Scale 的 prefill/chunked/decode/writer/changed-input Decode Graph 通过。60 文件/454093 B 镜像逐 SHA、冻结源码与记录数值交叉核验。算子相对压缩值 CPU 参考的 L2 为 .001785–.002657；随机输入的量化后 Attention 相对 BF16 参考 L2 为 .11036–.16218，不是模型质量损失。Full KV 516096 B 加全局 Scale8 B，相同 shape 的 BF16 为1835008 B；GDN6720 B另计。没有全缓存 BF16 shadow，不声称 SM90 FP4 MMA。未执行 C++ server/27B 模型、质量、AIPerf 或性能。原始 output/reference tensors 未持久化，因此本地没有重新计算 GPU 数值误差，保留此审计限制。
- 源码分支 `production/nextgen-native-kv-v1`：混合层绑定与 Scale 身份、NVFP4 分配/页复制/直接 writer+FA2、typed backend/K/V scales 已实现。`862c29ca` 修正 common::Options 漏传并增加正式 prefill/decode 选择；`8aa486d7` 提供 BF16 范围观察器，后续纯 Python 提交增加真实 Graph 资格检查与 FP8 无 shadow marker。观察器目前只有 CPU 验证，尚未通过实际 GPU Graph/model 校准。
- CPU d 在1051/1111附近因 common::Options 缺3字段失败，无 binary；完整失败日志已取回。修复后的 CPU E 为独立 source/manifest/container，复用自己失败的 D/build，旧 host D/source、manifest、result、log 不变。E 一次部分 prepare 已只读证实对象导入完成、source clean、无容器，再明确 no-refetch resume+start；E configure 因 brpc 必须在私有 source/src/butil 生成 config.h.tmp 而失败，容器已退出/noOOM/noGPU；新 F 仅为新副本该子目录开放RW，配置已过，正在编译145个剩余target。F完成145/145及36/36 CXX CPU tests(26+8+2)，随后frozen driver误用bin路径导致原FAILED；实际CMake产物为build/xllm/xllm。G已独立封存568639344B binary，SHA dbc1c53a6d711ad883d2227c55429c1977347acd9e1bfa50b1fb479a0cc25f09；CPU ldd无missing、实际--help注册5新flags。首轮metadata漏已有seccomp profile的sampler线程失败保留，独立verify-v2按已有非root/无GPU/断网/只读profile已通过。私有newbinary映射切换，GPU模型仍待运行。不得重启 d/E/F 或覆盖失败记录。
- 控制器260项 CPU通过。正式 typed kernel/模型调度163项 CPU通过，4项已知旧 NPU 环境测试排除；另有源码180项/110 GPU跳过的较早组合记录。更广 WSL suite 有 torch_npu collection 环境错误，不能称全套通过。
- 核心质量输入1720条（1719 core +1最长独立文档补充），GSM8K1319 +8个LongBench任务各50；重新按新 tokenizer 冻结，最长67736 prompt tokens，无旧模型 context 过滤，无预测。真实校准 b136条 =原128独立ShareGPT +各4份gov_report/qmsum长文，完整context SHA与质量集合排重，max52433、总428735 prompt tokens，自然停止且保持来源生成上限。早期校准 a 把 tokenizer 返回 mapping 的键数误当 token 数，已标 INVALID_CPU_METADATA，无GPU且原文件保留；b用显式input_ids，实际服务还须逐条usage一致。

下一步依赖顺序：

1. 定位 E configure 的真实首错，修自己的隔离构建脚本/挂载；新构建结果与失败目录隔离，尽可能保留并复用自己的中间文件。使用现有 Torch/CUDA/Rust/vcpkg/14个gitlinks，无公网、无GPU、有限CPU/RAM/disk。通过分配器/Scale复制/容量/Graph admission C++测试后冻结新 binary 和 ldd/hash；旧 ceff binary 不改。
2. 新 binary 上做正式 typed BF16 服务及范围观察器实际 changed-input Graph gate，20次生命周期检查后quiescent reset，跑已冻结136真实校准请求，drain后export；原始responses、实际prompt usage、所有Full层/head/prefill/decode计数与候选Scale须独立核验。禁止沿用14B固定Scale，不把范围覆盖等同质量通过。性能观察器必须关闭。
3. 新模型 FP8、NVFP4 各自部署，原生压缩页、Full/GDN分配、writer、Graph、缓存复用和真实输出均过门槛。BF16优先FA3，BF16/FP8 XQA仅在实际D256/GQA6/声明context的数值与Graph门槛通过后使用；NVFP4明确FA2+FlashInfer，不静默回退或全缓存解压。先进版本/算子按实测价值私有隔离升级，不能固定旧FA2弱基线。
4. 正式同模型核心质量：BF16参考→FP8→NVFP4，数学、长文/检索、代码，补有真实工具语义的多轮检查。逐题/任务、cap命中、失败与配对差异及区间，输出/标签分离，不能用20检查或80pilot宣称质量通过。并发、Graph、backend、Scale与性能部署一致；任何明显损伤先定位，不筛选理想数据。
5. 核心生产 AIPerf：真实自然停止chat、独立长文/RAG与可追溯Mooncake/工具到达统计；同时覆盖常规低载及BF16驻留开始限制的并发压力。优先8K/32K/64K真实正文；当前没有128K真实文档证据，不重复填充。32K×C8约262K逻辑tokens是可考虑的容量压力点，但必须检查实际唯一prefix、GDN状态/调度与resident页；不把逻辑并发乘积冒充物理容量。
6. 记录实际有效RPS/output tok/s、TTFT/ITL/TPOT/E2E尾部、排队/拒绝/抢占、payload/Scale/GDN/workspace/Graph、GPU带宽与利用率。关键生产点定位writer/Attention/GDN/MLP开销，分别解释容量收益、kernel收益、Q量化和量化写入代价，不无限扩矩阵。
7. 完成通用dtype/Scale/layout/能力检测与明确不支持组合、测试/配置/工程文档；集中Markdown报告和代码生成图表，分别标明已实现/已部署/已验质量/已验性能，给出合入条件和障碍，不在阶段checkpoint结束任务。

规划容量（不是实测收益）：Full KV BF16=65536 B/token；现FP8 payload+per-head FP32 Scale=33280 B/token；NVFP4 payload+block16 Scale=18432 B/token另加2全局标量。GDN按状态slots而非token线性计费，权重、workspace、Graph、padding另计。正式比较相同GDN状态预算并测实际字节，不把FullKV压缩倍数说成整体显存压缩。

安全继续固定唯一run/票据/账本、每个UUID空闲/预约/120秒/容器/驱动/watchdog检查；未知与失联先只读对账，不复制作业。SSH当前已在有界重连后恢复，早先partial保留。用户已委托按需空闲授权，Agent有限总执行窗口12→24 GPU-hours已按该直接授权记录，budget_id/key和累计11.250323315 GPU-hours不变，reserved0；GPU每job仍2h、最多8卡，其他目录/系统/他人权限不变。大型下载、共享修改、不可逆操作仍先风险检查并确认。
