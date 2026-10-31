# 原会话恢复与工作接续

原会话 `01a1765d-9953-7760-a9fa-102c05c766d7` 已完整恢复，原末尾尚未完成的报告与接续提示词已经补齐；原27B原生KV生产研究已完成本轮交付。最终状态入口是 [STATE](../../../state/STATE.json)、[NEXT_ACTIONS](../../../plans/NEXT_ACTIONS.md)和[NEXT_SESSION_PROMPT](NEXT_SESSION_PROMPT.md)，它们优先于历史快照。

恢复读取了129515448字节原JSONL，SHA256 `3c38100d41de4af38ccfd4f67fe9383e7c52af763307c126a0d38cc9ebe587d8`；35205条记录、20次压缩、全部37条user消息。全部用户原文保存在[USER_INSTRUCTIONS_FULL](USER_INSTRUCTIONS_FULL.md)，索引与原末尾动作见[SOURCE_RECOVERY](SOURCE_RECOVERY.json)、[LAST_ACTIONS](LAST_ACTIONS.json)，原始来源重新校验见[恢复验收](ORIGINAL_SESSION_RECOVERY_ACCEPTANCE-a1.json)。没有将原会话内部分析日志复制到交付归档。

原末次主工具动作是34487，冻结AE/AF CPU工具包；原最新user消息35175要求集中报告与新会话提示词。AE/AF的旧CPU协议随后被判无效并保留，未冒充GPU结果。旧AB其实已于2026-10-29原COMPLETE，本次恢复只读核实、结清、镜像和评分，没有重复启动。旧自动队列HALTED历史不会续发。

AGENTS.md、MASTER_PROMPT、生产决策标准、新一代模型标准、远端安全、文件归属与授权规则均已阅读。最新用户指令优先：Qwen3.8-27B原生BF16/FP8/NVFP4生产实现与实测，必要时才合理替代，不再开旧Qwen2.5矩阵。用户已委托授权处理；本次沿用固定私有授权、同budget_id与累计账本，Agent自限24GPUh及单卡隔离，没有清零预算或占用他人任务。每个新GPU派发只经launch_guarded_once，新ID、原intent/ticket/ledger/exclusive回执；Unknown/SSH中断先只读原票据，不重发。

远端先只读恢复，再写经确认的项目新隔离目录。H800原身份为UID/GID1005、boot5afaf1c0-ecf6-4bda-a039-5341ef954efb，GPU7 UUID GPU-47211491-ba0d-78bf-5c84-f968c0648213；共享资源/归属/空闲窗口/已知预约门槛逐run复查。未下载大权重、修改共享系统/全局依赖/身份/网络，未在H800运行Codex，也未向官方push或发PR。只读权重与模型体18/18 SHA来源证据保持。

Python模型源03134b72、C++源862c29ca，实际binary SHA `dbc1c53a6d711ad883d2227c55429c1977347acd9e1bfa50b1fb479a0cc25f09`；40文件主线patch与增量bundle固定基线11e7cddb..03134b72，已验证。它们是本地主线评审材料，不冒充当前上游CI或已合并代码。控制修复261ceea及研究分析e124bce已提交；WC原票据保留发票时HEAD fd2238d，冻结模块逐SHA与261ceea一致，见[来源对账](session-protocol-code-provenance-a1.json)。

已经接续完成原2048配置三格式核心1720评分、9组真实AIPerf（528正式+60外部warmup+180known）与三格式真实工具32任务。工具分数BF16/FP8/NVFP4为26/26/24of32；FP8一题预算未完成保留计0。旧质量FAILED及FP8不重叠918+802恢复保留原终态，未按答案选择。核心实现、CPU、实际SM90模型、Graph、质量与性能分别记录，未把CPU fixture当模型结果。

仅新增一个有根据的8192/2048调度候选。真实S pilot136请求/20known/A-B-A/Graph与同采样4running/2265FullKV页资格通过。新T/U/V分别单原COMPLETE/释放/SETTLED，各完整1720独立评分，共5160；GSM flexible96.5883/96.8916/96.4367，LongBench53.8767/53.4349/53.6044。与新BF16配对，不合并旧配置统计，也不因CI跨0宣布质量等价或默认生产接受。见[集中研究报告](../../synthesis/NEXTGEN_NATIVE_KV_20261030.md)。

新压力W真实FAILED：32HTTP记录只有27唯一session-turn，5重复首轮/5缺后轮。原FAILED、490355487GPUus及109SHA完整保留。CPU负控复现request-count-only会话循环，修复指定准确session总数并保留request cap。实际CLI五cohort194负载/C16异质延迟、全部payload/元数据/轮次/SSE/usage通过，1005SDK源与远端逐SHA相同，442CPU通过。Luna有界只读静态审计确认建议已落实。

新WC明确replicate1/typed recovery绑定原W票据/失败/raw及修复proof，已原COMPLETE/释放/SETTLED；113SHA与32唯一turn/16完整两轮严格审计通过。真实200.924秒/.159264RPS，对旧BF16 O328.678秒更快，但请求平均ITL p95从26.541ms增至758.491ms，原SSE内容包到达gap也显示停顿，不自动称新配置更优。新FP8 XC亦已118SHA/完整32独立过，222.837秒/.143603RPS；同采样9running/5600有效FullKV页，超过同预算BF16物理3868页，但请求率比新BF16低9.833365%。容量与服务收益分开报告。

新NVFP4压力YC已原FAILED/释放/SETTLED：32个offered中31成功、1个约600秒Timeout；115SHA独立核验，原失败保留，不重跑或放宽条件。附加应用采样SSH回复丢失，active页/并发峰未知，不补0。新工具BF16 ZC独立26/32，FP8 ZA独立27/32（task2预算未完成计0，110/112轮）；最后NVFP4 ZB原COMPLETE/释放/SETTLED、286SHA/21779727B完整取回，32会话/112轮生成完成，独立评分已完成25/32（78.125%），无未完成会话。所有计划内GPU条件及后续本地分析、图表、归档均已完成。

最终只读资源证明已实际通过：83个账本run全部SETTLED、currentreserved0、UNKNOWN0，75个远端run子树均为已知归属、无running labelled容器；source/binary/model SHA匹配、模型源031clean。固定累计20.7042928222GPUh，未重置budget_id。实际8张H800/driver535.216.03，每张81559MiB；设备/GPU7 UUID和boot等精确证据见[最终资源证明](final-resource-closeout-a1/CLOSEOUT_ACCEPTANCE.json)。这里只报告该只读时点，不声称其他用户之后不会使用资源。

Luna曾在人工阶段切换时有611秒检查间隔，失败与修复保留。固定driver随后完成9次每300秒实际S0（00:33:17至01:13:17），均SSH成功；main资源证明通过后写stop receipt，driver01:18:11Z读到后自退，PID371559由main确认不存在。executor没有数值shell exit code，记录null；见[监控最终回执](CLOCKED_LUNA_FINAL_EXIT_REPORTED-a1.json)。

原质量、原9组服务、原三组工具、S pilot、新三格式质量、W失败与修复均已本地归档并逐成员重读SHA；最新两个包额外由Luna只读流式复核通过。新quality289成员/53890603B，SHA `d96c90e28ed815421bd40b6cecc7a177c5b9b4096d63642fa5087220d54e4f7b`；W失败+修复205成员/10649504B，SHA `46322019721ec078a728012a0c38a80d0819f42caae669edac3a666009fc4253`。归档不含权重/完整安装环境，不是部署镜像。新压力三条件及保留失败的图/驻留图已完成并归档406成员、gzip SHA15f0032a7ce31524855d64354c8f8e7ec72b0ff45005ec464f5ef88723b44e02；新工具三格式26/27/25评分、配对/图与1182成员mixed-tools归档均已完成；资源/83ledger/监控已收尾，统一目录/control bundle/26项范围审计也已完成，本轮交付完成，不代表默认生产接受。完整26项原范围见[最终审计](COMPLETION_REQUIREMENTS-c1.json)，原b1保留为中途历史。

统一[九分区交付目录](../../../results/archive/nextgen-native-kv-20261030/CATALOG-a1.md)已重新核对全部实际gzip/manifest SHA，共303619329B；控制器完整bundle固定4f07cfde、SHAe79bd08f...946ac、无前置commit依赖，模型patch/bundle仍固定11e7cddb..031。最终metadata提交晚于冻结bundle并明确区分，未向官方push/发PR。
