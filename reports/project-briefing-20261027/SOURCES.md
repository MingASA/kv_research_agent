# 来源、身份与核查

数字优先取既有独立raw审计及已验收汇总。原16张表/CSV与45行INDEX/STATE保持原字节，由[原提取器](build_evidence_tables.py)派生；第46另加4表、27行完整buffered时钟CSV，由[补充生成器](build_post45_supplement.py)读取原审计复核。共20张表，本汇报未运行GPU、生成或重评分模型预测。

[完整来源manifest](data/source_manifest.json)分列原45与第46输入SHA；[产物manifest](data/artifact_manifest.json)覆盖本目录文档/表格/CSV/脚本。SHA认证身份，不自动表示质量接受。原45冻结层与[第46补充](06_第46条BF16Q完整质量与停止决定.md)分开，当前报告仅封存至run46，UTC及下次续接规则见[封存记录](data/REPORT_SEAL.json)。

## 关键身份

| 身份 | 值/边界 |
| --- | --- |
| 服务源码 | 1c2be15976b7fbb1d8b79e469659b2c8e52cff00 |
| 正式C128质量源码 | 11e7cddbfe33eb21d209c7f48f5e461ac56545eb；推理引擎与服务版逐字节一致 |
| 复用C++ binary SHA-256 | 277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3 |
| 质量实际逐shard模型identity | cf67b8e03cf159dd6017e36da84c28e143e59ce766261db2f6c97aabca7aae73 |
| ShareGPT冻结输入SHA-256 | 810415584dc2c751f0cf0e895cd7c4f1418ad338c03f70781d9abfc4e0df44a1 |
| ToolAgent源内容SHA-256 | 48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71 |
| 原生BFCL buffered输入SHA-256 | 47b4b1229449514dba089ab6f6de60a156e1f1df81b5e516b83fc493a2efbcbf |
| heldout128范围输入SHA-256 | ca61bbf5c47e491179c5749cfe5a6c67044daf84d643885e8ae87178d4141869 |
| 新BF16 Q decode源提交 | 78004505e2db44e88994899e8ea3e25c109978a7；仅算子证据 |
| 当前主要runtime | Torch2.10.0+cu130、Triton3.6.0、Transformers5.3.0、FlashInfer0.6.18.post1、AIPerf0.13.0；以各manifest实际记录为准 |

以上身份由下表对应审计/索引绑定；性能run未逐次重新hash全部权重，不能声称每次均有全权重内容审计。各run不可变request/terminal/arm seal、mirror每文件SHA由原审计保存，本汇报的manifest不重复复制所有巨大raw。

## 关键来源与字段

| ID | 支持的结论 | 精确来源 | 关键字段/口径 | SHA-256 |
| --- | --- | --- | --- | --- |
| S01 | 生产目标 | [PRODUCTION_DECISION_STANDARD.md](../../ops/PRODUCTION_DECISION_STANDARD.md) | 真实请求、现代后端、质量与SLO、条件化支持；旧要求覆盖规则 | 7ed3727acd2433901c4a216447a7bcf2bae5f64fbc977f0d845c24df7c646388 |
| S02 | 现代ShareGPT两配对 | [modern_BF16_ShareGPT_ABBA_comparison_v1.json](../../reports/operations/20261026T075846Z/modern_BF16_ShareGPT_ABBA_comparison_v1.json) | rows[*].metrics/delta_pct；duration_weighted_observation_pool；frozen_assets_identical | 8eba54cc42093a88358513817a9af2aef9a1924ba1239849ae45f18c401eca51 |
| S03 | 现代ShareGPT第一对raw审计 | [bf16_xqa_reference_b_independent_audit1.json](../../reports/operations/20261026T075846Z/bf16_xqa_reference_b_independent_audit1.json) | metrics；allocation；封存与source/payload/SSE验证 | c3e7a09b7412e8169a34cb416b0369c80f00b46eb9a44461f6ab4e31f4551120 |
| S04 | 现代ShareGPT反序raw审计 | [reverse_sharegpt_independent_audit_v2.json](../../reports/operations/20261026T075846Z/reverse_sharegpt_independent_audit_v2.json) | metrics；Graph/原payload/usage/驻留与终态 | e6c9fd73b8ba69dd3d4c3af88fd48f1965a43147cabd1cd4802d439df535d471 |
| S05 | 现代ToolAgent两点 | [modern_BF16_ToolAgent_pressure_comparison_v1.json](../../reports/operations/20261026T075846Z/modern_BF16_ToolAgent_pressure_comparison_v1.json) | points[*].arms.measurement/completion_timing/application_observations；跨点冻结核对 | 1d0d466acc341dc9f8df73cc1a6eb6dbf7f5bb5b2bf78b281c82cce235956298 |
| S06 | ToolAgent .125 raw审计 | [toolagent_s0125_independent_audit_v1.json](../../reports/operations/20261026T075846Z/toolagent_s0125_independent_audit_v1.json) | arms；1689正式/399warm、source clock/payload/OSL/SSE与allocation | 903c806a0f12573857bea558599c334ce867e175f034239d48da73126a1aa286 |
| S07 | ToolAgent .25 raw审计 | [toolagent_s025_independent_audit_v1.json](../../reports/operations/20261026T075846Z/toolagent_s025_independent_audit_v1.json) | arms；1689正式/399warm、drain、同输出、Graph/bvar | 4c42db71a841f362b1ea71df8a5b026990c3933de8c9603eab5e8376cf58d96d |
| S08 | C128完整质量raw审计 | [production_C128_quality_independent_audit_b1.json](../../reports/operations/20261026T075846Z/production_C128_quality_independent_audit_b1.json) | task_scores；paired_flips；model_weights_sha256；production_configuration；quality_acceptance | 225142374bcffb2a198c4fcc2fc0ec8190a16297598e66e1c7fc02e2cbc5080a |
| S09 | C128质量及C8敏感性 | [production_C128_quality_analysis_b1.json](../../reports/operations/20261026T075846Z/production_C128_quality_analysis_b1.json) | task_deltas；longbench_macro_delta；C128_minus_C8_*；bootstrap及finish | 222836530894dd80791d2e70a1ceaffe5af5ce3395dc531724e1df549325f1c5 |
| S10 | 新BF16 C8参考 | [BF16_XQA_full_quality_independent_audit_a1.json](../../reports/operations/20261026T075846Z/BF16_XQA_full_quality_independent_audit_a1.json) | 1719原始预测/权重/模型Graph/官方重评分 | 5a40a904937d538b10bcba74e48f3ac8512818f1fbcbf833886696e3b3928898 |
| S11 | 新BF16对保留FP8 C8 | [BF16_XQA_full_quality_existing_FP8_comparison_a1.json](../../reports/operations/20261026T075846Z/BF16_XQA_full_quality_existing_FP8_comparison_a1.json) | 同题跨run分数/配置绑定；FP8保持80+1639缓存边界 | 29e0d1efccd2c7ce5e94cf380457d4a4e2a60ac76a90f8e9778c51dc452e7114 |
| S12 | 原生工具258 raw审计 | [native_tool_buffered_independent_audit_b1.json](../../reports/operations/20261026T075846Z/native_tool_buffered_independent_audit_b1.json) | arms.*.metrics/official_argument_matches；paired_task_outcomes；checker及异常参考 | 62a9989c74e4a471085f90d97d8067ffa7f4ab753fbfbe40e5d2d09e3f7b1e9d |
| S13 | 原生工具有限cohort分析 | [native_tool_buffered_descriptive_analysis_b1.json](../../reports/operations/20261026T075846Z/native_tool_buffered_descriptive_analysis_b1.json) | arms；changes_pct；length_outlier；latency_changes_pct | 7525e866543cdc0aa99cb37d7bb26deb262075cdb26b1e45f294b2ebfe7e83d1 |
| S14 | streaming失败原证据 | [native_tool_stream_failure_a_independent_evidence_v1.json](../../reports/operations/20261026T075846Z/native_tool_stream_failure_a_independent_evidence_v1.json) | BF16首warm失败，无AIPerf/FP8；原FAILED保留 | aa1260564b467373eaac9b700219788848366794c25dca1a662a56f7beb8abe1 |
| S15 | streaming键排序CPU缺陷 | [native_tool_stream_CPU_late_key_defect_v2.json](../../reports/operations/20261026T075846Z/native_tool_stream_CPU_late_key_defect_v2.json) | 原C++解析/flush复现；合法分块与旧坏wire，原token顺序未导出 | 1605e337f7508d5349b66a530ff0829694c0269fbb3607290ba03d0bd0c38a76 |
| S16 | 组件诊断raw审计 | [component_diagnostic_independent_audit_c1.json](../../reports/operations/20261026T075846Z/component_diagnostic_independent_audit_c1.json) | 116文件SHA、637请求/arm、3窗口/48forward、CUDA/Graph时间与phase | 3f3836d9255400142065f2d7ce26dc3b6ac4e23d7f061931e150efcdb7de74c1 |
| S17 | 组件描述性汇总 | [component_diagnostic_descriptive_summary_c1.json](../../reports/operations/20261026T075846Z/component_diagnostic_descriptive_summary_c1.json) | arms.*.component_kernel_sum_ms/writer_share_of_all_kernel_time；matched_recorded_batch_metadata | 09cc7c299fb5d391d4b1bafa48c2ea48150ce053a0fb90dd31da8eed97d4c83c |
| S18 | 组件活动归属 | [component_diagnostic_activity_times_c2.json](../../reports/operations/20261026T075846Z/component_diagnostic_activity_times_c2.json) | CPU scope、kernel sum、busy union、Graph与未归属边界 | 44a87a528aaa270de94ec722fab1b886094452fd5489aa1cb69ebb27c7f86cd0 |
| S19 | prefill工作区算子 | [prefill_workspace_independent_audit_a1.json](../../reports/operations/20261026T075846Z/prefill_workspace_independent_audit_a1.json) | timings.raw_ms/median_ms；workspace_reserved_bytes；model_loaded=false/AIPerf_ran=false | be9bb4fdfd4a0c759942355419237c22edc89faf9c2f332796605499ef350686 |
| S20 | 真实K/V c恢复审计 | [range_probe_c_independent_recovered_audit_v1.json](../../reports/operations/20261026T075846Z/range_probe_c_independent_recovered_audit_v1.json) | distribution；original_root_status；worker_measurement_seal_present；rate_clock_precision_verification | b24a66735b65b94688f1b1a6c414d5f128d2e29c988275ca5ad87297bf8acc85 |
| S21 | 真实K/V摘要 | [audit_summary.json](../../reports/experiments/production-kv-range-probe-20261027/audit_summary.json) | 1536group、finite值/absmax/clipped/underflow/relative_l2、reset/原FAILED | c7e2157fd8aaf885378902bb03fb636514f7176b947668cf7b47df865794f02c |
| S22 | writer覆盖与下溢界 | [range_probe_c_coverage_and_error_mechanism_v1.json](../../reports/operations/20261026T075846Z/range_probe_c_coverage_and_error_mechanism_v1.json) | 214986=143037−2048+74125−128；underflow_energy_bounds | 748ef2b24592a8df3b8ced25700c9e8ea92c13dcc8271cc9f526aa2ae60b52a6 |
| S23 | scale预导出规则 | [range_probe_candidate_scale_rule_before_export_v1.json](../../reports/operations/20261026T075846Z/range_probe_candidate_scale_rule_before_export_v1.json) | 先冻结判定规则，后查看真实分布 | 6d4fa35f311ef7f4fc9aad9cf6f349e5f2f2d3cbe80b22443b2f70b6a81a563d |
| S24 | scale保持决定 | [range_probe_c_derived_scalar_proposal_v1.json](../../reports/operations/20261026T075846Z/range_probe_c_derived_scalar_proposal_v1.json) | 保持K=.125/V=.0625，无新global scale | f257da5395cb10eb051980f4f431e385354250bf44b8b4c59227c239139ca18e |
| S25 | BF16 Q新备选算子 | [bf16q_decode_operator_independent_audit_a1.json](../../reports/operations/20261026T075846Z/bf16q_decode_operator_independent_audit_a1.json) | graph128.dynamic_live_rows_and_context/backend；installed_sources_sha256；无模型/质量/服务 | 96408bfd9afad5b9df6b86d7de0fc23273439525939a88c52195b7f44f7afec8 |
| S26 | BF16 Q原终态/占卡 | [bf16q_decode_status_a4.json](../../reports/operations/20261026T075846Z/bf16q_decode_status_a4.json) | status=COMPLETE；files.result.json.value.allocation_intervals_ns；gpu_release_verified | 8dcca8a1fa7b1c97ff330865e75984ead8a21cd51b4b68a69e6266b77bdfc1de |
| S27 | 历史四臂归档 | [provenance_readonly.json](../../reports/operations/20261026T040426Z/provenance_readonly.json) | historical_quality.files.summary.json.task_scores/longbench_8_task_macro；原始summary/CSV SHA | 3a75c56b2316a57e484e0c8858645afbb12e763c625939c31fd0df1fd7e28037 |
| S28 | ceff18case/九条件审计 | [audit.json](../../reports/experiments/ceff-takeover-20261026T040426Z-v2/audit.json) | runs；case_count；graph/存储/hash与limitations | 17e2777d4344978cf20c75e3a2dfee9070b8b95c712a9ed96d37093bdb98603f |
| S29 | ceff九配对原CSV | [pairs.csv](../../reports/experiments/ceff-takeover-20261026T040426Z-v2/pairs.csv) | 同FA2自然停止、实际token、吞吐/ITL/TTFT变化 | 3dcf965d80ac70ba9bd3e6c8af79d6cd24358b77fa37db80a804ad72010545a5 |
| S30 | 早期FA3全量质量 | [audit.json](../../reports/experiments/production-fa3-quality-full1719-20261026/audit.json) | 80+1639合并、完整官方重评分、质量差/抽样区间 | 301cf8511595c59d234c5faf4763f11703044bc1612beb356cb68b04a33f1a39 |
| S31 | 旧XQA全量质量 | [audit.json](../../reports/experiments/production-xqa-quality-full1719-20261026/audit.json) | 三run恢复合并1719；同FP8保留预测、旧参考 | 7caebece84319d1bc21a612c13eb9acd1bb2a5179357f0daccbc06a97940d39e |
| S32 | 旧原速ToolAgent raw审计 | [toolagent_trace_independent_audit_c2_py312.json](../../reports/operations/20261026T075846Z/toolagent_trace_independent_audit_c2_py312.json) | arms；全部2088尝试、正式1689成功/拒绝、source/clock/usage | 4c7c44217dccec39425460d9c047336d48e833e3320ebe2f126b2cc4a9d417b3 |
| S33 | 旧BurstGPT raw审计 | [burstgpt_f_independent_audit1_py312.json](../../reports/operations/20261026T075846Z/burstgpt_f_independent_audit1_py312.json) | arms；6385全部/6221正式，offered限制 | 3bf842b9578e1fb66877111d2809ff28ee8c40c9fcc8f2d83f9ce939d4b7a314 |
| S34 | 旧conversation raw审计 | [conversation_f_independent_audit1_py312.json](../../reports/operations/20261026T075846Z/conversation_f_independent_audit1_py312.json) | 原CONFLICT保留，独立恢复measurement与完整时钟 | e73db3e5c0d5d5bacb14a5fce2a667af1a88fbfc3c1d7865aa714c69b4a3ab1a |
| S35 | 旧synthetic raw审计 | [synthetic_f_independent_audit1_py312.json](../../reports/operations/20261026T075846Z/synthetic_f_independent_audit1_py312.json) | 过载成功/拒绝与较慢生成期，完整原始记录 | 5c8737ec81f837220287fc7a5d736e2ff35e63c4b8fa6fefffa6cd9206c530a5 |
| S36 | 内容/末SSE计时 | [STREAM_TIMING_FINDINGS.md](../../reports/operations/20261026T075846Z/STREAM_TIMING_FINDINGS.md) | 实际AIPerf源码、14556条成功测量核对、非TCP EOF | 94c07d3ded0638f67b3d1ebc86d4a8007f07357bdcb820b8a60eaf509afd4de9 |
| S37 | trace源语义 | [PRODUCTION_TRACE_FINDINGS.md](../../reports/operations/20261026T075846Z/PRODUCTION_TRACE_FINDINGS.md) | 原长度/到达/prefix来源，匿名内容，过滤和BurstGPT eligible窗口 | 3d9ee4a1539c66d3936e2d5ff6829acffe491c3244beebff518cb2f3e6a1feef |
| S38 | 源码机制 | [PRODUCTION_SOURCE_FINDINGS.md](../../reports/operations/20261026T075846Z/PRODUCTION_SOURCE_FINDINGS.md) | BF16/FP8连续化、writer、固定scalar、非完整校准、代码评分 | 6ca47a10fda10bac5cab111160e023c7c77746bbf661b84bfa5369d1f092cf72 |
| S39 | 上游14直接文件 | [flashinfer_sm90_direct_headers_comparison_v1.json](../../reports/operations/20261026T075846Z/flashinfer_sm90_direct_headers_comparison_v1.json) | 现装/固定上游14文件字节SHA一致，不证明全后端性能 | fa7c6a2207e4156e52c8efe43edb915d598a7fb20c8591312f06a7718407f745 |
| S40 | 新候选质量计划 | [production-fp8-bf16q-full-quality-v1.md](../../plans/experiments/production-fp8-bf16q-full-quality-v1.md) | 仅新FP8候选完整1719/C128，保留BF16参考；第45原模型前FAILED，无新质量结果 | a8018eed3654bb11d53d9a34cff002376bc8436798fac1ec6df11c36b28ff2e8 |
| S41 | 新候选质量a失败 | [bf16q_quality_status_a2.json](../../reports/operations/20261026T075846Z/bf16q_quality_status_a2.json) | files.result.json原FAILED/SHA与allocation_intervals_ns；driver释放，无模型质量；本地b未派发 | a9ec60d1907a19424e9e287b5cd32ee40d1192e65b7897dd84d7d9a494ff895e |
| S42 | FA3 pilot输入manifest | [pilot_manifest.json](../../results/mirrors/prod-quality-fa3-pilot80-r0-20261026b-c1/files/support/agent_work/runs/prod-quality-fa3-pilot80-r0-20261026b/input/pilot/pilot_manifest.json) | samples_per_arm=80；GSM64；八任务各2、最短和最长各1；输入准备不是质量结果 | bc012cc149f0bb0de6f1a6e431266d4aac4c9a2166d25038a0d50e8ffc977700 |
| S43 | XQA pilot输入manifest | [pilot_manifest.json](../../results/mirrors/prod-quality-xqa-pilot80-r0-20261026c-c1/files/support/agent_work/runs/prod-quality-xqa-pilot80-r0-20261026c/input/pilot/pilot_manifest.json) | selection_rule/selected_source_indices与FA3 pilot完全一致，64+8×2=80 | bc012cc149f0bb0de6f1a6e431266d4aac4c9a2166d25038a0d50e8ffc977700 |
| S44 | 第45独立失败审计 | [bf16q_quality_a_independent_failure_evidence_v1.json](../../reports/operations/20261026T075846Z/bf16q_quality_a_independent_failure_evidence_v1.json) | 72文件/20427910B逐SHA、零预测/无measurement seal；actual_GPU_hours与原interval一致，原FAILED保留 | a230bc8bad6ced98a8823fd3e6514006170955b7075fc9d6baa45254fc574770 |
| S45 | 本地b完整CPU dry-run | [bf16q_quality_b_full_actual_dry_run_CPU_v1.json](../../reports/operations/20261026T075846Z/bf16q_quality_b_full_actual_dry_run_CPU_v1.json) | source acc0ed6b；原full1719/C128实际dry-run、frozen policy有效；GPU_started=false，不计新模型结果 | 5cd7e9074c8965ccf32c51795797397fb002a684e0a8b783101cf2b58d28bf2d |
| S46 | 本地b源码检查 | [bf16q_quality_source_CPU_v2.log](../../reports/operations/20261026T075846Z/bf16q_quality_source_CPU_v2.log) | 85 passed；仅本地CPU检查 | 85b2e42f5a248480b9f9ca36dd886f3fb6f6686f366117b3bd83f0d08f3d29a5 |
| S47 | 本地b控制器检查 | [bf16q_quality_controller_CPU_v2.log](../../reports/operations/20261026T075846Z/bf16q_quality_controller_CPU_v2.log) | Ran207/OK；仅本地CPU检查 | 34874d9bc39dbe20e6b6674b9f5375dff2b2a132d782a54dd9615c1c1a5e4252 |
| S48 | 第46候选完整原始审计 | [bf16q_full1719_quality_independent_audit_b1.json](../../reports/operations/20261026T075846Z/bf16q_full1719_quality_independent_audit_b1.json) | 原COMPLETE/driver释放；1719原题/权重/Graph/官方重评分；actual_gpu_hours；audited_source_files_sha256 | 5496ad307e815ab2cbe35cc0530e4232f58678c964ae150b1d192a8f7675b433 |
| S49 | 第46三配置逐题质量对照 | [bf16q_full1719_quality_comparison_b1.json](../../reports/operations/20261026T075846Z/bf16q_full1719_quality_comparison_b1.json) | task_scores/task_deltas；LongBench_macro；paired_bootstrap_95ci；paired_score_changes/cap_finish_counts；quality_accepted=false | e8e89981a58e98c08dc425e7c7cbac101bb66318e4b800c58ee1a03945030f78 |
| S50 | 第46有限buffered cohort时钟 | [bf16q_full1719_quality_cohort_latency_b1.json](../../reports/operations/20261026T075846Z/bf16q_full1719_quality_cohort_latency_b1.json) | tasks.*.metrics/paired_descriptive_changes；new_GPU_runs=0；不同run/GPU/缓存，非AIPerf稳态 | cc9c0f90ce5fe531bcc935796730800598c8659f0056b3310ea8df19a60a7c99 |
| S51 | 第46正式报告摘要 | [audit_summary.json](../../reports/experiments/production-bf16q-quality-20261027/audit_summary.json) | canonical_audit/comparison/cohort_clocks三项SHA；123文件/44462952B；extra_service_matrix_stopped=true | 3bd6f08a7f98267e28c2b0fe9f55d4c033908f6ea5871a7ad9a214fb29b2e6f3 |
| S52 | 第46现成质量图manifest | [manifest.json](../../reports/experiments/production-bf16q-quality-20261027/figures-v1/manifest.json) | comparison/generator/SVG/PNG/CSV SHA；固定原题相对现代BF16差值，非Q单因子 | a8bfb982b43d455e9e071f3099bb363428abf04ee2254b96a11ad7a1c6c6590e |
| S53 | 第46完整gzip镜像 | [SNAPSHOT.json](../../results/mirrors/prod-quality-bf16q-full1719-c128-r1-20261027b-terminal-b2-gzip/files/SNAPSHOT.json) | 123文件/44462952B逐SHA；原终态及所有封存质量证据 | 14a87766538f52026ab7bc9354899756434b81b74c90ad6380101d45c3e6bc4c |
| S54 | 第46xz取证失败 | [transport.json](../../results/mirrors/prod-quality-bf16q-full1719-c128-r1-20261027b-terminal-b1/transport.json) | rc255/keepalive无响应、partial不当有效raw、remote_writes=0；不是GPU失败 | 5fdbfd08dc073bb96e7144e8a9e4b727ba0701b70bc8d6513adc0f7a90057aaf |
| S55 | 第46gzip取证恢复 | [transport.json](../../results/mirrors/prod-quality-bf16q-full1719-c128-r1-20261027b-terminal-b2-gzip/transport.json) | rc0/gzip、客户端keepalive2→6/10秒；不重跑GPU | 8376f66828ed8bc85529822e52e47252c52cffbce79e0610402a5dd534a03675 |
| S56 | 第46官方评分逻辑兼容 | [bf16q_quality_scoring_compatibility_CPU_v1.json](../../reports/operations/20261026T075846Z/bf16q_quality_scoring_compatibility_CPU_v1.json) | only_new_strict_profile_literal_diff；evaluator SHA与candidate/reference audit绑定 | 63c3f1ce8b6f1f82c7d8831c63c5d307028a9e03d2d9494816e2567a9c97ead3 |

## 核查方式与结果

原[数值核查](data/numeric_checks.json)保留14组，覆盖既有服务/容量/质量/工具/组件/工作区/范围/Graph与pilot、失败及CPU准备。第46[补充核查](data/POST45_checks.json)新增9组：三配置原题/分数/差值/逐题/cap、既有bootstrap口径、27行cohort分母与实际token、123原镜像SHA、终态/ceil微秒成本、取证恢复和35个原文件字节不变；另验证正式报告/既有图SHA，合计23组。

[静态核查结果](data/validation.json)给出正文/来源链接、canonical SHA绑定、占位符及输入身份检查。独立完整性/官方评分沿用已验收审计；本汇报只复核已评分值与文件，没有独立重复GPU数值或模型预测。

原45行INDEX/STATE/CUTOFF与原16表/CSV的35文件SHA保持不变；第46完整质量未整体修复、停止该候选额外服务GPU矩阵。报告封存至run46，后续仅在用户要求时从run46之后续接；不主动跟踪或更新，UTC见封存记录。
