# 真实长文 QA 的 KV 范围诊断（执行前方案）

状态：proposal，输入CPU冻结，不是GPU或质量结果。汇报文件已截止第46条封存，本实验仅维护运行和证据记录。

生产决策问题：既有FP8-XQA有高压服务收益但长文/代码质量有损；保留BF16 Q的完整1719题候选仍未整体修复。短聊天128探针没有clipping，但最长仅2980token，不能排除真实长文上下文的固定scale饱和。此诊断决定是否值得实现可部署的scale覆盖修复，不填性能矩阵。

输入：`results/prepared/calibration-long-document-QA-heldout128-20261027a2/`。canonical SHA `1a36f83244b09b3edd85895586ec324624640ade004eb3ece672e94ee5dcfe5a`，manifest SHA `ffe9f783cc749580a383da83639c80c07e0c96bbed085aaa0f703bfc4fe36c12`。128请求，Qasper/MultiFieldQA-en/HotpotQA/MuSiQue各32，4096到未截断模型上限内按长度四分位各8条、SHA排序seed20261027。实际4103–20227token，共1458582，84条>=8K、34条>=16K。保留原prompt/template、chat模式、自然EOS、各任务128/64/32/32输出上限；不拼接、不截断、不强制生成。

来源为官方[zai-org/LongBench](https://huggingface.co/datasets/zai-org/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672)，固定revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`；data.zip 113932529B，SHA `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`，仅本地下载。旧50题是源内固定抽样，不能假定前50行；按ID映射并逐字段核实原记录完全相同。排除旧全部400题的ID、完整record SHA和context SHA，并对新上下文去重。源公开记录含答案，但不用于选择；上传/请求不带答案，不以旧模型预测/分数选题。不声称语义或训练语料独立。

唯一GPU候选 `prod-calib-bf16xqa-longqa128-r0-20261027a`，单BF16 FA3+XQA，复用已验收source `5e2c99f9dca1d1b22de9222722dfb24f75045c0c`、原binary/image/模型。生产Graph/prefix/chunked prefill，128seq/10240tokens/200admission/.8，客户端C128；不是固定GPU batch128。只在全新隔离目录运行；0warmup以省略AIPerf选项表示，bootstrap后静止RESET，完成后先EXPORT再解析。1536个layer/phase/KV/head计数组，105216B统计缓冲，无完整激活/ShadowKV。每次复查真实数值、Graph重放、reset/export门槛。

预算：最多1800秒/单卡，即预留0.5GPU-h，输出上限3GiB。原12GPU-h自限账本不重置；执行前9.7240637278已结算/0预留。必须fresh只读巡检、UUID/归属/预约/容器/共享CPU-RAM-disk/watchdog及120秒稳定空闲通过；SSH掉线只读恢复原任务，不能重复launch。任何他人占用/排队不启动。

执行前验收：实际冻结policy/资产SHA、全部控制器CPU回归、真实输入逐条独立对账、安装AIPerf的loader+raw/mmap发送路径256次本地loopback、完整未来CLI在实际Cyclopts CLIConfig解析通过。冻结运行的控制代码不得热改。

预先决定：K=.125/V=.0625保持不变观察。若全部有效写入无clipping，不因这份数据降低/调整global scale，也不启动盲搜GPU；两次幂改scale在共同normal范围不改善FP8舍入精度。若出现真实clipping，先独立核对有效slot/epoch/数值/精确覆盖范围，只提出最小两次幂coverage scale，后续新配置仍须质量/性能验收，不能直接宣称修复。underflow只给误差能量界，不能仅凭计数调scale。

限制：这是BF16实际生成轨迹的写入激活及固定FP8 QDQ代理，不测FP8反馈/Q误差；长文QA不覆盖代码completion。AIPerf用于请求完整性，不以含探针开销的时钟宣称吞吐收益。原始终态、失败、恢复日志均保留，最终逐SHA/usage/Graph/epoch和计数独立验收。
