# 原生工具调用：匹配率相同，FP8出现生成长尾

confirmed：完整JSON原生工具API的两臂各258测量+16独立warm全部完成，AIPerf均退出0/真正结束，无HTTP或参数JSON契约错误；109份5542146bytes原始证据逐SHA、全部payload/usage/源码/Graph/封存及官方检查器独立验收。root COMPLETE、driver释放。相同总匹配率没有转化为此批的整体收益：FP8有一条重复输出至1024上限，拉长整个有限cohort的完成时间。该异常完整保留，没有剔除重跑。

| 指标 | BF16 FA3+XQA | FP8 FA2+XQA |
|---|---:|---:|
| 原官方函数名/参数匹配 | 182/258（70.5426%） | 182/258（70.5426%） |
| 正式服务完成 | 258/258 | 258/258 |
| observation秒 | 7.856381 | 16.134905 |
| 整批成功服务req/s | 32.8395 | 15.9902 |
| 整批官方匹配正确req/s | 23.1659 | 11.2799 |
| 实际输出token/s | 1681.181 | 958.853 |
| 实际输出总tokens | 13208 | 15471 |
| 完整请求延迟P50 ms | 2480.013 | 2296.807 |
| 完整请求延迟P95 ms | 4994.450 | 4770.198 |
| 完整请求延迟P99 ms | 5944.110 | 5992.239 |
| length上限终止 | 0 | 1 |

175题双臂匹配正确、69题双臂错误、7题只BF16正确、7题只FP8正确。总分相等不证明逐题相同或非劣；未获用户接受的质量/SLO界不自行假定。两条公开参考共7个required array候选为空，原checker无法匹配，原258分母和答案保持不变。此为函数名/参数匹配代理，不执行外部工具，不认证完整多步agent能力。

长尾源ID live_simple_175-101-0：BF16为78tokens/2.777秒、输出一次原生调用；FP8无工具调用，反复输出sourceMapping直到1024tokens/13.623秒。两臂原官方匹配均错，所以总准确率掩盖了这条生产生成行为差异。该请求稍后才被closed-loop发出，成为FP8末完成请求；完整observation增至16.135秒。FP8总输出多17.13%，整批服务与匹配正确率每秒均少51.31%，实际输出TPS少42.97%；P50/P95稍好而P99基本持平。只有一对短有限cohort、非稳定持续到达/峰值吞吐，不将这组比例泛化到所有工具生产流量，也不通过去掉长尾构造更有利的headline。

H800 GPU0/Qwen2.5-14B/source11e7cddb，推理/API字节与已测1c2一致，复用ceff binary SHA277bd8f391fbef978c86d77a976d488d567b0781bf19c0cc8a9c6763fccb21f3。BF16 FA3 BF16-Q/BF16-KV prefill+XQA BF16 decode；FP8 FA2 BF16-Q/FP8-KV prefill+XQA内部FP8-Q decode，PDL=true/K.125/V.0625。生产Graph/prefix/chunked-prefill/128seq/10240tokens/200admission/.8、closed-loop C128、自然EOS/greedy/cap1024；每臂独立缓存/120秒空闲/数值与真实Graph/API门槛均验收。全部258配对server prompt counts相同。性能阶段未重新hash全权重，不冒称新权重全内容审计。resident容量仍约1.94倍，不等于此工具负载收益。

输入为原BFCL commit6ea57973的完整258 live_simple和16独立live_relevance暖请求。原消息/tools/采样不改，执行前独立选择完整JSON协议，仅stream=false/移除stream_options，dataset SHA47b4b1229449514dba089ab6f6de60a156e1f1df81b5e516b83fc493a2efbcbf。原输入在答案下载前冻结，父SHA/逐payload绑定保持；完整JSON无流式TTFT/ITL，不用完整响应到达时间冒充首token。

## 流式失败与维护条件

旧a在首warm原生API门槛失败，未开始AIPerf/FP8；root FAILED、driver释放、0.096657004GPU-h和55份875600bytes证据永久保留，不重启。原SSE标准delta拼接包含重复后缀，参数JSON无效，HTTP200/tool_calls/DONE不能代替有效参数。

CPU已确认原C++增量解析器的键排序缺陷：合法模型JSON按prompt,width,height,image_format生成时，nlohmann.dump排序改变已发送前缀，旧sent长度仍用于切片，prefix.substr越界异常又被吞。原Base/Qwen25/partial JSON核心和API最终flush函数体不改，chunk1/2精确复现旧坏参数字节，其他多种分块还能静默形成合法但错误的参数；排序键控制9种分块全对。原始pre-parser模型token流未导出，历史实际键顺序仍unverified，不称逐token历史回放。此为共享基线接口缺陷，不归因量化；当前完整JSON实验不认证流式接口，没有客户端补字/去重或热改binary。

CPU准备通过188控制器检查、274sessions/548次实际AIPerf HTTP与TextResponse/RawRecordInfo JSON序列化。验证脚本导入/dataclass API错误均在GPU前修正，v5通过；只读自定义RPC缺ok封装拒绝不是SSH断线，采用已SHA验证镜像。新prefill源码分支的CPU/离线编译属于独立候选，不混入本run或本结果。

证据：canonical reports/operations/20261026T075846Z/native_tool_buffered_independent_audit_b1.json（SHA62a9989c74e4a471085f90d97d8067ffa7f4ab753fbfbe40e5d2d09e3f7b1e9d）、reports/operations/20261026T075846Z/native_tool_buffered_descriptive_analysis_b1.json、native_tool_stream_CPU_late_key_defect_v2.json。原始镜像results/mirrors/prod-tools-bf16xqa-bfcl258-buffered-c128-r0-20261027b-terminal-c1/files，复算scripts/audit_native_tools_run.py。实际占卡约0.201068223GPU-h，含两臂JIT/加载/Graph/warm/全部请求/回收；累计9.045736586GPU-h、预留0，第一12未重置。更新2026-10-27T09:53:29.017209+00:00。
