# proposal：真实原生工具调用的任务成功率与服务收益

动机：已有匿名ToolAgent仅复现到达/长度/prefix，不能证明工具调用正确；LongBench代码相似度也不是任务成功。本项补实际生产API的函数名/参数契约，不以新benchmark代替完整质量或多步agent评估。当前组件诊断c继续运行，不因此新增GPU作业。

来源为[BFCL官方分类](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/TEST_CATEGORIES.md)中的用户贡献live_simple。固定commit6ea57973c7a6097fd7c5915698c54c17c5b1b6c8，全258条原问题及系统消息，16条独立live_relevance用于warm，类别/源ID不重叠。公开数据卡与仓库LICENSE声明Apache-2.0。全部274原消息逐项比对，未根据评测答案选题；16条warm包含多工具，258条测量均单工具单轮，不能说已覆盖多轮状态或代码执行。

请求使用原生OpenAI兼容tools/tool_choice=auto，官方convert_to_tool及其type mapping的原AST定义保持不变（不伪称已安装全部BFCL provider SDK）。明确dots→underscore映射、原source行SHA与payload SHA。原文不改写，保留自然EOS/greedy/cap1024、实际生成长度与完整流式tool_calls/usage/finish；不强制输出长度或正确调用。客户端closed-loop C128是服务压力协议，不冒称原生产到达时间线。

CPU输入已冻结results/prepared/bfcl-native-tools-258-20261027a；258测量prompt估计173–845tokens，warm288–1681。输入358947bytes/SHAf01bb2122967a06d906cca8e0562ae6344312af573fc9fa1c791951c200ed83a。actual安装AIPerf源码的loader/模型/client/HTTP到WSL loopback测试，274sessions/548requests、dict与preencoded两路径原payload保持；ToolCallResponseData确实保留tool-only和mixed prose/tool文本。证据native_tool_AIPerf_wire_CPU_v1.json。这没有模型生成、质量分数或GPU性能，不进结果索引。

正式前仍需：冻结官方Python AST checker/答案与xLLM静态model profile；先CPU运行原函数、正/负契约验证，不执行模型生成的代码或真实外部API。xLLM源码有qwen25 parser与tools时保留special tokens，实际ceff二进制仍须在新独立任务中验证--tool_call_parser=qwen25、完整SSE工具delta合并及API/usage。工具参数不正确、无调用、格式/拒绝/超时都计入任务或服务失败，不删样本、不按答案重试修正。不以HTTP200认定任务成功。

候选服务仍H800/14B/先进BF16 FA3+XQA对FP8 FA2+XQA、当前固定scale、生产Graph/prefix/chunked-prefill/128seq/10240token/200admission/.8。新source分支/manifest与明确workload kind，旧性能/质量/诊断runtime不改。先CPU/真实operator及原生API门槛，再一次同GPU双臂完整258；计成功服务req/s、实际token/s、首可见工具内容延迟、末SSE完成延迟及官方参数匹配正确的完成率。原AIPerf指标与工具/完整完成口径分别留存；不是统一的语义goodput或用户已接受的SLO。

当前没有此任务GPU许可之外的额外权限要求，仍按已授权空闲按需范围和第一12GPU-h未重置余量决定预算。预估1卡≤1GPU-h/3GiB只是候选，真实派发前必须准入实现/CPU通过、读账本与全部稳定空闲/归属/预约/UUID/容器/共享资源/watchdog。先收完并分析c组件证据，再判断本任务和实际瓶颈修复的顺序，不机械铺矩阵。

2026-10-27实现更新：controller/native_tools提供完整原始SSE tool delta/ID/JSON参数/usage/末SSE时钟审计，HTTP200纯文本拒绝/无调用/错误全保留，官方匹配在本地独立算。187CPU/25冻结资产/frozenpolicy及live S0通过，GPU尚未由本段宣称启动。独立新run prod-tools-bf16xqa-bfcl258-c128-r0-20261027a，3600s/1GPU-h/3GiB。复用实际clean source11e7cddb及其bundle；所有推理/API源字节与1c2一致、C++与ceff二进制源码一致，因此无需为原生API测试改业务源码/重编译。source identity 首个CPU preparer检查因旧expected1c2拒绝，未写目录/未派发；已改为精确11e7+逐文件1c2核验。

CPU评分完整性限制：258参考中256个可构造的合法例通过原官方checker；live_simple_106-63-0与112-68-0共7个required array参数的参考候选为[]，官方type/value逻辑无法匹配，两个记录及原258分母保留，不改答案或输入/不按答案剔除。首版CPU夹具还把可省略的timespan空串sentinel当成实际整数值，已只修夹具的optional省略、未改官方checker/模型payload；旧失败v1保留，v3全部契约检查通过。这不是模型得分或完整agent认证。

## 流式门槛失败与完整JSON独立协议（2026-10-27）

a在首条独立warm原生工具请求上失败：真实SSE按标准delta拼接得到重复参数后缀，严格JSON无效。55文件875600bytes逐SHA镜像和原始stream已独立核验，Graph/算子通过但无AIPerf/FP8臂；rootFAILED及0.096657004GPU-h/driver释放保留。内部模型输出与stream parser的具体责任未复现证明，不进行客户端补字、去重或按答案修正。

新b `prod-tools-bf16xqa-bfcl258-buffered-c128-r0-20261027b` 在执行前明确选择已有生产完整JSON工具调用接口。保留全部258+16原消息、tools、greedy/自然EOS/cap1024、原顺序和服务配置，仅stream=false/去掉stream_options。父输入和逐payload SHA、变更字段及派生脚本均冻结；新dataset SHA47b4b1229449514dba089ab6f6de60a156e1f1df81b5e516b83fc493a2efbcbf，原输入不改。不是运行失败后回退，不把完整JSON结果认证为流式成功。

实际AIPerf loader/model/HTTP传输274sessions/548次loopback请求通过，dict和预编码两路径、单TextResponse及实际RawRecordInfo JSON序列化均核验；188控制器CPU通过，25冻结资产/实际policy/S0通过。CPU验证脚本的导入路径和dataclass API误用在GPU前修正，v5通过；不是服务端失败。正式时仍须每臂真实operator/Graph/完整JSON工具API/usage门槛，再执行完整AIPerf并独立官方匹配。完整JSON只报告完整请求延迟、actual tokens、成功服务率与官方匹配正确完成率，TTFT/ITL不作为流式指标。预算仍1卡/3600秒/1GPU-h/3GiB，第一12未重置，全部共享机安全门槛保持。
