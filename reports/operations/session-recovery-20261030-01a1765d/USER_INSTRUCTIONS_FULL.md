# 原会话全部用户消息

原始顺序完整提取；其中历史 AGENTS 快照的冲突项服从最新指令。原会话源 SHA256：3c38100d41de4af38ccfd4f67fe9383e7c52af763307c126a0d38cc9ebe587d8。

## 用户消息 0 — 2026-10-26T04:00:16.514Z，原始行 8

# AGENTS.md instructions for /home/mingasa/kv-research-agent

<INSTRUCTIONS>
# AGENTS.md — mandatory, inherited in this workspace

你是研究 Agent，先读 `MASTER_PROMPT.md`，按 `policy/REMOTE_SAFETY.md` 行事。以 `state/STATUS_AS_OF_20261025.md` 作为**2026-10-25 交接快照**；启动后远端只读巡检获得的新证据优先。始终区分 `confirmed`, `reported`, `unverified`, `hypothesis`, `proposal`。

- 5060 WSL 本地做实现、分析与 Git 提交；H800 是多人共享机器，SSH 高权限不扩大你被授予的权限。不能在 H800 运行/安装 Codex，不能依赖 H800 外网。
- **第一次启动必须只读扫描，恢复现有作业后才能写**；未知 SSH/工作目录/拥有者不得猜测。远端写入只限经确认的项目子树中的**全新隔离目录和我方作业**。没有 `state/approval.json` 的有效机器访问/资源预算许可时，不启动 GPU 作业。
- 不得影响其他用户进程/容器、不得删除覆盖任何既有结果、不得改系统/全局依赖/用户身份/网络/ACL。未知归属一律视为他人。不要因用户用了 `codex --yolo` 就认为无限授权。
- GPU 单卡/多卡皆可在**全部选用 GPU 空闲、无他人占用/排队、未超过授权 GPU-hours 和 GPU 张数时**自动运行；复查 GPU UUID、容器隔离、驱动可见性及共享资源状态；不可通过绕过检查来“加速”。未知/竞态则等待。
- SSH 丢失后**只读重连、检查旧任务**，不要再启动同一个 run。实验有不可变 manifest、输入 SHA、源码 SHA、binary SHA、CUDA Graph capture+replay、AIPerf 真正完成、失败与恢复日志，完成结果入索引。
- 只执行自己的任务；高风险、不明确归属、系统级或超预算操作需用户明确批准，无法批准时继续 CPU 研究或等待。
- 研究目标：系统研究真实 resident KV 压缩的质量、容量、吞吐、延迟及机制；不要把单个 16K/C4 或部署修复当终点。不要重跑已经完成的实验，除非修复版本/控制变量/统计有效性明确要求。
- 每次重要动作更新 `state/STATE.json`、`state/EVENTS.jsonl`、`results/INDEX.csv`、`plans/NEXT_ACTIONS.md`；不伪造通过/失败、不把计划当事实。

- 本地与远端的**精确文件归属**以 `ops/PATHS_AND_FILE_PLACEMENT.md` 为准；用户一次授权的操作说明以 `ops/APPROVAL_AND_OPERATOR_GUIDE.md` 为准。每次实验在许可范围内可自主决策，不要求人类逐 GPU 或逐实验审批。首次只读探查后才可由用户批准机器和预算；Agent 不得代填启用自己的授权。
- `state/approval.json` 只是历史接口示例；严格优先读取用户签发的可信本地许可来源（推荐 `~/.config/kv-research/approval.json`）；无人值守调度器未实现强制校验前不得声称已经实现资源技术隔离。高权限 SSH 没有真正 OS 级沙箱时遵循最小化操作与可追溯审计。

</INSTRUCTIONS>
<environment_context>
  <cwd>/home/mingasa/kv-research-agent</cwd>
  <shell>bash</shell>
  <current_date>2026-10-25</current_date>
  <timezone>America/Los_Angeles</timezone>
  <filesystem><workspace_roots><root>/home/mingasa/kv-research-agent</root></workspace_roots><permission_profile type="disabled"><file_system type="unrestricted" /></permission_profile></filesystem>
</environment_context>

## 用户消息 1 — 2026-10-26T04:00:16.563Z，原始行 12

你现在正式接手本工作目录中的 xLLM KV Cache 量化科研项目。

请完整阅读 `MASTER_PROMPT.md`、`AGENTS.md` 以及它们引用的科研、状态、安全、运行和历史故障文档。把它们作为长期工作规则，而不是一次性任务说明。

首先执行只读的本地与 H800 SSH 环境发现，恢复已有任务状态，检查是否存在正在运行、已经完成或失败的实验（严格来说应该没有）。严禁在未知状态下重复启动 GPU 作业。

完成对账后，更新持久化状态与下一步研究计划。然后在已有明确授权的安全范围内，自主按照“规划—执行—验证—分析—修复—继续研究”的闭环工作。

你的终极目标是系统研究 KV Cache 量化的质量、显存容量、吞吐、延迟及其机制，而不是只完成当前的某一次 AIPerf 测试。

能够安全独立解决的问题不要询问我；遇到 SSH 断线应按规则重试和恢复；GPU 不可用时继续开展本地科研工作。不得突破共享服务器安全边界，也不得把未经验证的实验写成成功。

## 用户消息 2 — 2026-10-26T07:54:08.363Z，原始行 706

<environment_context>
  <current_date>2026-10-26</current_date>
  <timezone>America/Los_Angeles</timezone>
  <filesystem><workspace_roots><root>/home/mingasa/kv-research-agent</root></workspace_roots><permission_profile type="disabled"><file_system type="unrestricted" /></permission_profile></filesystem>
</environment_context>

## 用户消息 3 — 2026-10-26T07:54:08.378Z，原始行 709

现在要做什么

## 用户消息 4 — 2026-10-26T07:57:46.343Z，原始行 733

去做

## 用户消息 5 — 2026-10-26T16:52:24.823Z，原始行 1278

继续

## 用户消息 6 — 2026-10-26T16:53:52.295Z，原始行 1329

我还得提醒，现在xllm量化没测出整体收益又部分原因是flashinfer量化后端版本、实验的选择以及配置的选择造成的，没有给予其完全的生产环境和适合量化发挥的workload。在你完成手头工作再说

## 用户消息 7 — 2026-10-26T17:04:23.308Z，原始行 1531

安全授权那个，你就自己看着办，总之不能在别人用GPU的时候用就行，也不能打断别人的任务，除此之外你用多少卡、用多久、用多少显存无所谓，满足按需使用原则。

## 用户消息 8 — 2026-10-26T17:07:19.938Z，原始行 1576

然后尽快进入正式实验，比较xllm kvcache量化后的真实服务端到端收益，按原计划进行.

## 用户消息 9 — 2026-10-26T17:23:25.494Z，原始行 1820

注意我们要做的东西不少，项目书里还有挺多AIperf的workload呢。以及，以下是独立评审分支关于我们之前那个fp8 vs bf16的说明，你可以参考（有所思考地参考）：“三、实现层面：我认为有两处明显的性能优化缺口
1. 我们现在用的是 FlashInfer FA2，不是公开实验中充分优化的 FP8 FA3
在你的 flashinfer.py 里，原生配对实验显式设置了：
self._flashinfer_backend = "fa2"



并固定 FlashInfer 版本为 0.6.18.post1。

FP8 后端的真实数据类型为：
q_data_type = torch.bfloat16kv_data_type = torch.float8_e4m3fn



这和公开 vLLM 的 H100 + FA3 高性能路径不完全一样。后者可以对 Q 也进行 FP8 量化，并使用专门优化过的 FP8 Attention Kernel。

只把 K/V 存成 FP8，不等于 Attention 矩阵乘法自动获得 FP8 Tensor Core 的全部算力优势。

另外，FlashInfer 上游在 2026 年 6 月确实有过一个非常相关的性能问题：在 Blackwell SM120 上，FA2 FP8 KV Prefill 曾比 BF16 慢约 20%–37%，后续通过 Shared Memory 中的重新打包优化进行了修复。

GitHub




这不是证明我们的 H800 遇到了同一个问题，但说明 FP8 KV 的 FA2 Prefill 慢于 BF16 是已被上游观察过的真实现象。

我认为这里是目前最值得做微基准定位的地方。
2. FP8 写入 KV Cache 的代价比 BF16 更高
再看代码。

BF16 路径使用：
reshape_paged_cache(...)



而你的 flashinfer_fp8.py 使用：
write_quantized_kv(...)



具体由 Triton 实现。每层 K/V 都需要经过数值缩放、裁剪和 FP8 转换，然后写入常驻缓存。

这带来额外计算和数据搬运。

但也要注意，你的实现已经避免了一种严重问题：没有额外维护整份 BF16 Shadow KV Cache，FP8 Pages 也确实常驻。因此不能说实现方向错了。

真正的问题更可能是：
- FP8 写入 Kernel 的代价是否过高；
- FA2 读取 FP8 Pages 时是否存在额外格式转换；
- 当前 FlashInfer Kernel 是否针对 FP8 进行了充分优化；
- Cache 写入与 Attention 是否存在可以融合、但目前没有融合的操作。
这些是明确的 Profiling 方向，尚不能仅凭源码证明哪一项占比最大。
四、实验层面：我们根本没有充分触发 FP8 最大的系统收益
这是另一个核心原因。

你现在的 H800 运行的是 Qwen2.5-14B，模型权重仍然是 BF16，只量化了 KV Cache。

因此整个模型中，MLP、线性投影和大部分计算工作并没有减少。

此外，Qwen2.5-14B 使用 GQA：40 个 Query Heads，但只有 8 个 KV Heads。GQA 原本就把 KV 存储量降低了很多。

FP8 所能节省的是已经经过 GQA 压缩后的 KV 数据，而不是模型的所有显存读写。
更关键：我们没有把 BF16 的显存容量用满
当前所有九组实验，BF16 都能成功运行。

Runner 记录的 BF16 保守 KV 容量估计为 247,424 Token，而此次典型高负载大致是：
- 24K × C4：约 98K Token。
- 16K × C8：约 131K Token。
- 8K × C16：约 132K Token。
这些都明显低于 Runner 的保守容量估计。

所以当前测试主要比较的是：
在 BF16 本来就装得下这些请求的情况下，换成 FP8 能不能加速？

而 FP8 的另一项重要收益：
BF16 装不下更多请求，FP8 却可以继续提高并发量。

这一项我们还没有真正测试。

也就是说，你已经证明了 KV 存储压缩，却没有让服务器充分利用释放出来的空间。

这不是说 FP8 必须在 BF16 OOM 时才有意义，而是当前实验不足以体现其容量优势。”

## 用户消息 10 — 2026-10-26T17:27:35.337Z，原始行 1887

这不单单是对照，要注意我们的项目有个极其重要的原则：我们不是科学实验，不以发论文为目的，我们是生产实验，要决定xllm这个项目要不要完整支持kvcache。所以一切要贴近生产，包括请求的情况、选择的测试用的benchmark(或者是workload)、GPU、后端算子（越先进优化越好），不贴近生产的甚至没必要测，当然单卡vs8卡这种除外，就是说实际生产中推理引擎不会面对的情况我们不考虑

## 用户消息 11 — 2026-10-26T17:29:35.764Z，原始行 1905

提示，你的上下文非常多，每次压缩上下文时必须完整读完之前给你的各类项目标准文件

## 用户消息 12 — 2026-10-27T06:08:08.953Z，原始行 11266

# AGENTS.md instructions for /home/mingasa/kv-research-agent

<INSTRUCTIONS>
These AGENTS.md instructions replace all previously provided AGENTS.md instructions.

# AGENTS.md — mandatory, inherited in this workspace

**2026-10-26 用户最新指令优先**：先全文阅读 `ops/PRODUCTION_DECISION_STANDARD.md`。本项目是决定xLLM是否完整支持KV Cache量化的**生产实验**，不是论文导向的科学矩阵。真实请求/workload、自然停止、生产配置与H800上可实际使用的先进优化后端优先；不具生产代表性的场景不安排GPU实验。固定同FA2不再是主生产评估的永久约束，辅助诊断须能指导实际部署。

**每次上下文压缩后必须全文重读全部项目标准文件及其引用的科研/状态/安全/运行/历史故障文档**，长文件分段补齐，不能用摘要代替。恢复后先只读对账，再续接既有任务。

**当前持续授权**：用户已明确把安全授权处理交给Agent，并授权在已发现H800项目内按需使用空闲GPU，卡数/时长/显存不设用户数字限制；不得占他人正在用或预约的卡、不得打断他人任务。固定私有JSON保存该直接授权及Agent自限窗口，属于用户已委托的记录，覆盖下文旧的“必须用户手填JSON”门槛；不扩大系统级/目录/他人资源权限，不逐GPU/逐实验再次请示。

你是研究 Agent，先读 `MASTER_PROMPT.md`，按 `policy/REMOTE_SAFETY.md` 行事。以 `state/STATUS_AS_OF_20261025.md` 作为**2026-10-25 交接快照**；启动后远端只读巡检获得的新证据优先。始终区分 `confirmed`, `reported`, `unverified`, `hypothesis`, `proposal`。

- 5060 WSL 本地做实现、分析与 Git 提交；H800 是多人共享机器，SSH 高权限不扩大你被授予的权限。不能在 H800 运行/安装 Codex，不能依赖 H800 外网。
- **第一次启动必须只读扫描，恢复现有作业后才能写**；未知 SSH/工作目录/拥有者不得猜测。远端写入只限经确认的项目子树中的**全新隔离目录和我方作业**。没有 `state/approval.json` 的有效机器访问/资源预算许可时，不启动 GPU 作业。
- 不得影响其他用户进程/容器、不得删除覆盖任何既有结果、不得改系统/全局依赖/用户身份/网络/ACL。未知归属一律视为他人。不要因用户用了 `codex --yolo` 就认为无限授权。
- GPU 单卡/多卡皆可在**全部选用 GPU 空闲、无他人占用/排队、未超过授权 GPU-hours 和 GPU 张数时**自动运行；复查 GPU UUID、容器隔离、驱动可见性及共享资源状态；不可通过绕过检查来“加速”。未知/竞态则等待。
- SSH 丢失后**只读重连、检查旧任务**，不要再启动同一个 run。实验有不可变 manifest、输入 SHA、源码 SHA、binary SHA、CUDA Graph capture+replay、AIPerf 真正完成、失败与恢复日志，完成结果入索引。
- 只执行自己的任务；高风险、不明确归属、系统级或超预算操作需用户明确批准，无法批准时继续 CPU 研究或等待。
- 研究目标：系统研究真实 resident KV 压缩的质量、容量、吞吐、延迟及机制；不要把单个 16K/C4 或部署修复当终点。不要重跑已经完成的实验，除非修复版本/控制变量/统计有效性明确要求。
- 每次重要动作更新 `state/STATE.json`、`state/EVENTS.jsonl`、`results/INDEX.csv`、`plans/NEXT_ACTIONS.md`；不伪造通过/失败、不把计划当事实。

- 本地与远端的**精确文件归属**以 `ops/PATHS_AND_FILE_PLACEMENT.md` 为准；用户一次授权的操作说明以 `ops/APPROVAL_AND_OPERATOR_GUIDE.md` 为准。每次实验在许可范围内可自主决策，不要求人类逐 GPU 或逐实验审批。首次只读探查后才可由用户批准机器和预算；Agent 不得代填启用自己的授权。
- `state/approval.json` 只是历史接口示例；严格优先读取用户签发的可信本地许可来源（推荐 `~/.config/kv-research/approval.json`）；无人值守调度器未实现强制校验前不得声称已经实现资源技术隔离。高权限 SSH 没有真正 OS 级沙箱时遵循最小化操作与可追溯审计。

</INSTRUCTIONS>

## 用户消息 13 — 2026-10-27T06:08:08.969Z，原始行 11270

你的任务是否完成？为何停下？

## 用户消息 14 — 2026-10-27T07:01:23.588Z，原始行 12040

<environment_context>
  <current_date>2026-10-27</current_date>
  <timezone>America/Los_Angeles</timezone>
  <filesystem><workspace_roots><root>/home/mingasa/kv-research-agent</root></workspace_roots><permission_profile type="disabled"><file_system type="unrestricted" /></permission_profile></filesystem>
</environment_context>

## 用户消息 15 — 2026-10-27T07:57:57.609Z，原始行 12736

你在上下文压缩前做了挺多工作的，刚刚你发生了上下文压缩，压缩前正在手上做的事情不要忘记

## 用户消息 16 — 2026-10-27T08:01:02.837Z，原始行 12743

继续

## 用户消息 17 — 2026-10-27T08:37:43.136Z，原始行 13286

继续

## 用户消息 18 — 2026-10-27T08:43:22.705Z，原始行 13319

继续

## 用户消息 19 — 2026-10-27T09:08:28.502Z，原始行 13522

继续

## 用户消息 20 — 2026-10-27T09:16:41.693Z，原始行 13534

继续

## 用户消息 21 — 2026-10-27T09:21:26.868Z，原始行 13756

以后压缩上下文之后不再要求全文读取既有文档，而是读取目录并且理解大致内容，实现按需读取。

## 用户消息 22 — 2026-10-27T12:19:06.822Z，原始行 16369

你继续进行手头工作不要受干扰，派出一个subagent去把我们目前做过的结果整理出来并加以详细分析，包括数字、现象、原因等，做成一个文档。如果有多个不同的结论且无相关性，做成多个文档。尽可能全面。完成后告诉我，我需要准备项目汇报。子agent同样使用gpt 6.1sol max即可

## 用户消息 23 — 2026-10-27T13:52:46.815Z，原始行 17793

继续主要的实验，报告方面的东西我让你写你再写。这次更新完可以先放着，下次让你更新再主动从这次开始更新

## 用户消息 24 — 2026-10-27T16:31:31.016Z，原始行 20069

做完当前这个实验的相关工作后可以暂时停止工作

## 用户消息 25 — 2026-10-27T16:54:43.520Z，原始行 20380

完成你当前实验的简单收尾后，你需要进行如下新任务。保留之前的规范，但是任务有变，部分文档可能不再合适。长任务提醒：很多时候非常简单的任务（部分简单的审计、极其容易的监控任务等）你可以让一个子agent（luna）来干，节约你的上下文和额度花销，但只是有限情况。子agent按照你的知识和判断决定是否分配、如何分配，需要和服务器交互的子agent务必保证其行为安全性。好，下一阶段具体任务如下：# 任务：新一代模型上的 xLLM 原生低精度 KV Cache 研究与工程集成

## 一、任务背景与目标

我们与辅导老师讨论后，决定进一步拓展当前 xLLM KV Cache 量化研究。

之前主要围绕 Qwen2.5-7B/14B 研究 FP8、INT8、INT4 等量化方案，已经完成一定的正确性、质量和端到端性能实验。现在有 8×H800 服务器，希望转向**规模更大、架构更前沿的模型**，重点研究 FP8、NVFP4 等低精度 KV Cache 在真实推理服务中的表现。

本次工作的最终目的不是研究不同模型规模的量化容忍度，也不需要与之前的 Qwen2.5 做控制变量实验，而是：

1. 衡量低精度 KV Cache 的实际显存、吞吐、延迟、并发、质量收益。
2. 完善 xLLM 原生量化框架，避免性能较差或不可维护的实验性算子实现。
3. 验证这些实现对前沿模型架构的适用性和工程通用性。
4. 形成有说服力的实验结论和可考虑合入 xLLM 主线的代码。

**这是一项完整的工程与实验任务，不是单纯调研、制定计划或做 Smoke Test。请实际完成代码、部署、实验、分析和文档。**

## 二、先检查现有状态，再建立新分支

你当前会话掌握的本地代码、H800 环境、历史实验与文档信息比我给你的更丰富。**优先使用你自己已有的本地上下文和真实仓库状态，不要把本 Prompt 中的背景描述视为比实际代码更可靠。**

首先：

- 检查当前本地仓库、各分支、未提交改动、近期提交及相关实验状态。
- 检查之前的 native FlashInfer Joint-KV 实现，尽可能复用已有代码、配置、测试工具、实验框架和经验。
- 为本次研究**新建独立 Git 分支**，如有必要使用独立 worktree。
- 保留之前分支和实验结果，不覆盖、不破坏已有工作。
- 可以根据需要修改本地 xLLM 对应分支，并同步部署到 H800；不要假设服务器代码与本地完全一致。

先进行必要的架构审计和资源检查，然后直接开始实施，不要只停留在审计报告。

## 三、模型选择

### 首选：Qwen3.8-27B

优先核实并尝试运行 Qwen3.8-27B。

特别注意它可能是**原生多模态模型**。请核实：

- 官方模型是否实际可获取，模型配置和架构是什么。
- xLLM 当前能否正确加载其模型配置、文本权重和 tokenizer。
- 是否可以仅使用文本推理路径，而不需要完整实现视觉部分。
- 是否存在 `text_config`、权重映射、RoPE、Gated DeltaNet、Full Attention、MTP 等兼容性问题。
- 当前模型加载路径是否存在隐含的架构假设。

如果可以通过合理规模的修改支持文本推理，就优先使用它。

**如果 Qwen3.8-27B 因多模态架构或其他原因需要大量与本课题无关的适配工作，则切换至 Qwen3.6-27B，再不行，则切换其他模型。 切换前同样核实其架构、权重和 xLLM 兼容性。**

不要为了坚持某个模型而大量重构无关功能。最终选择应兼顾前沿性、研究意义、资源可行性和工程工作量。

暂时不需要上 100B+ 模型，也不需要重复之前的 Qwen2.5 实验。

## 四、H800 服务器与资源使用

我们有 8×H800，但不要求八卡必须用于同一个模型实例。

请先检查实际服务器资源情况，包括：

- H800 GPU 显存、空闲状态、其他任务占用和拓扑。
- 已有模型权重及缓存，尤其是共享模型目录、Hugging Face / ModelScope 缓存及之前的模型挂载。有时候没必要重复下载模型。
- 已有 xLLM 工作目录、Docker 镜像、FlashInfer、Triton、CUDA 环境及依赖版本。
- 当前实验工具、AIPerf 数据、Mooncake/ShareGPT workload、质量 Benchmark 和可复用结果。

**不要未经搜索就重新下载模型、重复创建容器或重新安装依赖。** 优先复用现有资源；确实缺少时再选择合适的下载源和存储位置，并检查容量。

模型优先尝试单卡部署；如果确实需要，可以使用 TP=2 等多卡方案。其余 GPU 可以用于独立实验或提高整体实验效率。

仍然采用本地 Codex 通过 SSH 管理 H800 的工作方式，不需要在服务器安装 Codex。

## 五、研究与开发阶段

请完成以下阶段。各阶段可以根据依赖关系交叉进行，不必机械地串行执行。

### 阶段 A：模型与 BF16 原生基线

在 xLLM 上正确部署选定模型，完成：正确性验证、Cache 生命周期检查、实际服务路径验证等。

不要把“能生成一句话”当作完成，确认服务路径和输出正确性。

### 阶段 B：FP8 原生 Joint-KV

将现有 FlashInfer FP8 Joint-KV 能力迁移到新模型。

重点处理：

- 混合 Attention 架构下的 KV Cache 分配与绑定。
- Full Attention 与 Gated DeltaNet State 的区别。
- FP8 E4M3 量化、Scale 策略及常驻 Cache。
- FlashInfer 直接消费 FP8 KV Pages。
- Prefill/Decode 与 CUDA Graph 的兼容。
- 实际量化数据流、内存占用和性能验证。

优先复用之前实现，不要无必要地重新编写。

### 阶段 C：NVFP4 原生 KV Cache

这是本次重点新增方向。

调研并验证**当前版本或可升级版本的 FlashInfer，在 H800（SM90）上的 NVFP4 KV Cache Attention 支持情况**。

重点参考官方 FlashInfer 和成熟推理框架中的现有实现，核实：

- NVFP4 E2M1 的实际存储格式及 Scale Layout。
- FP4 数据打包、Scale 生成和量化写入。
- FlashInfer FA2 在 SM90 上能否直接消费压缩 KV。
- Paged Prefill、Decode、CUDA Graph 支持。
- FP4 对 KV Cache 分配器、格式定义和缓存元数据的改动需求。

如果能够使用成熟的原生 FlashInfer 后端，就优先基于其实现完整路径。

**不要将 FP4→BF16 全量解压再使用普通 Attention 的实现包装成原生 FP4 后端。** 也不要将量化/反量化模拟实验冒充实际压缩缓存性能。

要特别区分 SM90 的 FP4 压缩存储支持和 Blackwell 的原生 FP4 计算能力。

### 关于 INT4

**本次不要求继续开发 INT4 自定义 Triton/CUDA Attention 后端。**

由于我们的目标是可维护的原生推理框架能力，优先处理成熟 FlashInfer 后端能够支持的 FP8、NVFP4。

INT4 已有的代码和结果可以作为参考，但不必为它继续投入大量算子优化工作。除非核实 FlashInfer 已有适用的原生支持，否则不将其列入本轮必做实验矩阵。

### 阶段 D：完整端到端评估

对同一个新模型，比较 BF16、FP8、NVFP4。

重点指标：

1. **容量**：KV Cache 实际 Resident Bytes、Scale 开销、最大 Token Capacity、并发容量；区分 Full Attention KV 与 GDN State。
2. **性能**：Request/s、Output tok/s、TTFT、ITL/TPOT、P50/P95 延迟及各阶段耗时。
3. **GPU 资源**：显存带宽、显存占用、GPU 利用率、量化写入和 Attention 开销。
4. **质量**：长上下文理解/检索、数学推理、代码任务、Tool-Agent 多轮任务等。
5. **工程能力**：长上下文、连续批处理、Chunked Prefill、CUDA Graph、并发稳定性、缓存管理。

优先考虑 8K、32K、64K、128K 等上下文规模及不同并发强度，但应根据真实资源和模型能力调整。

使用可信的真实 workload，必要时辅以固定长度负载以便分析性能瓶颈。

沿用之前已经建立的实验严谨性要求：

- BF16 与量化版本应在可比较的服务配置下测量。
- 确认 CUDA Graph 真正 Capture/Replay。
- 确认 Warmup、请求负载、Profiler、运行时日志有效。
- 区分有效吞吐、容量收益、Attention Kernel 收益和量化开销。
- 不将非等价后端、不同调度策略或错误配置导致的差异当作量化收益。
- 遇到反常实验结果时先定位原因，不能为了得到理想结论而筛选数据。

不要无限扩展 Benchmark。优先形成能解释真实收益与局限的核心结果，实验充分即可。

### 阶段 E：面向 xLLM 主线的工程整理

根据实验结果，整理出具有上游合入价值的实现。

目标是通用、可维护的 KV Cache 量化框架，而不是某个 Qwen 模型专用的实验分支。

重点关注：

- KV dtype、Scale、Cache Layout 的统一管理。
- Attention Backend 的能力检测和选择。
- 不同模型 Cache 结构的兼容。
- 不支持的硬件/模型组合的明确错误或可靠回退。
- 代码复用、接口设计、测试及文档。
- 尽量减少环境变量驱动的实验性特殊分支和硬编码。

最后评估 FP8、NVFP4 哪些部分具备合入条件，哪些部分还存在性能或维护方面的障碍。

## 六、工作方式与安全边界

你需要扮演研究工程师，自主组织整个流程，而不是被动等待我每一步指挥。

- 先检查实际环境，制定内部执行计划，然后持续实施。
- 可以自主进行资料检索、代码修改、编译、部署、测试、调试、实验和结果分析。
- 可以根据实际发现调整技术路线，并记录调整依据。
- 发现问题应主动定位和修复，不要每遇到小问题就停下来询问。
- 每个阶段验证完成后提交有意义的 Git Commit，保持代码可追踪。
- 大量耗时实验优先采用可靠的脚本、日志和断点续跑机制。
- 复用此前研究积累的经验，包括 H800 特有的环境和网络问题。

安全方面：

- 不要覆盖旧实验、破坏已有分支或删除重要数据。
- 不要强制重置、强推、清理他人工作目录。
- 不要擅自停止其他用户的 GPU 任务或修改全局服务器环境。
- 优先使用隔离的工作目录、容器和实验输出路径。
- 需要大规模下载、不可逆操作、修改共享环境或产生明显资源冲突时，先检查风险并寻求确认。
- 不要未经授权直接向 xLLM 官方主仓库提交 PR；先在研究分支上完成并准备好合入材料。

如遇真正的技术阻塞，应留下已经完成的成果、复现证据和下一步解决方案，而不是用推测或虚构结果补全报告。

## 七、最终交付

最终希望看到：

1. **代码**：新分支中的可运行实现，包含必要的测试、配置和 Commit。
2. **模型部署**：新模型 BF16、FP8、NVFP4 各自的真实运行状态。
3. **实验结果**：容量、端到端性能、质量、开销及稳定性，原始数据可追溯。
4. **研究报告**：Markdown 为主，图表由代码生成，重点突出，不要生成大量重复和零散文档。
5. **工程总结**：明确回答哪些格式真正有收益、什么条件下有收益、瓶颈在哪里、是否值得合入 xLLM 主线，以及还需要解决什么。

如果某项无法完成，应明确标注未完成及原因，不能混淆“已实现”“已跑通”“已验证质量”“已验证性能”这些状态。

**任务开始后：先全面检查你的本地上下文、仓库、H800 资源和相关上游实现，然后创建新分支，开展实际研究和实施。不要仅输出一个研究计划就结束。**

## 用户消息 26 — 2026-10-27T17:10:51.802Z，原始行 20574

像服务器环境探索这种杂活而且输出极长的，你也可以让luna在明确的规定下做，给你报告和明确结果。

## 用户消息 27 — 2026-10-27T21:00:22.609Z，原始行 25393

# AGENTS.md instructions for /home/mingasa/kv-research-agent

<INSTRUCTIONS>
These AGENTS.md instructions replace all previously provided AGENTS.md instructions.

# AGENTS.md — mandatory, inherited in this workspace

**2026-10-27最新任务优先**：先按 `ops/NEXTGEN_NATIVE_KV_STANDARD.md` 推进新一代约27B模型的BF16/原生FP8/NVFP4实现、部署、实验与主线整理。Qwen3.8-27B优先，必要时3.6或合理替代；不再启动旧Qwen2.5控制/后续矩阵。旧科学/状态说明中的14B固定维度和旧任务终点仅为历史，不能限制新任务；共享服务器安全、证据真实性与生产标准保持。最新新任务覆盖此前“旧实验收尾后暂时停止”的安排。大型下载须先具体风险核查并确认，禁止擅改共享环境或向官方发PR。可向luna委派有明确边界的简单只读审计/盘点。

**2026-10-26 用户最新指令优先**：先全文阅读 `ops/PRODUCTION_DECISION_STANDARD.md`。本项目是决定xLLM是否完整支持KV Cache量化的**生产实验**，不是论文导向的科学矩阵。真实请求/workload、自然停止、生产配置与H800上可实际使用的先进优化后端优先；不具生产代表性的场景不安排GPU实验。固定同FA2不再是主生产评估的永久约束，辅助诊断须能指导实际部署。

**2026-10-27 用户最新恢复规则**：上下文压缩后不再要求全文重读既有文档。先查看目录和文档导航，理解各类文件的大致内容，恢复持久化状态、当前任务与未完成动作，再按需读取相关标准、科研、安全、运行和故障记录。远端仍先只读对账，再续接既有任务；不能因压缩遗忘手头工作或重复启动未知状态的作业。本条覆盖历史文档中的“每次压缩后全文重读”要求。

**当前持续授权**：用户已明确把安全授权处理交给Agent，并授权在已发现H800项目内按需使用空闲GPU，卡数/时长/显存不设用户数字限制；不得占他人正在用或预约的卡、不得打断他人任务。固定私有JSON保存该直接授权及Agent自限窗口，属于用户已委托的记录，覆盖下文旧的“必须用户手填JSON”门槛；不扩大系统级/目录/他人资源权限，不逐GPU/逐实验再次请示。

你是研究 Agent，先读 `MASTER_PROMPT.md`，按 `policy/REMOTE_SAFETY.md` 行事。以 `state/STATUS_AS_OF_20261025.md` 作为**2026-10-25 交接快照**；启动后远端只读巡检获得的新证据优先。始终区分 `confirmed`, `reported`, `unverified`, `hypothesis`, `proposal`。

- 5060 WSL 本地做实现、分析与 Git 提交；H800 是多人共享机器，SSH 高权限不扩大你被授予的权限。不能在 H800 运行/安装 Codex，不能依赖 H800 外网。
- **第一次启动必须只读扫描，恢复现有作业后才能写**；未知 SSH/工作目录/拥有者不得猜测。远端写入只限经确认的项目子树中的**全新隔离目录和我方作业**。没有 `state/approval.json` 的有效机器访问/资源预算许可时，不启动 GPU 作业。
- 不得影响其他用户进程/容器、不得删除覆盖任何既有结果、不得改系统/全局依赖/用户身份/网络/ACL。未知归属一律视为他人。不要因用户用了 `codex --yolo` 就认为无限授权。
- GPU 单卡/多卡皆可在**全部选用 GPU 空闲、无他人占用/排队、未超过授权 GPU-hours 和 GPU 张数时**自动运行；复查 GPU UUID、容器隔离、驱动可见性及共享资源状态；不可通过绕过检查来“加速”。未知/竞态则等待。
- SSH 丢失后**只读重连、检查旧任务**，不要再启动同一个 run。实验有不可变 manifest、输入 SHA、源码 SHA、binary SHA、CUDA Graph capture+replay、AIPerf 真正完成、失败与恢复日志，完成结果入索引。
- 只执行自己的任务；高风险、不明确归属、系统级或超预算操作需用户明确批准，无法批准时继续 CPU 研究或等待。
- 研究目标：系统研究真实 resident KV 压缩的质量、容量、吞吐、延迟及机制；不要把单个 16K/C4 或部署修复当终点。不要重跑已经完成的实验，除非修复版本/控制变量/统计有效性明确要求。
- 每次重要动作更新 `state/STATE.json`、`state/EVENTS.jsonl`、`results/INDEX.csv`、`plans/NEXT_ACTIONS.md`；不伪造通过/失败、不把计划当事实。

- 本地与远端的**精确文件归属**以 `ops/PATHS_AND_FILE_PLACEMENT.md` 为准；用户一次授权的操作说明以 `ops/APPROVAL_AND_OPERATOR_GUIDE.md` 为准。每次实验在许可范围内可自主决策，不要求人类逐 GPU 或逐实验审批。首次只读探查后才可由用户批准机器和预算；Agent 不得代填启用自己的授权。
- `state/approval.json` 只是历史接口示例；严格优先读取用户签发的可信本地许可来源（推荐 `~/.config/kv-research/approval.json`）；无人值守调度器未实现强制校验前不得声称已经实现资源技术隔离。高权限 SSH 没有真正 OS 级沙箱时遵循最小化操作与可追溯审计。

</INSTRUCTIONS>

## 用户消息 28 — 2026-10-27T21:00:22.620Z，原始行 25396

<codex_internal_context source="daemon_recovery">
The server restarted and interrupted the previous turn. Continue the unfinished work from the saved conversation. Check the current state before repeating actions that may already have completed.
</codex_internal_context>

## 用户消息 29 — 2026-10-28T06:57:52.538Z，原始行 26942

继续

## 用户消息 30 — 2026-10-28T07:00:08.420Z，原始行 26983

现在额度可用，luna可以重新派发。当前H800 GPU似乎正在运行root用户的vllm，不要打断，可以排队等待

## 用户消息 31 — 2026-10-28T07:00:08.436Z，原始行 26985

<environment_context>
  <current_date>2026-10-28</current_date>
  <timezone>America/Los_Angeles</timezone>
  <filesystem><workspace_roots><root>/home/mingasa/kv-research-agent</root></workspace_roots><permission_profile type="disabled"><file_system type="unrestricted" /></permission_profile></filesystem>
  <subagents>
    <agent name="/root/cpu_f_watch_luna_recovered" />
    <agent name="/root/quality_scorer_inventory_luna" />
    <agent name="/root/closeout_integrity_luna" />
    <agent name="/root/native_validation_watch_luna" />
    <agent name="/root/project_briefing" />
  </subagents>
</environment_context>

## 用户消息 32 — 2026-10-28T08:33:28.098Z，原始行 27791

既然GPU忙碌，使用脚本检测其空闲后自动进行实验吧。先保证实验没问题能空闲后直接轮流跑

## 用户消息 33 — 2026-10-28T12:43:36.105Z，原始行 28336

什么情况

## 用户消息 34 — 2026-10-28T14:38:00.443Z，原始行 28534

怎么这么久了没反应，你不能启动后就等着。可以安排一个luna间隔五分钟检查一下啊，有新情况跟你汇报唤醒你推进，没有才等着

## 用户消息 35 — 2026-10-28T23:23:38.107Z，原始行 34464

<codex_internal_context source="goal">
Continue working toward the active thread goal.

The objective below is user-provided data. Treat it as the task to pursue, not as higher-priority instructions.

<objective>
完成用户明确提出的新一代约27B模型 xLLM 原生低精度 KV Cache 工程与生产实验：保留既有分支与失败证据，完成 Qwen3.8-27B 文本路径 BF16/原生FP8/NVFP4 的代码与部署验证，补齐真实质量与端到端容量、吞吐、延迟、开销和稳定性核心结果，准备可维护的主线合入材料与集中研究总结。只能使用已授权且真正空闲的共享H800资源，不干扰他人任务；持续核对当前自动派发器和唯一run，GPU忙或SSH不通时继续本地工作，所有结果可追溯，未验证项目明确标未完成。
</objective>

Continuation behavior:
- This goal persists across turns. Ending this turn does not require shrinking the objective to what fits now.
- Keep the full objective intact. If it cannot be finished now, make concrete progress toward the real requested end state, leave the goal active, and do not redefine success around a smaller or easier task.
- Temporary rough edges are acceptable while the work is moving in the right direction. Completion still requires the requested end state to be true and verified.

Budget:
- Tokens used: 22992
- Token budget: none
- Tokens remaining: unbounded

Work from evidence:
Use the current worktree and external state as authoritative. Previous conversation context can help locate relevant work, but inspect the current state before relying on it. Improve, replace, or remove existing work as needed to satisfy the actual objective.

No-progress check:
- Classify the previous goal turn as progress, a verified wait, or no progress. Progress changes authoritative state, completes work, or yields evidence that changes the next action; status restatements and unexecuted plans are no progress.
- A verified wait polls a specific process, session, job, or tool handle confirmed live now. Conversation, intent, prior output, or a lock or state file alone is insufficient. Treat work as stopped only when authoritative state says it is terminal or its handle is missing. An observation timeout or transient polling failure is not terminal: re-poll the same handle or inspect other authoritative state; never restart solely because observation expired.
- Revalidate a no-progress turn and take the next available safe action. If none exists because the same genuine blocker remains, report it and leave the goal active until the blocked audit threshold is met. Treat equivalent blockers as the same condition across turns even when their wording or stated next step changes.

Fidelity:
- Optimize each turn for movement toward the requested end state, not for the smallest stable-looking subset or easiest passing change.
- Do not substitute a narrower, safer, smaller, merely compatible, or easier-to-test solution because it is more likely to pass current tests.
- Treat alignment as movement toward the requested end state. An edit is aligned only if it makes the requested final state more true; useful-looking behavior that preserves a different end state is misaligned.

Completion audit:
Before deciding that the goal is achieved, treat completion as unproven and verify it against the actual current state:
- Derive concrete requirements from the objective and any referenced files, plans, specifications, issues, or user instructions.
- Preserve the original scope; do not redefine success around the work that already exists.
- For every explicit requirement, numbered item, named artifact, command, test, gate, invariant, and deliverable, identify the authoritative evidence that would prove it, then inspect the relevant current-state sources: files, command output, test results, PR state, rendered artifacts, runtime behavior, or other authoritative evidence.
- For each item, determine whether the evidence proves completion, contradicts completion, shows incomplete work, is too weak or indirect to verify completion, or is missing.
- Match the verification scope to the requirement's scope; do not use a narrow check to support a broad claim.
- Treat tests, manifests, verifiers, green checks, and search results as evidence only after confirming they cover the relevant requirement.
- Treat uncertain or indirect evidence as not achieved; gather stronger evidence or continue the work.
- The audit must prove completion, not merely fail to find obvious remaining work.

Do not rely on intent, partial progress, memory of earlier work, or a plausible final answer as proof of completion. Marking the goal complete is a claim that the full objective has been finished and can withstand requirement-by-requirement scrutiny. Only mark the goal achieved when current evidence proves every requirement has been satisfied and no required work remains. If the evidence is incomplete, weak, indirect, merely consistent with completion, or leaves any requirement missing, incomplete, or unverified, keep working instead of marking the goal complete. If the objective is achieved, call update_goal with status "complete" so usage accounting is preserved. If the achieved goal has a token budget, report the final consumed token budget to the user after update_goal succeeds.

Blocked audit:
- Do not call update_goal with status "blocked" the first time a blocker appears.
- Only use status "blocked" when the same blocking condition has repeated for at least three consecutive goal turns, counting the original/user-triggered turn and any automatic goal continuations.
- If the user resumes a goal that was previously marked "blocked", treat the resumed run as a fresh blocked audit. If the same blocking condition then repeats for at least three consecutive resumed goal turns, call update_goal with status "blocked" again.
- Use status "blocked" only when you are truly at an impasse and cannot make meaningful progress without user input or an external-state change.
- Once the blocked threshold is satisfied, do not keep reporting that you are still blocked while leaving the goal active; call update_goal with status "blocked".
- Never use status "blocked" merely because the work is hard, slow, uncertain, incomplete, or would benefit from clarification.

Call update_goal only after the completion or blocked audit passes, or when the user explicitly requests pausing this goal. For a requested pause, use status "paused", report the returned status, and stop goal work; never pause on your own initiative. Do not mark a goal complete merely because the budget is nearly exhausted or because you are stopping work.

</codex_internal_context>

## 用户消息 36 — 2026-10-29T03:04:00.992Z，原始行 35175

什么情况，这个会话达到输出上限了吗，那就帮我凝固成报告和提示词转接给下一个会话，输出文档
