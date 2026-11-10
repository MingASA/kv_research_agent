# Native KV 交付分支：FP8 PR 主线与独立 Q4 研发

本轮只整理实现、审查和推送；没有新增 GPU 作业，也没有重跑正式质量／服务矩阵。

## 三条有效交付分支

| 用途 | 分支 | 提交 |
| --- | --- | --- |
| FP8 PR 主线 | [pr/native-fp8-kv-cache](https://github.com/MingASA/xllm_quant_kvcache/tree/pr/native-fp8-kv-cache) | a00beb76a8a4b7aabc36ae34ea212371e8518418 |
| NVFP4 独立研发 | [research/native-nvfp4](https://github.com/MingASA/xllm_quant_kvcache/tree/research/native-nvfp4) | 3539bcad0c2df91dd70736a0e6d6e61f5195a425 |
| MXFP4 独立研发 | [research/native-mxfp4](https://github.com/MingASA/xllm_quant_kvcache/tree/research/native-mxfp4) | 3ee9b8e9989f065a54abbe850d526977b42d073f |

三条提交的作者和提交者均为 **MingASA**，使用该 GitHub 账号的 noreply 邮箱。
两个研发分支直接以 a00beb76 为父提交，彼此独立。旧 main 和历史实测分支均未覆盖／删除；这里的“PR 主线”明确指新分支 pr/native-fp8-kv-cache。

```text
官方固定 base 261274e5
└─ 独立 CUDA 构建 prerequisite 721afdab
   └─ FP8 PR 主线 a00beb76
      ├─ NVFP4 研发 3539bcad
      └─ MXFP4 研发 3ee9b8e9
```

[FP8 PR 差异](https://github.com/MingASA/xllm_quant_kvcache/compare/261274e5f45449e98d7740144c3a958a29f1fd3e...pr/native-fp8-kv-cache)。没有创建官方 PR；可供后续提 PR 的正文见 [PR_DESCRIPTION.md](PR_DESCRIPTION.md)。

## FP8 主线保留哪些基础设施

| 组件 | 主线决定 | 依据 |
| --- | --- | --- |
| K/V 实际 UInt8 分配、FP8 typed alias | 保留 | 保证真实一字节常驻，避免 BF16 Shadow；全页复制沿用原缓存生命周期 |
| 固定 K/V Scale 配置、JSON 与 Python Executor 透传 | 保留 | Writer 和 Attention 必须使用同一不可变 Scale；全局默认 1，模型配方显式选择 |
| Native FP8 容量与零逐 Token INT8 Scale 槽核算 | 保留 | 与实际字节分配一致，避免错误容量或重复 Scale 预算 |
| fused FP8 Writer、步幅感知 Query 转换 | 保留 | 与最终保留实测源码逐字节相同 |
| FA3 packed Prefill／普通混合批次、纯 Decode XQA | 保留 | 保持最终实测路由；已否决的混合拆分未带入 |
| XQA workspace 首次置零与稳定 paging metadata 适配 | 保留 | 既有库接口正确性与 Graph 生命周期要求；没有逐 Token memset |
| CUDA Python／模型形状／非 PD／非 Host Offload 等准入检查 | 保留 | 显式拒绝未支持组合，避免文档限制没有落实到入口 |
| 公共 Scheduler、Graph runner、Prefix Cache、GDN 管理 | 不改动 | 复用官方机制，不把公共优化收益计作量化收益 |
| Q4 packed head-dim helper、额外 capacity dtype 字段 | 移入对应研发分支 | FP8 逻辑 D 与物理 D 相同，主线不需要这些布局变更 |
| 非幂次 Packed 页的 VMM 几何和相关 Host 测试 | 移入研发 | 当前 CUDA Native VMM 不合格，不进入 FP8 PR；研发保留几何工作，不冒充 GPU VMM 验证 |
| BF16 backend 的 standalone import 调整 | 仅研发保留 | 便于独立算子资格验证；FP8 PR 的公共 BF16 backend 与官方逐字节相同 |
| Scale-prevalidated NV Writer／Host Metadata helper | NVFP4 研发中未启用 | 保留有价值候选及测试，未证明的优化不进入生产默认路径 |
| MXFP4 标准编码、Writer、私有 FA2 适配、Graph Mask 修复 | 仅 MXFP4 研发 | 保留标准 block32/E8M0 和最终 full planner；不混入 NVFP4 或拒绝后的快规划路径 |

FP8 主线相对 CUDA prerequisite 仅修改／新增 19 个文件：运行代码、正式测试与两份 CLI 接口文档。未带入本人的提示词、实验输入／数据／日志、交接文档、操作说明或本机环境文件。官方原有 README、AGENTS、CLAUDE 和贡献者规则原样保留，没有附加个人指令。

扫描主线全部 3,261 个非 third_party 的 tracked/new 文本文件，无尺寸截断；没有发现本机用户路径、E 盘路径、私有服务器路径／连接信息。扫描不读取依赖子模块内的第三方代码。硬件形状约束、数值上界和标准默认值不等同于本机路径硬编码。

## 检查与证据边界

| 分支 | CPU tests passed | subtests passed | skipped |
| --- | ---: | ---: | ---: |
| FP8 PR | 13 | 18 | 3 |
| NVFP4 研发 | 44 | 18 | 3 |
| MXFP4 研发 | 24 | 18 | 14 |

使用现有 Torch／pytest 环境，CUDA 隐藏。跳过的是 SM90 或依赖设备／FlashInfer 的检查，不能记作通过。全部改动 C++ 文件通过 clang-format 20.1.6；新增／改动运行 Python 和测试通过 Ruff、语法及 git diff whitespace 检查。MX 原始资格工具原样保存为研发档案，其既有 B023 closure/import 排序提示不包含在运行代码与测试的 Ruff 通过结论中。

FP8 Writer、Query 和 XQA 文件与最终保留版本 3b49556a 逐字节一致，prepare AST 一致，execute 在选择原有 FP8 分支后 AST 一致。研发分支保持同样的三个 FP8 文件；MX 模块与 6e99ad95 的计算／源码字符串 AST 一致，仅整理 import 顺序。详细核对见配套 JSON。

本次未重新构建完整 C++/CUDA 服务：本地缺少 native export 与全套 C++ 构建依赖。历史实测 binary/source、正式质量与服务数据仍按原身份引用；分支拆分不能冒充新 binary 或新 Graph／模型验证。进入 upstream 合并前需要正常 C++/CUDA build/CI 与相应集成检查。本轮产物是可审查的 **FP8-only PR 候选实现**。

完整历史质量／性能结论见 [FP8/NVFP4/MXFP4 报告](https://github.com/MingASA/kv_research_agent/tree/research/mxfp4-fp8-20261105/reports/experiments/mxfp4-fp8-20261105)，修复后 MX 见 [最终 MX 报告](https://github.com/MingASA/kv_research_agent/blob/bc260191a6f36f0a1517d4a2d1313856f2e58dab/reports/experiments/mxfp4-p0-20261109/REPORT.md)。

## 审查结论

### Strengths

- FP8 真实压缩常驻、Allocation 预算和 Scale 链路完整；核心实测计算路径保留。
- 已否决的混合路径不再启用；Q4 格式、私有后端适配和公共规划候选与 PR 隔离。
- 测试覆盖真实 CPU 页别名／复制、形状边界、执行分派，以及可在 SM90 运行的独立数值／Writer Replay 检查。

### Issues

- **Important — 构建／集成证据**：新抽取提交尚无完整 native C++/CUDA 构建或新模型实测；合并前需正常 build/CI，不以当前 CPU 结果替代。
- **Important — 质量／收益范围**：FP8 通过冻结质量点阈值，但 RepoBench-P 的统计非劣效仍不确定；ShareGPT 没有一致收益、GovReport 吞吐收益伴随 P95 ITL 增长。PR 描述保留这些限制。

### Assessment

Ready for PR review: **Yes**。Ready to merge without further build/integration checks: **No**。
FP8 是唯一进入生产 PR 审查的格式；NVFP4、MXFP4 保持研发支持级别。没有重新搜索配置或更改冻结评价标准。
