# 本地证据交付目录

以下九个分区的实际gzip与文件清单SHA均已重新核对；每个分区有写包后逐成员重读回执。原始FAILED保留。

| 分区 | 成员 | gzip字节 | SHA-256 |
|---|---:|---:|---|
| [quality-a1](quality-a1/ARCHIVE_ACCEPTANCE.json) | 342 | 92731072 | `2d64a53410634d203810159950b8bd2f70bc10f98338205f4a030679db9141eb` |
| [service-original-a2](service-original-a2/ARCHIVE_ACCEPTANCE.json) | 1109 | 69959131 | `9f4463510f538af2f9ba0eaaadfc7638d160342da3189fef7927ead0826d57aa` |
| [tools-original-a2](tools-original-a2/ARCHIVE_ACCEPTANCE.json) | 1214 | 21033830 | `e26be8e20a26db2828da78b975de6c64977604abdf2267c6135f596a6e1d4a25` |
| [mixed-pilot-a1](mixed-pilot-a1/ARCHIVE_ACCEPTANCE.json) | 123 | 12822104 | `5a81e704eba77b7f120db0dd583cb8481f681eaab0f622fe15fd72eecee491ed` |
| [local-failure-index-a1](local-failure-index-a1/ARCHIVE_ACCEPTANCE.json) | 27 | 61670 | `a1c2d40e62d1b6695193c9484558dd977af30972502cea342ff93a88d3236886` |
| [mixed-quality-a1](mixed-quality-a1/ARCHIVE_ACCEPTANCE.json) | 289 | 53890603 | `d96c90e28ed815421bd40b6cecc7a177c5b9b4096d63642fa5087220d54e4f7b` |
| [session-protocol-failure-and-repair-a1](session-protocol-failure-and-repair-a1/ARCHIVE_ACCEPTANCE.json) | 205 | 10649504 | `46322019721ec078a728012a0c38a80d0819f42caae669edac3a666009fc4253` |
| [mixed-pressure-a1](mixed-pressure-a1/ARCHIVE_ACCEPTANCE.json) | 406 | 25008811 | `15f0032a7ce31524855d64354c8f8e7ec72b0ff45005ec464f5ef88723b44e02` |
| [mixed-tools-a1](mixed-tools-a1/ARCHIVE_ACCEPTANCE.json) | 1182 | 17462604 | `f8c3d8d09dceea87542b77e2a9467cf68e716e7ae8b78ea57cbb0010fc84229b` |

local-failure-index只包含历史失败索引与选定分析，不表示全部历史失败raw均打包。

源码与归档分开：模型patch/增量bundle固定11e7cddb..03134b72，需要已有base；控制器完整bundle无prerequisite。
控制器固定commit：`4f07cfde8401d4d3ac66553ac4e524ddc60b666f`。最终catalog与状态收尾元数据在后续Git提交中，不冒充已包含在该bundle。
不额外收集权重、安装环境、固定私有授权源文件或完整原会话内部日志；原票据中的授权快照随本地原始证据保留。这不是可直接部署镜像。
