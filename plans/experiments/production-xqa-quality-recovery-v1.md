# XQA完整质量：仅补缺失BF16参考

当前结果已完成：missingBF16 b真实COMPLETE/driver释放，106文件42,334,503bytes逐SHA镜像。c80+a恢复FP8的1639+b缺失BF16的1639已独立按原索引重评分每臂1719；报告production-xqa-quality-full1719-20261026。GSM flexible+0.4549pp、LongBench macro−1.3358pp，未经生产质量接受。三批共0.829428522GPU-h。以下恢复设计与旧执行前资源数字保留，不表示仍在运行；不再生成这些已成功预测。

confirmed：completion1639 a在最后FP8预测结束4.60秒后触发归属保护。FP8单臂measurement与59份封存证据、完整quality/complete和1639原始预测已在104文件41.9MB镜像中逐SHA/官方重评分验收；root保持CONFLICT、driver释放。BF16从未开始。此为恢复单臂，不是完整配对通过。

新run只执行BF16原1639，严格同ede2b9e7/ceff binary/模型/输入/Graph/生产调度/C8。pilot80仍引用已独立通过的c；不重复FP8的1639，不重跑80。新完成片必须同时绑定c父批result/audit和a的原terminal/FP8 measurement/recovered audit SHA；原有BF16结果或未知启动痕迹存在则拒绝。最终按原始索引明确三路合并：c双臂80、a已恢复FP8的1639、b新BF16的1639，统一重评分1719。不得将a改成COMPLETE，也不称b单臂为全量配对成功。

CPU故障注入发现退出生命周期的另一边界：已知我方PID退出为Z时cgroup可在/proc移到根。新保护仅在PID/UID/start_ticks/boot与此前本方证明精确相同，且二次stat为Z或已消失时，最多3次新driver读取/1秒退避；必须确认原PID真实消失，其他新PID仍逐个核验。活进程cgroup变化、未知/复用PID、NVML仍报告均拒绝。不信任同UID就算本方，不杀其他进程。旧a未记录当时Z状态，因此其具体触发原因仍unverified；CPU修复不倒推旧冲突为误报。

GPU安全边界和每次120秒稳定空闲、独立watchdog/驱动释放/全生命周期计量保持。初始总12GPU-h自限不重置，当前已结算1.925285GPU-h；新单臂仍给≤2GPU-h整项硬窗口、实际按释放结算。没有GPU可用就继续本地验收和生产trace准备。
