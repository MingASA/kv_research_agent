# BF16 FA3-prefill/XQA-decode：算子通过，随后归属保护停止

confirmed：source1c2be159/H800 GPU4，真实单层GQA40/8、非连续Q、partial/replan、reserved page0+padded Graph、有限尾页与恢复参考检查通过。BF16 prefill relativeL2=.00228243，decode=.00390603，changed/padded/restored=.00287112；同误差门槛未降低。这是operator证据，不是14B质量或服务通过。

root为CONFLICT，无arm seal，无AIPerf测量。01:36:00.565798Z，驱动仍报告PID2594157，先前证明属于本容器adc435...；新读取的PID/UID1005/start366882282/boot均不变，但cgroup已回到根、stat stateR，触发原安全边界。只回收本方容器，driver确认释放；模型加载日志在停止过程中仍出现，不把开始加载或算子输出当服务成功。实际占用见state和status原始interval，完整失败保留，不重启旧ID。

hypothesis：Linux退出时可以先设置PF_EXITING、拆除cgroup，而stat仍R。原事件没有flags，不能回溯确认为误报。S0证实host为5.10.0-1.oe.jd_619.x86_64、旧PID已消失；本地24个自有CPU子进程退出均实际观察到R+PF_EXITING，证明这种生命周期可能，不证明旧H800事件flag。官方5.10 do_exit先exit_signals(PF_EXITING)，后cgroup_exit/exit_notify；sched.h定义0x4，proc stat第9字段输出task flags。来源：[exit.c](https://github.com/torvalds/linux/blob/v5.10/kernel/exit.c)、[sched.h](https://github.com/torvalds/linux/blob/v5.10/include/linux/sched.h)、[array.c](https://github.com/torvalds/linux/blob/v5.10/fs/proc/array.c)。

最小修复只增加内核exit flag观测：必须是此前精确证明属于本容器、同PID/UID/start/boot，并有Z/消失或PF_EXITING正证据，才允许最多3次driver只读复查/1秒等待；必须真实从driver消失再逐个验证剩余PID。普通R/S cgroup变化、未知/复用PID、无exitflag、仍被driver报告或新foreign均拒绝，不向未知进程发信号，不绕过120秒空闲。新run须重新完整准入，CPU不能倒推旧失败成功。

证据：bf16_xqa_reference_status_a3.json、bf16_xqa_conflict_host_lifecycle_readonly_a1.json、local_CPU_running_state_PF_EXITING_observations_v1.json（均在reports/operations/20261026T075846Z）。完整失败41文件5911811bytes已只读镜像并逐SHA核验，operator输出/终态/无AIPerf measurement再次确认；这是失败raw完整性，不是服务成功。证据bf16_xqa_failure_a_raw_integrity_v1.json。
