# H800 GPU 空闲排队执行（极简说明）

**文件：**`gpu_idle_queue.py`（Python 3.9+，仅标准库）。先把它和你要运行的 `.sh` 脚本复制到 **H800 宿主机**。

**默认规则：**每个任务启动前，要求 **8/8 张卡**显存使用量均 **<1024 MiB**、GPU 利用率均 **≤5%**、无 NVIDIA compute 进程，**连续 120 秒**满足才启动；每 30 秒检查。检查失败就等待，不占 GPU、不杀别人的进程。**只是启动前检查，不是 GPU 预约；无法阻止别人随后抢卡。**

## 1. 写任务清单（JSON 数组顺序就是执行顺序）

在 H800 创建 `gpu_jobs.json`：

```json
{
  "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
  "expected_gpu_count": 8,
  "max_memory_mib": 1024,
  "max_utilization_pct": 5,
  "stable_idle_seconds": 120,
  "poll_seconds": 30,
  "jobs": [
    {
      "name": "14b-quant3",
      "cmd": ["bash", "/export/home/yinjinrun/.borrow/chenjunyu/xllm-h800/scripts/joint14b_quant3.sh"],
      "on_error": "stop"
    },
    {
      "name": "7b-aiperf",
      "cmd": ["bash", "/export/home/yinjinrun/.borrow/chenjunyu/xllm-h800/scripts/aiperf.sh"],
      "on_error": "continue"
    }
  ]
}
```

- 增删 `jobs` 就能控制**脚本数量与顺序**；可以重复添加同一脚本，但每项 `name` 必须不同。
- `on_error: "stop"`：记录失败并**退出监控程序**，修好 `.sh` 后手动 `--resume`，**从失败任务重跑**，不重复成功任务。
- `on_error: "continue"`：记录失败并运行下一项；最后有失败时，整个队列退出码仍为 1。
- 可选任务字段：`"cwd": "/绝对路径"`、`"env": {"KEY": "VALUE"}`、`"timeout_seconds": 3600`（默认 0：不限时）。默认**每个任务前重新等 8 卡空闲**。

## 2. 后台启动（SSH 断开也继续）

```bash
BASE=/export/home/yinjinrun/.borrow/chenjunyu/xllm-h800
RUN="$BASE/experiments/gpu_queue/queue-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$(dirname "$RUN")"
nohup python3 -u "$BASE/gpu_idle_queue.py" \
  --config "$BASE/gpu_jobs.json" --run-dir "$RUN" \
  > "${RUN}.stdout.log" 2>&1 < /dev/null &
echo "PID=$!  RUN=$RUN"
```

**注意：**`--run-dir` 必须是**全新或空目录**；后台 stdout 重定向在目录**外面**，否则启动保护会拒绝运行。保存打印出的 `RUN` 路径，重连后继续使用。

## 3. 看日志、修复后继续

```bash
tail -f "$RUN/events.log"                        # 等待/开始/成功/失败/停止
python3 "$BASE/gpu_idle_queue.py" --run-dir "$RUN" --status
ls "$RUN/jobs/"                                  # 每个脚本每次尝试独立日志
```

任务失败且设置了 `stop`：修好原 `.sh` 文件（**保持 JSON 中的脚本路径、任务顺序不变**），执行：

```bash
nohup python3 -u "$BASE/gpu_idle_queue.py" \
  --config "$BASE/gpu_jobs.json" --run-dir "$RUN" --resume \
  > "${RUN}.resume.log" 2>&1 < /dev/null &
```

若想**放弃失败任务**继续后续任务：上述恢复命令再加 `--skip-failed`。若程序被 `kill -9` 等意外中断，留下 `running` 状态，会拒绝直接恢复；**先核实旧脚本/Docker 容器已停止**，再加 `--retry-uncertain`。

## 两条必须遵守的实验约定

1. **`.sh` 必须返回真实退出码**：例如 `docker run ... 2>&1 | tee "$OUT/console.log"` 后**紧接** `RC=${PIPESTATUS[0]}; echo "EXIT=$RC"; exit "$RC"`，否则 `echo`/`tee` 可能掩盖实验失败。GPU 被抢时也要 `exit` 非零，不能只 `echo STOP` 后正常退出。
2. **任务必须前台运行并自行做好 GPU 隔离**：Docker 保留已验证的 `--gpus device=1`、UUID 检查、seccomp、`PYTHONPATH`、TVM FFI `LD_LIBRARY_PATH` 等参数；不要用 `docker -d`、`nohup` 或 `&` 让实验在脚本退出后继续运行。监控器只管理宿主机任务进程组，不能保证 Docker daemon 的残留容器被自动清理；中断/崩溃后恢复前自行检查。

**安全边界：**同一 `RUN` 不允许两个监控程序同时执行；不同 `RUN` 或其他用户之间**没有跨用户互斥锁**。监控器不主动停止运行中的其他 GPU 任务，也不会在实验开始后抢占或强制终止后来占卡的用户。
