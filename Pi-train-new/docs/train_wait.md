# 等待 GPU 后训练

```bash
cd ~/output/Fish/Pi-train-new
bash scripts/train_wait.sh
```

数据集、卡数、超参数、名称、确认、前后台选项完全复用 `train.sh`。确认后先入队，等待足够的空闲 GPU，再执行原有「训练 → 评测 → 报告打包」流水线。默认后台运行，SSH 断开后继续等待和训练。

```bash
# 所有原训练参数均可用
bash scripts/train_wait.sh --dataset <数据集名> --gpus 2 --name my_run --yes

# 调整空闲标准与检查间隔
bash scripts/train_wait.sh --idle-memory-mb 32768 --min-free-memory-mb 61440 --idle-utilization 10 --poll-seconds 10

# 查看队列，无需选择数据集
bash scripts/train_wait.sh --queue-status

# 把新阈值应用到已经等待的任务，不改训练配置、顺序或正在运行的任务
bash scripts/train_wait.sh --update-waiting
```

- **可用标准**：`nvidia-smi` 报告已用显存 **≤ 32768 MiB（32 GiB）**、实际剩余显存 **≥ 61440 MiB（60 GiB）**，同时 GPU 利用率 **≤ 10%**。三个条件都满足才分配；未知指标/采集失败都不当作可用，采集失败继续重试。
- **响应时间**：默认每 10 秒检查一次；下一次检查发现足够 GPU 时立即启动。
- **排队顺序**：在所有参数选完、配置保存后，以全局文件锁分配的序号为准，严格先提交先分配。队首要 4 卡但只有 2 卡空闲时，后面的 1/2 卡任务继续等，避免大任务长期被插队。
- **并行与占卡**：分配精确数量的 GPU UUID，通过 `CUDA_VISIBLE_DEVICES` 传给原流水线；每卡加锁到整个流水线（包括评测）退出。有剩余空闲卡时，后续任务可以并行启动。原环境设置的 `CUDA_VISIBLE_DEVICES` 也会限制可分配范围。
- **队列范围**：本服务器、当前项目下用 `train_wait.sh` 提交的任务共享 `runs/.train_wait_queue/<主机名>/`。锁只协调这些任务；外部任务不遵守此队列，仍可能在检测后抢卡。
- **进程退出**：正常退出/失败/取消会清理队列并释放锁。监控被强制结束的遗留队列记录会自动清理；若流水线还活着，它继承的 GPU 锁继续生效。服务器重启后需要重新提交任务。
- **原入口**：`train.sh` 原来的直接启动行为保持不变；`train_wait.sh --dry-run` 也只生成配置，不入队。
- **更新等待阈值**：`--update-waiting` 只更新 `queued` 条目。新版监控下一轮自动读取阈值；2026-11-02 的旧监控需在全局锁下交接到新版，保留任务 ID、提交时间、FIFO 顺序、训练配置、原命令和环境，仅监控 PID 更新（会在终端打印）。后台监控可自动交接，旧版前台监控会拒绝自动交接；不影响正在训练的任务。普通 TERM/HUP/中断或交接写入失败会回滚并恢复旧监控；不要强杀更新命令。
- **依赖与空间**：仅使用现有依赖与 Python 标准库；新增少量 JSON 状态/锁文件及文本日志，不新增权重或复制数据集。

## 日志、状态与取消

```bash
tail -f runs/my_run/logs/pipeline.log
cat runs/my_run/status.json         # 等待时 queued，启动后沿用原流水线状态
cat runs/my_run/queue_status.json   # 队列位置、PID、GPU 分配与最终退出码

# 同时取消监控和它启动的训练/评测子进程；PID 也会在提交后打印
kill -- -$(cat runs/my_run/logs/pipeline.pid)
```

`--foreground` 可在前台等待并显示输出；Ctrl+C 会取消这次任务。默认后台模式的等待与训练输出都写入 `pipeline.log`。

## 两卡训练的显存估算（2026-11-03）

参考本服务器已有训练记录，配置为两卡 DDP、每卡 batch 16、BF16、梯度检查点、两路 224 图像、chunk 50、全参数微调：

| 数据 | 每卡显存 |
|---|---:|
| `blue_block_2gpu_run1_18500steps`，925 条训练日志的 PyTorch 分配峰值 | 49.2 GB（十进制，约 45.8 GiB） |
| 当前 `yellow_block_2gpu_run1_9500steps` 的 `nvidia-smi` 占用 | 约 49,800 MiB（48.6 GiB，含缓存/运行时） |
| 对相近排队配置的估算，包含额外余量 | 约 50–55 GiB |
| 本次调度要求的实际剩余空间 | 至少 60 GiB |
| GPU 0：已用 29,617 MiB、实际剩余 67,668 MiB、利用率 0% | 剩余约 66.1 GiB，满足新标准 |

后两个训练需求数字是基于当前实测的估算和调度预算，并非对所有 pi0.5 配置的保证；包含速度输入的任务未另行实测。DDP 在每卡保留模型副本，不能把单卡显存简单除以二（[PyTorch DDP](https://docs.pytorch.org/docs/stable/notes/ddp.html)）。图像数、batch、冻结设置及梯度检查点都会影响需求（[LeRobot pi05 参数说明](https://huggingface.co/docs/lerobot/pi05)）。本次不修改这些训练参数，也不额外运行显存探测训练。

## 实现与验收范围

1. `train_wait.sh` 照搬 `train.sh` 的环境初始化，只切换到等待入口；原 CLI 增加可选启动回调，避免复制参数选择逻辑后产生分歧。
2. 独立队列模块负责 FIFO、GPU 采集、加锁、后台进程、状态与取消；原训练/评测/报告模块保持原样。
3. 全部代码完成后统一在服务器验收：临时目录模拟 GPU 采集和流水线，使用真实后台进程与文件锁，覆盖双阈值、FIFO、多任务并发、不重复分配、失败恢复与取消，不启动真实训练。
