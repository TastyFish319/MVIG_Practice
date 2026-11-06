# ACT_train_new

独立的 ACT 数据与训练工程。保留现有 LeRobot 0.4.4 模型和 v2/v3 输入输出语义，提供中文一键入口。本轮用于内部训练，不进行公开发布。

```text
原始MCAP + 审核记录 → 按episode划分 → LeRobot train/val + trace
                                           ↓
                         ACT训练 → checkpoint验证 → 固定样本评测
                                           ↓
                       离线HTML / 逐关节CSV / statistics.zip
```

## 从哪里开始

1. 按 [环境说明](docs/environment.md) 准备独立 Python 3.12 环境。
2. 将原始任务目录与审核JSON放入 `raw_data/`，或按 [迁移说明](docs/migration.md) 只读引用已有数据集。
3. 从本工程运行：

```bash
bash scripts/preprocess.sh
bash scripts/train.sh
```

训练入口依次选择数据集、1/2/4卡、是否使用速度、超参数和训练名。每项都有默认值；启动前显示验证集与有效配置。默认后台执行，SSH断开后继续，自动完成评测与报告。

完成后打开 `runs/<训练名>/statistics/report.html`。同目录上一级的 `<训练名>_statistics.zip` 是轻量统计包，不包含模型权重、原始视频与TensorBoard事件。

## 数据与模型约定

| 项目 | 约定 |
|---|---|
| 图像 | 左右腕RGB，等比例补边256×256 |
| 基础状态 | follower `[左6关节, 左夹爪, 右6关节, 右夹爪]`，14维 |
| 速度开关 | 默认追加右臂过去100ms速度7维，共21维；可关闭为14维 |
| 输出 | 32步×右臂7轴；20Hz；第一个目标在观测后50ms |
| 六关节 | 未来leader位置 − 当前follower位置；整块使用同一基准 |
| 夹爪 | leader绝对开度；可选软下限只作用推理后处理 |
| 统计 | 只用train；整条episode分集；训练排除不完整尾窗口 |

速度是额外输入，不是预测速度。兼容性以权重内的契约为准，不依赖训练目录名。完整说明见 [流程详解](docs/pipeline_explained.md)。

## 目录

| 目录 | 职责 |
|---|---|
| `acttrain/data/` | 独立MCAP读取、审核、划分、转换、验证、旧集引用 |
| `acttrain/model/` | 相对动作、速度历史、契约、归一化、推理、夹爪过滤 |
| `acttrain/training/` | 官方trainer适配、DDP、双组StepLR、CSV/TensorBoard、后台监督 |
| `acttrain/evaluation/` | checkpoint并行评测、指标、逐关节图、HTML与ZIP |
| `acttrain/cli/` / `scripts/` | 同一套交互与非交互入口 |
| `configs/` | 可编辑默认配置与示例 |
| `raw_data/` / `dataset/` / `runs/` | 用户数据与产物，不纳入代码版本 |
| `tests/` | 最终集中验收用例 |

## 常用操作

```bash
# 环境/数据预检，不会训练
bash scripts/doctor.sh --gpus 2 --dataset dataset/task_act

# 完全非交互；配置文件和命令行使用同一组字段
bash scripts/train.sh --dataset dataset/task_act --config configs/train_defaults.json \
  --name experiment01 --gpus 2 --devices 1,2

# 查看计划，不创建训练/后台目录
bash scripts/train.sh --dataset dataset/task_act --name preview --dry-run

# 关闭历史速度；两组StepLR可以分别开关
bash scripts/train.sh --dataset dataset/task_act --name position_baseline --no-use-velocity \
  --policy-steplr --policy-steplr-step-size 2000 --policy-steplr-gamma 0.5

# 重新评测：每个输出目录只能创建一次
bash scripts/evaluate.sh --dataset dataset/task_act --checkpoints runs/experiment01 \
  --output runs/experiment01/evaluation_repeat --gpus 2 --max-episodes 8
bash scripts/report.sh runs/experiment01 --evaluation runs/experiment01/evaluation_repeat \
  --output runs/experiment01/statistics_repeat

# 查看已写入的TensorBoard日志
 tensorboard --logdir runs/experiment01/tensorboard
```

默认完整保留ACT checkpoint；`--keep-top-k 2` 在训练完成后按val-prior保留最好的2个及final，再评测。它不减少训练过程中的峰值磁盘占用。`best_validation.json` 是验证loss最佳，`evaluation/best_checkpoint.json` 是离线综合评分最佳，两者可能不同。

日志位于 `runs/.jobs/<训练名>/run.log`，状态同时写入任务目录和训练目录。停止监督进程会停止当前训练/评测子进程并记录失败；不提供未经验证的resume入口。

## 文档

- [数据与ACT训练流程](docs/pipeline_explained.md)
- [环境与依赖](docs/environment.md)
- [调参指南](docs/tuning_guide.md)
- [旧数据/权重迁移](docs/migration.md)
- [来源与改动](docs/provenance.md)
- [验收记录与边界](docs/acceptance.md)
- [开发日志](changelog.md)

## References

- [ACT论文与项目](https://tonyzhaozh.github.io/aloha/)：动作块、Transformer与CVAE的训练/推理区别。
- [LeRobot v0.4.4](https://github.com/huggingface/lerobot/tree/v0.4.4)：直接使用ACT模型、数据读写、处理器与官方训练循环。
- [PyTorch StepLR](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)：学习率按完成的优化器更新次数衰减。
