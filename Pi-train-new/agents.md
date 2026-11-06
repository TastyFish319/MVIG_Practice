# agents.md — Pi-train-new 开发守则

> 本文件是本仓库所有后续开发（人或 AI agent）必须遵循的约定。修改约定时先改这里，再改代码。

## 1. 项目目标

基于 **π0.5（pi05）** 对 I2RT YAM 双臂（主从遥操作）采集的数据做微调，提供：

```
raw_data/ ──(一键预处理)──▶ dataset/ ──(一键训练)──▶ runs/<名称>/ ──▶ 评测 ──▶ <名称>_statistics.zip
```

- 部署位置：服务器 `ssh mvig-training` 的 `~/output/Fish/Pi-train-new/`，conda 位于 `~/output/miniconda3`，环境名 `pi_fish`。
- 本机开发路径：`C:\Users\TastyFish\Desktop\SELF_STUDY\MVIG_Practice\Project_MVIG\Tasty_Fish_created\Pi-train-new`。
- 本项目**完全自包含**：不 import 旧 ACT 仓库或 SuperInference 仓库的任何代码（只借鉴思路）。

## 2. 不可违反的数据契约

| 项 | 约定 |
|---|---|
| state（输入） | 只来自 **从臂 follower**；动作臂在前（默认 `[右6关节, 右夹爪, 左6关节, 左夹爪]`） |
| action（标签） | 只来自 **主臂 leader**；默认仅右臂 7 维（预处理可选双臂 14 维） |
| 动作表示 | **相对动作**：关节 `Δ = leader[t+k] − follower_state[t]`，同一 chunk 共用 t 时刻的 state 作基准 |
| 夹爪 | 标签用 leader 夹爪 **绝对值**（不做相对）；推理输出再经 **软下限滤波**（见 `pitrain/gripper.py`） |
| 速度 | 预处理总是计算 follower 速度并存为 `observation.velocity`；训练时选择是否拼进 state |
| 划分 | 按 **整条 episode** 划分训练/验证；归一化统计 **只用训练集** |
| 数据源 | 只使用审核通过（`decision == approved`）的 episode |

## 3. 技术选型（固定，变更需记录到 changelog）

- 训练框架：**LeRobot 0.6.1**（PyTorch 版 pi05，`lerobot/pi05_base` 预训练权重）。
- 多卡：HuggingFace **Accelerate**（torchrun 启动，DDP），卡数仅 1/2/4。
- 曲线：**TensorBoard** + CSV + matplotlib PNG（离线可看）。
- 终端界面：**rich**。
- 能用成熟库就不手写（LeRobot 的数据集/归一化/相对动作处理器、Accelerate、sklearn 指标等）。

## 4. 目录约定

```
pitrain/        Python 包（所有逻辑）
scripts/        一键启动脚本（*.sh 包装 python -m pitrain.cli.*）
configs/        默认配置（评分权重、预处理默认值）
docs/           中文文档（流程详解、调参指南、环境）
raw_data/       原始数据 + 审核记录（不进 git）
dataset/        预处理产物（不进 git）
pretrained/     预训练权重与 tokenizer（不进 git）
runs/           训练输出（不进 git）
tests/          轻量单元测试
```

## 5. 开发流程守则

1. **文档用中文**；代码注释中英皆可，面向"具身智能入门者"讲清楚为什么。
2. 每完成一个板块，在 `changelog.md` 追加记录：**列表/流程图优先，少堆文字**；需要解释概念时用通俗语言、给例子。
3. **测试节奏**：不频繁测试；一个板块开发完成后统一做验收测试，结果写入 changelog。
4. 不覆盖已有产物：数据集/训练目录已存在时拒绝写入，先写 `.partial` 再原子重命名。
5. 所有长任务必须能在 SSH 断开后继续（后台/tmux），并有日志与 `status.json`。
6. 权重体积大（~7GB/个）：只保留 top-k best + final；**不保存优化器状态**（除非显式要求）。
7. 新增依赖必须同步到 `requirements.txt` 与 `docs/environment.md`。
8. 借鉴/复制外部代码或思路，必须写进 `README.md` 末尾 References。
9. 服务器磁盘紧张（共享盘 99% 已用）：任何会产生大文件的功能都要说明体积并给开关。
