# Robot Learning Toolkit

面向 **I2RT YAM 主从遥操作数据**的具身智能工具集，涵盖演示数据审核、ACT / π0.5 模型训练、离线评测及机械臂部署。

仓库包含五个可分别使用的模块。源码与配置示例纳入版本管理，原始数据、模型权重、训练统计及本机环境不随仓库提供。

## 功能概览

| 模块 | 主要功能 | 使用说明 |
|---|---|---|
| **data_review** | MCAP 技术检查、双腕视频回放、逐 episode 人工审核、自动保存、审核 JSON 导入导出 | [审核工具](data_review/README.md) |
| **ACT_train_new** | 已审核数据预处理、ACT 训练、14/21 维状态选择、1/2/4 GPU 启动、checkpoint 验证与离线评测 | [ACT 训练](ACT_train_new/README.md) |
| **Pi-train-new** | π0.5 微调、可选速度输入、多 GPU 训练、checkpoint 选择、评测与统计报告 | [π0.5 训练](Pi-train-new/README.md) |
| **ACT_deployment_new** | ACT 权重加载、观测构造、动作调度与平滑、模拟/影子/真机模式 | [ACT 部署](ACT_deployment_new/README.md) |
| **Pi-deployment-fish** | π0.5 本地推理、夹爪过滤选项、时延处理、双臂控制与部署检查 | [π0.5 部署](Pi-deployment-fish/README.md) |

训练模块能够保存 CSV、训练曲线及可离线浏览的 HTML，并打包轻量统计 ZIP。部署模块提供设备检查与故障处理工具；实际行为以各模块文档和配置为准。

## 目录结构

```text
.
├── data_review/               数据审核工具
│   ├── backend/               MCAP 读取、检查、任务管理与本机 HTTP API
│   ├── frontend/              审核页面、样式与交互
│   ├── scripts/               启动辅助脚本
│   └── review.py              启动入口
├── ACT_train_new/             ACT 数据与训练流程
│   ├── acttrain/              data / model / training / evaluation / cli
│   ├── configs/               预处理、训练与评分配置
│   ├── scripts/               一键入口
│   └── docs/                  环境、流程、调参与验收说明
├── Pi-train-new/              π0.5 数据与训练流程
│   ├── pitrain/               预处理、训练、推理与评测
│   ├── configs/               默认参数与评分配置
│   ├── scripts/               预处理、训练与评测入口
│   └── docs/                  环境、流程与调参说明
├── ACT_deployment_new/        ACT 推理与部署
│   ├── act_deploy/            模型、控制、硬件、应用与工具
│   └── configs/               任务、硬件与控制参数
├── Pi-deployment-fish/        π0.5 推理与部署
│   ├── pi_deploy/             模型服务、控制、硬件与应用
│   └── configs/               任务、硬件与推理配置
└── README.md                  功能介绍、项目结构与使用入口
```

`Pi-statistics/` 为本机实验统计目录，已被 Git 忽略，不属于本仓库交付内容。各模块的 `raw_data/`、`dataset/`、`checkpoints/`、`pretrained/`、`runs/` 等运行目录按需在使用时创建或填充。

## 典型工作流

```text
原始 NEDF3 / MCAP 演示数据
          ↓
     data_review
   人工审核 + 技术检查
          ↓
 *_approved.json + *_review.json
          ↓
 ACT_train_new 或 Pi-train-new
  预处理 → 训练 → 验证 → 离线评测
          ↓
  完整 checkpoint + 输入输出契约
          ↓
ACT_deployment_new 或 Pi-deployment-fish
  环境检查 → 模拟/影子检查 → 受控真机验证
```

- 数据按完整 episode 划分训练集与验证集，归一化统计使用训练集。
- 本项目的训练约定以 follower 状态作为观测，leader 未来动作作为监督目标；关节采用相对量，夹爪保留绝对开度。
- ACT 与 π0.5 的频率、状态顺序、图像处理和权重格式有各自约定，不能直接互换。迁移或部署时以 checkpoint 内契约及对应模块说明为准。

## 快速开始

### 先体验数据审核

使用 Python 3.10 或以上，进入审核模块安装依赖，然后指定自己的原始数据根目录：

```bash
cd data_review
python -m pip install -r requirements.txt
python review.py --data-root "/path/to/raw_data" --save-config --choose
```

Windows 也可以使用 `prepare_task.cmd` 选择并生成任务，或使用 `start_review.cmd` 打开已有任务。页面默认只监听 `127.0.0.1:8776`。首次使用没有预置任务，需要自行提供受支持的 NEDF3 v2 数据。

### 训练 ACT 或 π0.5

先按对应模块的环境文档安装依赖并准备审核过的数据，再运行：

```bash
# 在 ACT_train_new/ 或 Pi-train-new/ 目录内
bash scripts/preprocess.sh
bash scripts/train.sh
```

- [ACT 环境配置](ACT_train_new/docs/environment.md) · [ACT 调参指南](ACT_train_new/docs/tuning_guide.md)
- [π0.5 环境配置](Pi-train-new/docs/environment.md) · [π0.5 调参指南](Pi-train-new/docs/tuning_guide.md)

### 部署模型

部署需要完整权重目录、模型所需资源及本机硬件配置。先阅读部署模块 README，配置设备与模型解释器路径，再执行环境自检：

```bash
# 在 ACT_deployment_new/ 或 Pi-deployment-fish/ 目录内
bash start.sh --check
```

任务名称、相机序列号、CAN 映射、初始姿态及运动限制都是需要核对的示例配置。影子模式的初始化与回零仍可能产生真实运动，首次使用应从模拟与受控检查开始。

## 环境要求

各模块分别维护依赖文件，**不要把五个模块的依赖合并安装到同一个环境**。

| 部分 | 基础环境 |
|---|---|
| 数据审核 | Python ≥3.10；NumPy、MCAP、protobuf、OpenCV、FFmpeg；无需 GPU 或设备 SDK |
| ACT 训练 | Python 3.12、LeRobot 0.4.4；GPU 训练以 Linux / CUDA 为目标 |
| π0.5 训练 | Python 3.12、LeRobot 0.6.1；需要对应预训练资源和 CUDA 环境 |
| ACT / π0.5 部署 | 设备进程与模型进程使用各自环境；设备侧需要相应 i2rt、CAN、RealSense 等支持 |

部分模块文档保留了开发机器上的路径和环境记录，使用时应替换为自己的配置。训练统计、验证误差和模拟检查不能代表闭环任务成功率。

## 验证范围

- ACT 训练工程已记录真实数据预处理、单卡 state21 与双卡 state14 的短流程验收；真实 4 卡运行与长训练效果不由此保证，见 [ACT 验收记录](ACT_train_new/docs/acceptance.md)。
- π0.5 训练的验证情况见其 [README](Pi-train-new/README.md)；部署模块的离线、GPU 与真机边界分别见 [ACT 开发记录](ACT_deployment_new/changelog.md) 和 [π0.5 部署验收](Pi-deployment-fish/docs/validation.md)。
- 审核工具的检查、回放与记录保护情况见 [审核说明](data_review/README.md)。

## 许可证

当前仓库用于成果展示，暂未指定开源许可证。第三方模型、库、SDK及相关资源分别遵循各自的使用条款；本仓库不随附这些资源的再分发授权。

## 参考

- [ACT 项目](https://tonyzhaozh.github.io/aloha/)
- [LeRobot](https://github.com/huggingface/lerobot)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
