# Pi-train-new — 基于 π0.5 的主从遥操作数据微调工程

```
raw_data/ ─▶ [scripts/preprocess.sh] ─▶ dataset/ ─▶ [scripts/train.sh] ─▶ runs/<名称>/ ─▶ <名称>_statistics.zip
```

## 快速开始（服务器 `~/output/Fish/Pi-train-new`）

```bash
# 0. 一次性：环境 + 权重（详见 docs/environment.md）
conda create -n pi_fish python=3.12 -y && conda activate pi_fish
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
bash scripts/download_models.sh

# 1. 原始数据放到 raw_data/<任务>/episode_xxxx，审核记录 *_approved.json / *_review.json 放同一文件夹
# 2. 预处理（交互：选数据 → 选验证集占比 → 确认验证集序号 → 写入）
bash scripts/preprocess.sh
# 3. 训练（交互：选数据集 → 卡数 → 超参 → 名称 → 后台运行 训练+评测+打包）
bash scripts/train.sh
# 4. 下载 runs/<名称>/<名称>_statistics.zip，打开里面的 report.html
```

其它：`bash scripts/evaluate.sh runs/<名称>`（重新评测+打包）；`tensorboard --logdir runs/<名称>/tensorboard`。

## 目录

| 路径 | 内容 |
|---|---|
| `pitrain/` | 全部代码（见下方模块表） |
| `scripts/` | 一键脚本 `preprocess.sh` / `train.sh` / `evaluate.sh` / `download_models.sh` |
| `configs/` | 预处理默认值、评测评分权重 |
| `docs/` | [流程详解](docs/pipeline_explained.md) · [调参指南](docs/tuning_guide.md) · [环境](docs/environment.md) |
| `raw_data/` `dataset/` `pretrained/` `runs/` | 数据与产物（不进 git） |

| 模块 | 作用 |
|---|---|
| `nedf3.py` `review.py` `split.py` | 读 MCAP / 读审核记录 / 分层均匀划分 |
| `timeseries.py` `images.py` `preprocess.py` | 30Hz 因果对齐、速度、静止裁剪、224 letterbox、写 LeRobot 数据集 |
| `gripper.py` | 夹爪软下限滤波 + 从数据估计参数 |
| `data.py` `policy_factory.py` | 组 state（可含速度）、归一化统计、构建 pi0.5 与处理器 |
| `train.py` `validation.py` `schedule.py` `metrics_log.py` | 多卡训练、验证、top-k 保存、曲线 |
| `evaluate.py` `inference.py` `report.py` `pipeline.py` | 多卡评测打分、推理封装、HTML 报告、流水线 |

## 关键设计（一句话版）

- **state = 从臂，action = 主臂**；关节预测**相对量**（主臂未来位置 − 当前从臂位置），夹爪预测**绝对值**。
- **夹爪**：标签用主臂，推理后经**软下限滤波**——模型越想夹紧，输出越平滑地趋近 floor（≈积木宽度−0.08），不会夹太猛。参数从训练集自动估计。
- **速度**：预处理总是计算；训练时选择"含/不含"。默认不含（与官方一致）。
- **权重**：每个 ~7GB，只保留验证误差最好的 top-k（默认 2）+ final，不存优化器。
- **评测**：验证集抽 ≤8 条，所有保留权重逐帧推理，8 项指标加权打分排名。

详细解释（数据如何变换、模型输入输出、架构、优化器、评分）见 **[docs/pipeline_explained.md](docs/pipeline_explained.md)**。

## 已验证 / 未验证

- ✅ 本地：预处理 6 条真实 episode；用"同结构微型网络"在 CPU 跑通 训练→验证→top-k 保存→评测→报告 zip 全流程。
- ⏳ 待服务器验证：真实 pi0.5 权重的显存/速度、多卡 DDP。首次建议先用小 steps 试跑。

## References

- π0.5 论文：Physical Intelligence, *π0.5: a Vision-Language-Action Model with Open-World Generalization*, arXiv:2504.16054.
- π0 论文：Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control*, arXiv:2410.24164.
- OpenPI 官方仓库：https://github.com/Physical-Intelligence/openpi —— 借鉴：相对动作（DeltaActions，夹爪保持绝对）、AdamW(β=0.9/0.95)+cosine 学习率、batch 32 / lr 2.5e-5、letterbox 缩放。
- LeRobot（HuggingFace）0.6.1：https://github.com/huggingface/lerobot —— 直接复用：pi05 模型与权重 `lerobot/pi05_base`、数据集 v3 格式、分位数归一化、`RelativeActionsProcessorStep`、`compute_relative_action_stats`、`EpisodeAwareSampler`、`ImageTransforms`。
- LeRobot pi05 文档（相对动作章节）：https://huggingface.co/docs/lerobot/pi05
- 旧 ACT 训练仓库（内部）：借鉴 MCAP 读取方式、因果时间对齐、按 episode 划分、训练集统计、std 下限思想（本项目改为分位数最小跨度）。
- 因果混淆 / 本体感觉"抄近路"：de Haan et al., *Causal Confusion in Imitation Learning*, NeurIPS 2019（速度版默认关闭的理由）。
- 工具库：Accelerate、TensorBoard、scikit-learn（F1）、Jinja2、rich。
