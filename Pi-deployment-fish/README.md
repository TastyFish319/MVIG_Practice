# Pi-deployment-fish · π0.5 推理与双臂部署

独立工程：复制 ACT 的设备/通信/控制代码，接入 Pi 训练工程的 Pi05Runner。在 Ubuntu 上用 RTX 5090 本地推理，原设备环境继续控制 I2RT YAM follower。

**当前验证范围：离线控制与 mock 通信通过；真实权重 + RTX 5090、RealSense、CAN 和真机动作待 Ubuntu 验收。** mock 不能验证模型是否学会任务。

开发规则：[agents.md](agents.md)；开发记录：[changelog.md](changelog.md)；安装搬运：[环境指南](docs/environment.md)；程序如何驱动机械臂：[流程详解](docs/concepts.md)；验收边界：[validation.md](docs/validation.md)。

## 一次运行

```bash
bash start.sh
# 任务 → 权重 → 模式 → 执行方式 → 夹爪 floor → 启动
```

| 按键 | 发生什么 |
|---|---|
| Enter（待命/零位时） | 平滑到固定初始姿态；不会立即开始抓取 |
| Enter（READY 时） | 真实观测预热通过后开始本轮模型执行 |
| p / Enter（暂停时） | 暂停/继续，丢弃暂停前未返回的推理 |
| f | 人工结束本轮：经初始姿态平滑回零，保持等待；成功/失败由人判断 |
| 空格 | 软件急停，停止采用模型动作并原地保持 |
| z | 结束本轮并回零；故障时先恢复通信/worker，再直接平滑回零 |
| r / q | 回零并确认到位后释放资源、退出 |
| h | 显示帮助 |
| Ctrl+C | 第1次急停；第2次回零释放；第3次退出主进程，worker 保持，使用 --rescue 显式救援 |

模型未提供通用成功信号；不会因为“夹爪闭合”就宣布任务完成。每个任务默认有 80 秒时限，可在任务表调整。正常结束或人工判定失败自动回零；通信/跟随等异常先保持，恢复后再按 z。

## 模式与夹爪选项

| 模式 | 硬件 | 执行阶段 |
|---|---|---|
| sim | 模拟机械臂、合成相机 | 默认真实 Pi 推理；检查链路/耗时，不能评价任务效果 |
| shadow | 真实机械臂、真实相机 | 只显示模型目标；**初始化和回零仍会运动** |
| real | 真实机械臂、真实相机 | 模型目标经平滑和限速后发送给机械臂 |

floor 是夹爪输出的软下限：模型越想夹紧，指令越平滑地靠近 floor，防止持续挤压物体。不同任务不一定需要：

- `checkpoint`：沿用所选权重的 enabled/knee/floor（默认）。
- `on`：启用，保留该权重的 knee/floor 数值。
- `off`：关闭软下限，允许完全闭合；仍把合法开合范围限制在 `[0,1]`。

推理**不使用训练时的随机图像增强**。保留训练一致的 RGB、JPEG q90（可配）、等比缩放、黑边和归一化。

```bash
# 已完成安装、放好权重后；--checkpoint 使用菜单显示的相对名称
bash start.sh --task blue_block --checkpoint blue_run/step_012600 \
  --mode shadow --preset chunk_auto --gripper-floor off

# 不启动机械臂的验收命令
bash start.sh --selftest
bash start.sh --benchmark --checkpoint checkpoints/blue_block/blue_run/step_012600
bash start.sh --check --task blue_block

# 上次控制台崩溃后，检查遗留 worker 并平滑回零释放
bash start.sh --rescue
```

`--backend mock` 只供软件验收，禁止接真实硬件。工程自测自动用临时元数据，不需要把假权重放进正式任务目录。

## 权重与任务表

```text
checkpoints/
  blue_block/blue_run/step_012600/
    config.json
    model.safetensors
    pitrain_contract.json
    policy_preprocessor.json
    policy_preprocessor_step_3_normalizer_processor.safetensors
    policy_postprocessor.json
    policy_postprocessor_step_0_unnormalizer_processor.safetensors
  fold_towel/towel_run/final/...
  open_canister/canister_run/final/...
pretrained/paligemma_tokenizer/...
```

支持把 checkpoint 放在任意正常子目录中，递归扫描最多4层。不要只复制 `model.safetensors`，也不要复制尚在写入的 `.partial` 目录。真实权重实测约 9.35GB/份；模型不需要额外的 pi05_base 权重副本。

`configs/tasks.yaml` 已配置蓝色积木、叠毛巾、开罐、黄色积木四个任务位置；尚未放权重的任务无法运行，不影响其他任务。任务 prompt 取 checkpoint 契约，不能把叠毛巾权重仅改任务名后当成开罐模型使用。

## 改什么、去哪里

| 需求 | 文件 |
|---|---|
| 新增任务 / 权重位置 / 执行时长 | `configs/tasks.yaml` |
| 固定初始姿态 | `tasks.yaml` 的 start_pose（当前全部逐值继承 ACT） |
| 零位 / 相机序列号 / 安全和时延参数 | `configs/deploy.yaml` |
| CAN / worker 端口 / 看门狗 | `configs/hardware.yaml` |
| 两个解释器位置 | `local.env`，参照 `local.env.example` |
| 可选速度窗口 / 相机特征映射 | 任务的 model_overrides：velocity_window_s / camera_map |
| 新模型数据语义 | `pi_deploy/model/`；模型加载和前后处理在 `pi_deploy/inference/` |

没有通用“把所有相对量累加”的逻辑：Pi05Runner 已把每块动作还原为绝对位置。设备端只按 action_names 映射，右臂模型的左臂保持本轮开始位置。

## 日志与来源

`logs/<时间>_<任务>_<模式>/` 包含 session.json（解释器、契约指纹、floor参数等）、server.log、worker日志、events.log 和每轮 npz/json。默认不保存图片；`recording.save_images=true` 会明显增加内存和磁盘占用。npz 中历史字段 `raw` 表示**收到的、已还原并滤波的服务输出**，不是网络内部的归一化增量。

本工程不运行时 import 参考工程；复制清单、原始文件 SHA256 与改动见 [vendored.md](docs/vendored.md) 和 [source-manifest.json](docs/source-manifest.json)。i2rt SDK 继续作为原设备环境中已安装的第三方库使用。

References：本地 `ACT_deployment_new`、`Pi-train-new`；[LeRobot](https://github.com/huggingface/lerobot)；[PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/)。
