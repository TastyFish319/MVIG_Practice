# 复制来源与改动

源码实际复制进本工程，没有运行时 import `ACT_deployment_new` / `Pi-train-new`。复制时原文件的 SHA256 见 `source-manifest.json`，源工程未修改。

| 本工程 | 来源（相对于 Tasty_Fish_created） | 改动 |
|---|---|---|
| `pi_deploy/hw/follower/runtime.py`、`config.py` | ACT_deployment_new/act_deploy/hw/follower/ | 复制；底层状态机/看门狗语义保持 |
| `hw/follower/endpoints.py` | ACT对应文件 | 复制，只通过环境中i2rt控制SDK |
| `hw/follower/worker.py`、`hw/arms.py` | ACT对应文件 | 包名；worker归属RPC/实例/PID校验及pidfd；逐侧救援；启动失败不直接释放健康worker |
| `hw/cameras.py`、`preflight.py` | ACT对应文件 | 复制；摄像头数据流和独立采集线程 |
| `control/trajectory.py`、`safety.py`、`scheduler.py` | ACT对应文件 | 复制；调度实际FPS由Pi契约提供，注释更新 |
| `control/history.py` | ACT对应文件 | 增加Pi训练使用的因果最小二乘速度；保留历史存储接口 |
| `ipc.py` | ACT对应文件 | `PI05`协议标识、请求编号回显、数组类型/大小校验 |
| `app/console.py`、`recorder.py` | ACT对应文件 | Pi界面、floor状态；记录写盘由控制器移至后台线程 |
| `app/engine.py` | ACT对应文件 | 预热/时延门禁、首块期限、在控制线程接收结果、代号丢弃旧结果、实际到位后释放 |
| `app/main.py`、`registry.py`、`config.py` | ACT对应文件 | Pi完整checkpoint扫描、floor/参数传递、严格配置、模型先加载再接硬件 |
| `tools/rescue.py`、`check_env.py`、`start.sh` | ACT对应文件 | Pi环境和入口；救援核对到位后释放；GPU/cu128检查 |
| `inference/inference.py` | Pi-train-new/pitrain/inference.py | Pi05Runner原封装复制，维持相对→绝对与夹爪滤波流程 |
| `inference/gripper.py` | Pi-train-new/pitrain/gripper.py | 复制，保持滤波数值定义；新增独立options模块覆盖enabled |
| `inference/policy_factory.py` | Pi-train-new/pitrain/policy_factory.py | 复制4个推理函数，移除训练构建函数；read_json转本地；增加严格加载回执，拒绝上游吞掉错误后的随机权重回退 |
| `inference/data.py` | Pi-train-new/pitrain/data.py 的 compose_batch | 复制所需常量与函数；不引入训练数据集模块 |
| `model/features.py` 图像部分 | ACT letterbox_rgb 与 Pi训练images.py | 保留RGB/JPEG/INTER_AREA/黑边流程；尺寸从Pi config取，关闭随机增强 |
| `model/spec.py`、`inference/options.py`、工具和测试 | 新增 | 按Pi契约实现并集中验收 |
| `model/stats.py` | ACT model/spec.py 的统计读取函数 | 复制；只读小型统计文件并加大小上限，使用输入state范围映射绝对示范边界 |
| `configs/hardware.yaml`、`tasks.yaml`固定姿态、`deploy.yaml`运动参数 | ACT configs | 保留CAN/相机/零位/初始姿态；Pi动作帧率和推理配置另行适配 |

ACT中的worker运行时本身来源于 SuperInference 的 i2rt teleop 实现，原始来源链见 ACT 的 docs/vendored.md。i2rt SDK 不复制，是已有设备环境中的第三方依赖。

## 2026-11-04 · 按任务读取 episode 0 初始姿态

按用户要求，`configs/tasks.yaml` 中 `fold_towel`、`open_canister`、`yellow_block` 的初始姿态改为 `mvig-training` 原始 episode_0000 的首帧 follower 实际状态；`blue_block` 保留原 ACT 数值。

服务器来源根目录：`/home/users/zhaoruiqi_yizu/output/Fish/Pi-train-new/raw_data`。

| 本地任务 | 原始 episode 目录 | 首帧 log_time（ns） |
|---|---|---|
| fold_towel | fold_towel/episode_0000 | 1793526041751812125 |
| open_canister | open_cup/episode_0000 | 1793529016941796398 |
| yellow_block | task1/episode_0000 | 1793196658209091141 |

读取 MCAP 中 `nmx/hal/i2rt/follower/{left,right}/joint` 与 `hand_state` 的第一条消息，每个任务四个 topic 首条时间戳一致。按关节名称 joint0..joint5 与 gripper 提取，排列为左6关节、左夹爪、右6关节、右夹爪；关节单位 rad，夹爪 0..1。不使用 leader 动作或图像，不使用数据集裁剪后的首帧（fold_towel 的 trim 起点为10，yellow_block 为56）。仅只读访问训练服务器，没有修改服务器文件。
