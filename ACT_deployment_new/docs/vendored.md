# 硬件层组件说明

硬件层位于 `act_deploy/hw/follower/`，负责从臂配置、设备端点、worker 进程和运行时状态管理。

| 文件 | 职责 |
|---|---|
| `act_deploy/hw/follower/runtime.py` | worker 看门狗、会话/阶段状态机、帧时效、租约与跟随误差故障 |
| `act_deploy/hw/follower/config.py` | `hardware.yaml` 的严格校验 |
| `act_deploy/hw/follower/endpoints.py` | 从臂及一体化 `i2rt_can` 夹爪端点，以及 `FakeFollowerEndpoint` |
| `act_deploy/hw/follower/worker.py` | worker 进程、RPC 接口、异常处理与生命周期管理 |
| `configs/hardware.yaml` | CAN、端口和看门狗参数；`command_lease_timeout_s` 为 0.30 |
| `configs/tasks.yaml` 的 `start_pose` | 各任务的初始姿态 |

## worker 运行行为

| 项目 | 行为 | 目的 |
|---|---|---|
| portal 异常处理 | `Server(port, errors=False)`，每个方法返回 `{ok, error}` | 被拒绝的请求不会直接终止 worker 进程 |
| 进程生命周期 | 不随控制台退出，忽略 SIGINT | 控制台崩溃时保持通电原地不动，由 `start.sh` 调用救援脚本回零后再断电 |
| PID 文件 | `logs/.workers/<side>.json` | 供救援脚本定位并接管遗留 worker |
| 日志 | 标准 logging | 记录运行状态和故障信息 |
| 控制对象 | 仅 follower 从臂 | 部署推理不控制主臂 |

i2rt SDK 由设备环境提供。当前设备环境 `<SI>/.venv` 通过 `i2rt.pth` 加载 `<SI>/third_party/i2rt`。
