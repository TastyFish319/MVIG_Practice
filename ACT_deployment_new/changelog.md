# changelog

## 2026-10-31 · v0.1.2 新增任务 yellow_block_into_bowl（v2 模型）暴露的问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `spec_from_checkpoints` FAIL | 测试写死了"所有权重都是 v3（state21）"；解析本身正确（v2: state14、右臂增量+夹爪绝对） | 改为按每个 checkpoint 自己的文件校验一致性（维度 vs 归一化统计、有无速度 vs 观测契约、增量 vs 动作契约），并按类型汇总 |
| （顺带发现）v2 模型没有示范范围 → 安全"范围裁剪"对 v2 失效 | 范围只从 v3 观测契约读 | 从 checkpoint 自带的归一化文件（safetensors，纯 numpy 读取）读 `observation.state.min/max`；同时交叉检查维度 |
| 速度/动作解码测试用"排序后第一个权重" | 若第一个是 v2 就会错 | 显式挑选带速度 / 带增量的权重 |
| 本机验证 | v1(绝对) / v2(增量) / v3(增量+速度) 三种权重混放 | ✅ 9/9 |

## 2026-10-31 · v0.1.1 部署机首次 selftest 反馈

| 现象 | 原因 | 处理 |
|---|---|---|
| `scheduler_smooth` FAIL（0.0335 > 0.03） | **测试本身的问题**：每个假计划加了未固定种子的随机偏移，阈值 0.03 是拍脑袋的；本机碰巧 0.0286 通过 | 固定随机种子；起步淡入与稳态分开判断；稳态上限按"轨迹速度 + 偏移跳变/淡化步数"推导；30 个种子全部通过 |
| （顺带发现）时间集成模式下新计划"突然"加入平均 | 只有前几个计划时会有小跳变 | 每个计划的权重也在 `blend_s` 内淡入 |
| `[arm_left] Connection to server lost…` 刷屏 | 无害：测试结束释放 fake worker 后 portal 客户端自动重连 | 释放时先关闭 RPC 客户端再停 worker |
| `worker_runtime_fake_arm` | 部署机设备环境 PASS：worker 进程 + RPC + 租约超时故障 + 复位 ✅ | — |

## 2026-10-31 · v0.1 从零搭建（本机离线完成，真机待验证）

### 0. 调研结论 → 设计决策

| 调研发现 | 决策 |
|---|---|
| 权重 `v3_*`：state21 = 14 + 右臂速度 7（100ms）；输出 32×7，右关节增量 + 夹爪绝对；20Hz，第 0 步 = +50ms | 全部**从 checkpoint 契约文件自动解析**（`model/spec.py`），不写死 |
| 训练图像：RGB→JPEG q90→等比缩放→黑边 256² | 部署端逐像素复现（selftest 验证） |
| 两个现成环境：设备 Py3.10（i2rt/portal/相机，无 torch）、模型 Py3.12（torch+lerobot） | 两进程：控制台在设备环境，模型服务在模型环境，本机 TCP 通信 |
| worker 需要独立处理帧时效、指令租约和跟随误差 | 在工程内提供看门狗运行时，统一管理会话与故障状态 |
| portal 默认 `errors=True`：一个被拒请求就会让 worker 退出 → 手臂掉落 | worker 使用 `errors=False` + 方法包装 |
| 推理通信需要降低连接开销，20Hz 动作点需要平滑执行 | 长连接二进制协议（开销约 0.6ms）；采用 100Hz 插值、分块提交与交叉淡化 |
| 回零速度与到位等待需要兼顾效率和异常处理 | 默认 0.4 rad/s，五次多项式；settle 超时只警告 |

### 1. 开发顺序

```mermaid
flowchart LR
    A["AGENTS.md"] --> B["模型层<br/>spec / features / ipc / server"]
    B --> C["控制层<br/>history / scheduler / safety / trajectory"]
    C --> D["硬件层<br/>worker / arms / cameras / preflight"]
    D --> E["应用层<br/>engine / console / main / recorder"]
    E --> F["工具<br/>check_env / selftest / rescue / start.sh"]
    F --> G["统一验收"]
```

### 2. 统一验收（本机 Windows，Py3.12 + CPU torch）

| 测试 | 结果 |
|---|---|
| `selftest` 9 项（契约解析、速度/补零、图像逐像素、动作还原、调度平滑、轨迹限值、安全过滤、协议） | ✅ 9/9（worker+portal 项在部署机运行） |
| sim 模式 + **真实权重 v3_long2_017000（CPU）** 全流程：初始化→执行→f 回零→再执行→急停→紧急回零→释放 | ✅ |
| 记录分析：每 ~0.42s 重规划（0.5s - 延迟提前量），指令每 tick 最大 0.014 rad | ✅ 平滑 |
| shadow 逻辑（`--sim-hardware`）：模型目标显示、手臂不动；暂停/继续 | ✅ |
| 故障路径：推理连续失败→急停→紧急回零；超时→回零；worker FAULT→z 复位→回零 | ✅ `tests/engine_failure_paths.py` |
| 发现并修复 | 脚本结束后空键盘对象报错；实时预热在相机未就绪时放弃 → 改为重试 5 次 |

### 3. 待真机验证（下次连上 mvig-ubuntu）

1. `bash start.sh --check`：两个解释器、CUDA、CAN、相机序列号
2. `bash start.sh --selftest`：含 fake worker + portal + 租约故障
3. **影子模式**：初始化/回零是否平滑、到位误差、"模型想要"是否合理、GPU 往返延迟
4. 真机：先 `chunk_10`，再对比 `chunk_4` / `ensemble`
