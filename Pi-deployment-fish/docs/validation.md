# 验收记录与待验证边界

日期：2026-11-03；本机 Windows，Python3.12，现有 pi_fish_dev 环境。没有连接 mvig-ubuntu，没有下载9.35GB真实权重。

| 验收 | 结果 | 能证明什么 |
|---|---|---|
| `python -m pi_deploy.tools.selftest` | 最终37项通过 | 契约/映射、真实服务器JSON元数据、padding、图像、floor、示范范围、控制异常、worker看门狗、mock TCP、救援与生命周期 |
| `python -m tests.optional_runner_smoke` | 通过 | CPU torch + 实际 LeRobot 相对/绝对动作处理器 + 本工程 Runner；两种速度配置、三种floor选项；微型本地张量严格加载与上游吞错拒绝 |
| `python -m tests.acceptance_cli --bash <Git Bash>` | 通过 | 实际 start.sh、模型子进程、完整ACT固定初始姿态、暂停恢复、结束回零释放；floor off写入session |
| `bash -n start.sh` | 通过 | Bash语法 |
| `ast.parse(feature_version=(3,10))` | 41个包内Python文件通过 | 设备端Python3.10语法兼容；不代替真实SDK运行 |
| 源文件SHA256与运行时import审计 | 42条来源匹配 | 两个参考工程的源文件未改变；运行时不import其包 |

CLI验收最初在本机8890端口遇到WinError10013；直接socket探测确认该端口范围被Windows拒绝绑定，28890可用；只调整测试用配置端口后通过。Ubuntu部署默认8890，不借此宣称已核对Ubuntu端口。

离线测试生成的checkpoint仅是**临时元数据夹具**，没有真实神经网络权重；不放进正式checkpoints目录。Runner处理器验收使用固定假网络输出，不证明大模型加载或任务成功。

最终日志：`logs/final_selftest.log`、`logs/final_runner_smoke.log`、`logs/cli_acceptance.log`。实际服务器JSON元数据保存在 `docs/reference-contracts.json`，包含采集时间和来源路径，无权重；用于核对三种任务的真实名称/维度/处理器约定。

## 独立审阅后的集中修正

- 移除任意姿态直接释放入口；UI异常只保持，不自行回零断电；故障清除旧释放意图。
- 同时核对config与处理器显式动作名称，防止同维但错臂的契约混配。
- 相机新鲜度在控制循环持续检查；计划耗尽锁存实测保持；首块接纳也检查独立截止时刻。
- 救援逐侧核对归属，操作者确认后重新采样；释放使用已验证身份及pidfd，不重新信任PID文件。
- sim/mock路径不访问真实worker；模型服务握手核对PID、随机实例号、后端和floor参数。
- 自动执行步数在菜单展示前解析；不完整权重不列为候选。无暂缓的minor审阅项。

上述异常路径已加入集中回归检查。审阅未判断真实GPU、实物映射与碰撞间隙，因为它们需要目标硬件；保留下方上机验收边界。

## 尚须在 Ubuntu 进行

- `pip check`、驱动/5090/bf16实际运算、CPU内存及显存余量。
- `start.sh --benchmark --checkpoint <完整真实目录>`：严格权重加载、真实网络输出、首轮与稳态时延、显存。
- `start.sh --check`：实际CAN与相机序列号/端口。
- shadow：真实相机+关节+本机TCP的全链路时延；初始姿态/回零/急停/资源释放。
- real：现场确认空间与物理急停后，小范围任务执行。无碰撞规划或自动任务成功检测。

以上未验收项不能用mock、CPU处理器测试或训练机H20上的记录代替。等待用户明确告知Ubuntu可连接后再开展。
