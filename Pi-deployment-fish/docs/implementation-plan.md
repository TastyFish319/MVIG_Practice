# Pi 部署 Implementation Plan

> For agentic workers: 使用 executing-plans，在本会话连续完成；常规细节自行决定，最后一次整体代码审阅。

**Goal:** 独立工程复用 ACT 控制链路与 Pi 推理封装，交付交互启动、环境部署说明、离线验收及 GPU 验收工具。
**Architecture:** 设备 Python 3.10 + 模型 Python 3.12，以本机 TCP 传 numpy；双臂 worker 保持硬件控制。Pi 输出在服务端还原为绝对值，设备端只映射与调度。
**Tech Stack:** numpy、OpenCV、PyYAML、portal/i2rt/RealSense；模型端 torch 2.8 cu128、LeRobot 0.6.1。
**Spec:** `docs/design.md`；项目约定 `agents.md`。

## Global Constraints

- 在用户指定 Pi-deployment-fish 目录开发；不创建/修改参考工程、git 分支或提交，不连接 mvig-ubuntu。
- 复制源码，禁止运行时 import 两个参考仓库。普通依赖使用中科大源。
- 输入输出以真实契约为准；推理不随机增强；夹爪 floor 可切换并记录。
- 按板块集中测试。真实 GPU 与真机结果必须留待实际验收，不能以 mock 代替。

## Review Focus

- 暂停、恢复、急停、回零时在途推理不得重新驱动机械臂。
- 双臂顺序与相对动作处理不能重复或错位；关节速度历史不能混用 ACT 定义。
- 模型加载失败、预热失败、首块超时、过期计划不能进入执行。
- 回零不到位或状态失效时不释放；救援保持可用且不能接管别的工程 worker。
- CLI 与菜单的 floor 选择必须真正进入 Runner，mock 不得驱动真机。

## Task 1 · 模型适配和独立基础

Files: 复制 ACT `pi_deploy/{hw,control}`、`ipc.py`、`config.py`、`registry.py`；新增 `model/{spec,features}.py`、`inference/`、`server/policy_server.py`、configs、tests/test_model.py。
Interfaces: `resolve_spec(path, overrides) -> ModelSpec`；`build_state(...) -> (state, robot14, info)`；`decode_actions(...) -> [H,14]`；`Pi05Runner.predict(batch) -> raw/filtered`；`PolicyClient.predict(state,images)`。
- [x] 先写契约/映射/速度/padding/图像/滤波测试，集中运行看失败。
- [x] 复制已验证基础，按 Pi 契约实现适配，推理依赖剥离，完整 checkpoint 校验。
- [x] 集中验收模块；记录结果。

## Task 2 · 控制、时延和交互

Files: 复制并适配 `app/`、`tools/rescue.py`、`start.sh`；新增 tests/test_engine.py。
Interfaces: 继承 ACT `EngineOptions`、`Engine.command`、`Engine.snapshot`；新配置 `inference` 与 `policy` 延时门槛；floor 在启动阶段传到模型进程。
- [x] 写控制状态机、预热门禁、故障释放、旧结果/超时测试，集中运行看失败。
- [x] 模型先准备，再接硬件；预热输出丢弃；控制线程接收推理结果；代号淘汰旧结果；释放严格到位。
- [x] 完成任务菜单、权重发现、三模式、floor 菜单和面板、救援。
- [x] 集中运行模拟生命周期与失败路径验收；记录结果。

## Task 3 · 工具、文档和整体审阅

Files: tools/{check_env,selftest,benchmark}.py；requirements.txt、local.env.example、README、docs/{environment,concepts,vendored,validation}.md。
- [x] 完成独立无权重自测、真实 GPU 基准工具、只读环境检查，文档写出完整复制/安装/验收命令。
- [x] 汇总运行全套离线测试、mock TCP/启动器演练、Python 3.10 语法解析和 shell 语法检查（工具可用时）。
- [x] 独立整体代码审阅；集中修复重要问题并验收。
- [x] 更新 changelog 与验证边界，不提交 git。

## 执行记录

- Ruling: 无 git 仓库且用户明确指定独立工程目录；以该新目录隔离，不运行 worktree/提交类操作。
- Ruling: 按用户要求以完整模块集中进行 red/green 验收，不按每个小函数反复跑测试。
- Pre-flight: Task 1 的 ModelSpec 与 IPC 供 Task 2 使用；统一绝对动作语义；Task 3 工具消费相同接口。
- Task 1: complete — 首轮模型8项；实际服务器三任务元数据已只读保存用于契约验收；不下载权重。
- Task 2: complete — 控制首轮10项；mock TCP及start.sh完整姿态生命周期通过。
- Final review: 已完成独立代码审阅，集中修复强制释放旁路、UI异常自动释放、同维错臂契约、单侧救援、过期计划、首块期限、相机断流、遗留释放意图、模拟触碰真实worker、模型进程身份等问题。
- Final: Ruling: 审阅列出的真实GPU/现场性能属于明确的待上机验收，不作为离线完成证明；保留benchmark/check/shadow步骤，当前不连接Ubuntu。
- Final: Ruling: 救援采用显式 --rescue 和逐侧身份绑定，不在模拟模式或异常退出后隐式运动；Linux pidfd为安全终止已核验worker的要求。
- Final: minor fixed — 菜单在展示前解析自动执行步数，扫描时过滤缺统计文件的checkpoint。无暂缓minor项。

- Task 3: complete — 最终37/37离线测试；CPU处理器/严格加载保护通过；实际start.sh模拟演练通过；41文件Py3.10语法和Bash语法通过。真实GPU/真机保留待验收。
