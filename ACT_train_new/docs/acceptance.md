# 最终验收记录

日期：2026-11-05（北京时间）。结果：本轮集中验收通过。只执行最后的集中测试，没有逐模块重复运行。

## 检查结果

| 项目 | 证据与结果 |
|---|---|
| 静态编译 | 本地Python编译通过；运行包无yam_act、consumer、runtime_profiles或外部sys.path注入依赖 |
| 集中用例 | `python -m unittest discover -s tests -v`：19项通过、0失败，约17秒 |
| CLI | 六个入口的help与非交互训练计划通过；dry-run未创建run或后台任务目录 |
| 真实预处理 | 两条已审核蓝积木episode；1 train / 1 val；861帧、799合法窗口 |
| 官方读取核验 | 861帧数值、4个完整chunk读回、12次代表图像RGB指纹核对通过；下一帧对齐与源时间因果检查通过 |
| state21单卡 | 物理GPU1，真实ACT训练2次优化器更新，保存2个完整checkpoint；验证、重载、495帧逐帧评测、排名和ZIP通过 |
| state14双卡 | 物理GPU1/2，真实DDP训练2次更新，2个checkpoint分配给两个GPU评测worker；报告与ZIP通过 |
| 独立学习率 | 两组初始1e-5；策略StepLR间隔1/gamma0.5，更新后记录5e-6、2.5e-6；骨干始终1e-5 |
| TensorBoard | 两轮各写出1份事件文件；同时有CSV、PNG、离线HTML |
| 统计包 | 单卡2,622,615字节；双卡2,592,883字节；ZIP完整性、离线链接与无权重检查通过 |
| 失败与路径 | 审核指纹变化、MCAP遗漏/越界、目录软链接越界、已有partial不改写、并行失败不发布排名、启动时SIGTERM回收子进程均通过 |
| 代码审阅 | 一次独立审阅发现的4项问题均已修复并纳入上述验收，见review.md |

服务器新工程放在`~/output/ACT_train_new`，没有覆盖`~/output/ACT`。使用新工程自己的venv overlay，基础Torch/LeRobot来自现有已验证act环境，新增依赖从服务器现有Python3.12分发复制到overlay；旧act与pi_fish环境未安装或修改包。

## 版本与可检查产物

精确主要依赖见 [requirements-validated.txt](../requirements-validated.txt)；机器可读的运行状态、权重契约、设备、学习率和体积见 [acceptance_evidence.json](acceptance_evidence.json)。

本地留有一份轻量验收样例：`runs/acceptance_state21_1gpu/statistics/report.html`。该目录只包含报告产物，没有原始MCAP或模型权重；服务器保留完整短验收产物。

## 本轮没有证明的内容

- GPU0已有显存占用，未干扰该进程；**真实4卡训练/评测未运行**。1/2/4卡启动坐标、设备数量约束与checkpoint分配逻辑已覆盖，不能写成4卡实跑通过。
- 每轮只有2次更新，证明流程、保存/重载和契约正确，不证明训练收敛、超参数优劣、性能提升或线性加速。
- 未运行长训练、恢复训练、机器人闭环和真机成功率评测；resume入口明确不支持。
- 真实数据验收仅2条episode；大规模数据、更多场景和固定8条val抽样的实验效果需正式使用时观察。

本轮不开源、不整理许可证，不提供数据/权重下载命令。正式训练仍使用README的一键入口与用户自己的数据。
