# Ubuntu 安装、权重搬运和验收

本文命令供 Ubuntu 可连接后执行。开发期间没有连接 `mvig-ubuntu`，没有在它上面安装软件或修改环境。

## 1. 环境分工

| 环境 | 用途 | 处理方式 |
|---|---|---|
| `<SI>/.venv/bin/python`（原 Python3.10） | i2rt/CAN、portal、RealSense、相机、控制台 | 直接复用；**不要向它安装 requirements.txt** |
| 本工程 `.venv-pi/bin/python`（Python3.12） | torch、LeRobot pi05、tokenizer | 新建独立环境；原 ACT 推理环境保留 |

原 ACT 模型环境是 LeRobot0.4.4，Pi 训练环境是0.6.1，不能只更换权重直接复用。可以用原 ACT 环境提供的 Python3.12 创建新 venv，pip 缓存可复用，避免改设备环境。

```bash
# 在 Ubuntu 上：先把完整 Pi-deployment-fish 文件夹复制到选定目录，然后进入
cd /你的路径/Pi-deployment-fish

# 以下是 ACT 记录的 SI 路径示例；以本机实际路径为准
SI_ROOT='/home/team1/桌面/Project_MVIG/SuperInference-feat-i2rt_NEDF3'
"$SI_ROOT/.venv/bin/python" --version
"$SI_ROOT/.superinference/local/act-inference-py312/bin/python" --version

# 应为 Python3.12；新环境不继承原 ACT 的第三方库
"$SI_ROOT/.superinference/local/act-inference-py312/bin/python" -m venv .venv-pi
.venv-pi/bin/python -m pip install --upgrade pip -i https://pypi.mirrors.ustc.edu.cn/simple

# CUDA 专用 wheel 使用 PyTorch 官方源；普通 PyPI 镜像不负责此 cu128 wheel 索引
.venv-pi/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv-pi/bin/python -m pip install -r requirements.txt \
  -i https://pypi.mirrors.ustc.edu.cn/simple
.venv-pi/bin/python -m pip check

cp local.env.example local.env
# 编辑 local.env，将 SI_ROOT、DEVICE_PY、MODEL_PY 改为当前主机实际路径
```

如果已有**匹配这些依赖且可正常运行 CUDA 的独立 Pi 环境**，直接把其解释器填入 MODEL_PY，无需重建。无需先激活两个环境，start.sh 会分别调用它们。工程可以放在 SI 外面，只要 local.env 指向正确解释器。

5090 是 Blackwell 架构；选择 cu128 而不照搬 H20 训练机的 cu126。[PyTorch 官方提供 torch2.8.0 / torchvision0.23.0 cu128 组合](https://pytorch.org/get-started/previous-versions/)，[Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)。系统驱动必须支持该 GPU 与运行时，`nvidia-smi` 可只读检查；不要仅凭“系统装了 CUDA”推断 PyTorch 已可用。通常使用官方 wheel 无需单独编译 CUDA toolkit。

## 2. 搬权重和 tokenizer

训练机已确认根目录：`/home/users/zhaoruiqi_yizu/output/Fish/Pi-train-new`。训练机上的不同任务可能还在更新 top-k，优先拷**已完整保存**的目录，记录实际 checkpoint 名。

| 任务 | runs 下已发现的训练目录 | 部署目标 |
|---|---|---|
| blue_block | blue_block_2gpu_run1_18500steps | checkpoints/blue_block/<run名>/<step目录> |
| fold_towel | fold_towel_1gpu_run1_10000steps | checkpoints/fold_towel/<run名>/final 或 step目录 |
| open_canister | open_canister_run1_10500steps | checkpoints/open_canister/<run名>/final 或 step目录 |
| yellow_block | yellow_block_2gpu_run1_9500steps | 待实际生成完整 checkpoint 后搬运 |

如果 Ubuntu 能通过 SSH 访问训练服务器（需要在 Ubuntu 配置地址/用户名/跳板，Windows 的 SSH 别名不会自动同步）：

```bash
# 在部署工程目录运行；mvig-training 必须已经在 Ubuntu 本机配置
mkdir -p checkpoints/fold_towel/fold_towel_1gpu_run1_10000steps
rsync -avP --partial \
  mvig-training:/home/users/zhaoruiqi_yizu/output/Fish/Pi-train-new/runs/fold_towel_1gpu_run1_10000steps/checkpoints/final \
  checkpoints/fold_towel/fold_towel_1gpu_run1_10000steps/

mkdir -p pretrained
rsync -avP --partial \
  mvig-training:/home/users/zhaoruiqi_yizu/output/Fish/Pi-train-new/pretrained/paligemma_tokenizer \
  pretrained/
```

若 Ubuntu 不能直连，可先在 Windows 使用 `scp -r mvig-training:<完整源目录> <本地目标>`，再把完整文件夹传到 Ubuntu。不要依赖 Windows 创建的指向外部文件的软链接。建议对源和目标的 model.safetensors 分别做 `sha256sum` 核对。

- 每份权重约9.35GB；环境、pip缓存、加载时内存另计，先检查可用磁盘和内存。不要一次复制所有 top-k。
- 必须保留处理器 JSON 和其引用的 safetensors；归一化统计不是训练时才用，推理也需要。
- tokenizer 只需复制一份；`inference.tokenizer` 默认解析到本工程 `pretrained/paligemma_tokenizer`。
- checkpoint 中保存的训练服务器 tokenizer 绝对路径会被加载器覆盖，不必手动编辑权重文件。
- 如果选择其他 step，菜单会自动发现；`.partial` 路径会跳过。复制完成前不要启动部署。
- 本工程默认 HF 离线；缺模型或 tokenizer 会明确报错，不静默联网下载。

## 3. 先做不动机器人的检查

```bash
# 纯软件测试，不需要权重、GPU、portal或RealSense，不上电
bash start.sh --selftest

# 环境自检，不上电；--skip-hardware 可只看软件
bash start.sh --check --task fold_towel

# 真实权重GPU加载+推理，不连接机械臂或相机
bash start.sh --benchmark \
  --checkpoint checkpoints/fold_towel/fold_towel_1gpu_run1_10000steps/final \
  --gripper-floor checkpoint --iterations 20 --output logs/towel_gpu.json
```

基准工具记录加载时间、首次推理、稳态p50/p95、输出形状和峰值分配显存。默认合成输入只能验证加载、推理数值和速度，不能说明真实任务表现。可传 `--observation file.npz`，其键为 state（按契约排序）及图像完整特征名，图片须是已预处理 uint8 HWC。基准只计算模型段耗时，真实相机/TCP全链路还要在 shadow 看。

返回码：0=检查或模型时延预算通过；benchmark 返回2=时延余量不足；异常为非0。自检不等于权重加载验证。

## 4. 再接入硬件

1. 关闭其他会占用 CAN/相机或端口1234/1235的采集与部署程序，保留物理急停可用。
2. 核对 `hardware.yaml`：右 follower can2、左 follower can3；相机序列号取 ACT 原值，实物若更换需修改。
3. `bash start.sh`：先选任务和权重，再选 **shadow**，选择 floor；确认后模型加载并合成预热，之后才启动硬件保持。
4. 面板显示“实时预热完成”并通过时延预算后，Enter 平滑初始化；检查姿态与空间，再 Enter 观察模型目标。
5. 验证空格保持、z 回零、r 回零后释放。确认初始/零位及反馈正常后，再进入 real。

软件没有双臂碰撞模型或障碍物规划；复用的固定姿态和回零路线需要现场确认畅通。软件急停保持不是切断电源，也不是替代物理急停。

## 5. 常用配置与故障

| 配置/现象 | 意义 / 处理 |
|---|---|
| execution.chunk_auto.exec_steps=0 | 使用权重的 n_action_steps；当前一般15步=0.5s，并非每0.5s才发送一次电机指令 |
| policy_server.warmup_iterations | 合成输入预热次数，输出丢弃 |
| policy.live_warmup_iterations | 真实观测预热次数，输出丢弃；未通过禁止执行 |
| policy_server.request_timeout_s | 一次推理RPC等待上限；控制循环独立运行，不会等待这5秒 |
| policy.first_plan_timeout_s | 首块有效动作迟迟没到，停止策略并保持 |
| policy.plan_margin_s | 新计划必须保有的可执行时间余量 |
| policy.starvation_timeout_s | 动作覆盖耗尽后的容忍时间；超过则急停保持 |
| motion.settle_timeout_s | 轨迹到达目标时间后，还允许反馈追上的时间；未到位不继续、不释放 |
| safety.workspace_margin | 从输入 follower state 的训练 min/max 构建示范范围余量；不使用关节增量统计作绝对限位；不是物理关节限位 |
| 夹爪开关 | 菜单每次选，或 `--gripper-floor off`；参数写 session.json 与面板 |
| 模型加载失败 | 查看 server.log；检查完整目录、tokenizer、版本和显存；不会回退随机初始化模型 |
| 预热超预算 | 面板阻止执行；查看GPU占用和基准数据，调整模型开销/执行步数后重新启动；不随意放宽反馈安全限制 |
| 相机或状态过期 | 原地保持，排查USB/CAN；修好后按z恢复回零 |
| 回零未到位 | 保持通电，检查卡住/负载/姿态；不自动断电 |
| 遗留worker | 运行 `bash start.sh --rescue` 显式救援；核验归属后逐侧平滑回零 |
| 只剩一臂worker / 其他工程占端口 | 救援可逐侧处理存活且身份匹配的本工程worker；不可连接或不属于本工程的侧拒绝接管，需现场检查，不能强制杀进程 |
| Ctrl+C在救援过程中 | 终止运动，worker保持；不会为了退出而释放 |

不要通过 `kill -9` 解决机械臂程序卡住的问题；进程终止可能导致电机失力。优先软件保持和救援，必要时用现场物理急停与人工支撑。

救援进程通过 Linux pidfd 固定已核验 worker 的进程身份，避免 PID 文件被替换后误杀其他进程。目标 Ubuntu 的 Python3.10 和内核需支持 `os.pidfd_open` / `signal.pidfd_send_signal`；可用原设备解释器只读检查：

```bash
"$SI_ROOT/.venv/bin/python" -c "import os,signal; print(hasattr(os,'pidfd_open'), hasattr(signal,'pidfd_send_signal'))"
```

缺少 pidfd 时救援明确拒绝发送进程信号。普通模拟无需此能力。控制台异常退出只请求原地保持并结束控制会话，不自动执行回零释放；明确运行 `bash start.sh --rescue` 后才进行救援运动。
