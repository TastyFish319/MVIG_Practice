# ACT_train_new 独立工程设计

状态：实现与本轮验收已完成；真实4卡与长训练未验收，见acceptance.md。日期：2026-11-05。

用户补充优先于正文旧描述：本轮暂不开源、不整理许可证声明与下载命令；所有实现完成后仅做最后集中验收，不进行逐模块测试。

## 1. 目的与已确认的路线

让一个只接触过少量具身智能训练的使用者，把整个工程复制到Linux服务器后，完成：

```text
raw_data/任务 + 两份审核JSON
          ↓ 选择数据、预览验证集编号、确认
dataset/数据集（train/val、trace、manifest、训练集统计）
          ↓ 选择状态输入、1/2/4卡、超参数、训练名
runs/训练名（checkpoint、日志、曲线、状态）
          ↓ 自动用相同设备预算评测所有保留checkpoint
离线HTML + 逐关节曲线/预测CSV + 排名 + 训练名_statistics.zip
```

用户已选择保留现有 LeRobot 0.4.4 ACT 与 v2/v3 语义、重构外围工程。
目录、文档和操作流程采用统一约定；模型参数、状态顺序和数据频率遵循 ACT 契约。

### 方案比较

| 路线 | 好处 | 代价 | 决定 |
|---|---|---|---|
| 保留模型与数据契约，整理训练外围 | 可与已有权重/部署/实验对照，变更范围清楚 | 需要针对0.4.4维护薄适配层 | 用户已选 |
| 整理时升级LeRobot | 新API与后续模型扩展更方便 | 需要额外核验数据、处理器、权重和训练数值 | 本轮不做 |
| 只把旧目录复制独立 | 改动少 | 保留多版本入口、私有环境耦合、过期文档 | 不满足目标 |

## 2. 范围

包含预处理、数据验证、ACT训练、checkpoint验证、离线评测、HTML/ZIP、环境文档、调参指南、来源记录、轻量验收。
保留14维与21维状态选择；默认21维以衔接当前v3实验，14维作为对照。
本轮固定现有右臂动作与双腕观测；维度/关节映射集中定义，双臂输出作为未来扩展，不冒充已兼容。
不启动机器人、不搬入HTTP服务、不发布到公共仓库、不覆盖服务器旧ACT、不运行正式长训练。
目标是新工程可开源；数据、权重、内部路径不进入发布包，内部来源权限另行标注。

## 3. 目录与入口

```text
ACT_train_new/
  agents.md / changelog.md / README.md / pyproject.toml
  requirements.txt / requirements-dev.txt / .gitignore
  configs/                 preprocess、train、scoring默认JSON
  acttrain/
    data/                  MCAP读取、审核、对齐、split、转换、验证、旧集导入
    model/                 契约、相对动作、历史速度、归一化、推理、夹爪滤波
    training/              配置、官方trainer适配、多卡、验证探针、日志
    evaluation/            指标、多卡工作分配、逐episode结果、报告和打包
    cli/                   preprocess / train / evaluate / doctor / report
  scripts/                 对应bash包装脚本，使用当前环境python
  docs/                    中文环境、流程、调参、迁移、来源、验证记录
  tests/                   纯逻辑与模块集成验收
  raw_data/ / dataset/ / runs/  .gitkeep，其余忽略
```

交互式一键脚本和非交互CLI共用逻辑。非交互命令必须能完整记录有效配置，方便复现实验。
脚本从自身位置定位工程，不能依赖用户先进入某个旧仓库。
`doctor`检查Python、依赖版本、GPU数量、模型/数据契约、可写路径，不启动训练。

## 4. 数据路径

### 审核与划分

- 在raw_data中选择任务与审核文件对；核对task、source_fingerprint、approved集合与review决定一致。
- 拒绝待定条目、重复条目、技术失败、缺片、SHA256不符、路径越界。
- 新划分按episode编号分段，用固定seed每段抽取，均匀覆盖采集顺序；明确这是覆盖顺序，不保证场景分布平衡。
- 用户选验证比例、预览准确编号，可手工指定后确认；训练阶段不再次询问或重划。
- 旧数据导入保留原split；不改数据、不重算划分、不将follower标签当leader标签。

### MCAP到数据集

独立读取MCAP内嵌protobuf schema；不import旧consumer模块，不读设备SDK。
按metadata把相机topic映射左右腕；只接受已核验的NEDF3 RGB v2。
20Hz公共时间轴；观测取不晚于网格时刻的最近消息；最大源龄默认100ms，超限拒绝，不用未来插值填缺口。
图像JPEG解码后转RGB，等比例缩放补边256×256，使用LeRobot无损image存储模式。
不照搬Pi的静止裁剪、30Hz或224图像，以免改变已有ACT行为。

LeRobot train/val保留基础state14与未来leader绝对action7；trace保存原始leader、follower、时间戳、源索引。
预处理计算右臂历史速度存trace；旧数据可按同公式由trace生成。
manifest记录审核与源文件指纹、划分、关节顺序、时间契约、图像变换、有效窗口数。
`.partial`完整写入并通过官方读取器验证后，才发布dataset目录；已存在则拒绝覆盖。

## 5. 模型输入输出

| 选择 | 输入 | 输出 |
|---|---|---|
| 不含速度（兼容v2） | 左右腕RGB + follower state14 | 32×7 |
| 含历史速度（兼容v3，默认） | 同上 + 过去100ms右臂速度7，共state21 | 32×7 |

基础state顺序固定 `[左6,左夹爪,右6,右夹爪]`。
速度为 `(follower[t,7:14]-follower[t-2,7:14])/0.1`，每episode前两帧补零。
它是输入历史，不是额外的输出速度，也不使用主臂或未来信息。

第k个关节标签：`leader[t+1+k,右关节] - follower[t,右关节]`。
例如当前从臂角度1.0，未来主臂1.2、1.3，则标签为0.2、0.3，不是0.2、0.1。
推理时整块加回同一个当前从臂角度。夹爪各步始终为leader绝对开度。

直接复用LeRobot的ResNet18 + Transformer + CVAE ACT。训练loss含动作重建L1和KL项；推理不把目标动作交给CVAE，使用prior。
保存官方config、safetensors、pre/postprocessor及其状态，外加现有两种契约文件；部署方复制完整pretrained_model目录。
动作契约schema保持2，速度契约schema保持3；新元数据可另外保存，不能改变旧字段含义。

## 6. 训练

- Python3.12；LeRobot0.4.4；服务器现有Torch2.7.1/cu126、torchvision0.22.1作为兼容基线。
- 独立环境文档与依赖，不往已有act/pi_fish环境安装新包。
- 使用官方优化器、模型与训练循环，工程自己的适配集中封装，锁定版本并在退出时恢复上下文修改。
- 训练采样排除每episode尾部31个不完整chunk；数值统计只用train，float64、std下限1e-3。
- 1/2/4卡Accelerate+torchrun，batch明确为每卡batch，同时展示总batch；设备不足直接报错。
- 默认兼容服务器：policy lr=1e-5、backbone lr=1e-5、dropout=0.05、KL=10；提供显式覆盖。
- policy/backbone分别设置StepLR开关、初始lr、step_size和gamma；两组默认关闭。
- StepLR单位为完成的优化器更新次数，不随卡数暗中改变；步数、checkpoint间隔均显式记录。
- 默认steps=10000、每卡batch=16、checkpoint间隔1000、workers每卡2；选择多卡时显示样本预算，不声称等价单卡。
- 训练名必须为安全单目录名；已存在的run拒绝覆盖。CLI提供dry-run仅输出计划。
- 默认后台日志与监督进程，SSH断开可继续；status.json标明阶段、配置、开始/结束/失败。

### checkpoint验证与保留

每个保存点及最终点进行固定train子集和完整val的prior验证（可配置stride与batch）。
记录posterior训练L1、训练总loss、KL、train-prior和val-prior，避免把条件不同的loss混为一谈。
CSV、TensorBoard、PNG同时保存；只由主进程发布聚合结果。
默认保留所有ACT保存点及可恢复训练状态，衔接原有习惯；可选top-k+final减少空间。
best依据val-prior归一化L1选择并明确记录；离线综合排名另存，不能把两种best混写。
本轮不承诺从中断恢复精确训练；官方状态文件保留用于后续扩展，resume入口明确不支持。

## 7. 评测与报告

自动评测使用与训练相同的CUDA可见设备集合和请求卡数，只在checkpoint数量不足时减少实际并发并记录。
任务按checkpoint分配给独立GPU进程；同一checkpoint所有选中episode用同一协议。
任一worker失败则停止兄弟worker，不发布完整排行榜或最终best；单项结果和日志保留。

- 从val按固定seed选至多8条；全部checkpoint共用同一清单并记录。
- 每条episode每帧仅传观测推理；同时统计前4步和完整32步，尾部只算有效目标。
- 导出每关节原始预测CSV/压缩NPZ；曲线叠加leader标签、follower实测和模型预测，标明目标时间轴。
- 指标按episode等权，再跨episode平均，避免长episode支配结论。
- 综合误差默认保留旧版：0.40×前4步归一化MAE + 0.25×全chunk归一化MAE + 0.20×归一化中位绝对误差 + 0.15×归一化最大绝对误差；越小越好。
- 每个评分分项保留旧版的0.8×六关节归一化误差均值 + 0.2×夹爪归一化误差；新增权重实验必须显式修改评分版本，不能静默替换。
- 评分版本、权重与使用的训练集scale固定写入结果。关节原始rad/deg误差与夹爪开度误差分开呈现。
- 补充RMSE、P95、逐轴误差、夹爪F1/混淆矩阵作诊断；不默认把任意工程安全阈值计入排名。
- 对照旧排行时必须用相同val清单；另提供全val评测开关，不能把8条分数与旧全val分数直接比较。

单个离线HTML包含训练配置、数据摘要、曲线、指标定义、checkpoint排名、逐episode链接与误差解释。
ZIP只含浏览必需HTML/PNG、汇总JSON/CSV和可选逐关节数据，不含权重、视频、TensorBoard日志、缓存或原始episode。
设置轻量/完整统计开关，默认轻量；包内相对链接完整，离线打开无需CDN。
旧数据集使用显式路径或导入描述文件（指纹+路径）引用，报告对外展示时移除私人绝对路径。

## 8. 夹爪

保留leader绝对标签：它表达人的闭合意图，follower接触后保持开度不代表人想张开。
提供可选、默认关闭的推理软下限：`f(g)=g (g>=threshold)`，否则 `floor+(threshold-floor)*exp((g-threshold)/(threshold-floor))`。
要求 `0<=floor<threshold<=1`，保持阈值处连续、斜率连续，输出随模型减小平滑趋近floor。
不从验证集标定参数，不自动把单任务物体宽度当通用下限；用户按训练集/部署标定显式指定。
保留raw与filtered两份输出，综合排名始终使用raw；软下限不是夹持力控制或硬件保护。

## 9. 工程质量与开源边界

中文README提供从零安装到下载ZIP的完整命令；docs分别解释数据流、ACT原理、调参、环境、旧工程迁移。
pyproject声明可安装包与CLI；requirements声明实际使用版本，Torch按CPU/CUDA独立安装说明。
配置不含本机路径、私有服务器账号、自动上传；数据/权重不入版本管理；发布检查扫描敏感路径与大产物。
工程模块保持清晰的职责与边界，公共第三方模块通过依赖安装；本轮不整理许可证说明。

## 10. 集中验收与交付标准

| 阶段 | 验收 | 不能由此声称 |
|---|---|---|
| 数据模块完成 | 审核、路径边界、split、因果对齐、相对动作/速度、官方数据round-trip | GPU训练通过 |
| 训练模块完成 | 配置与两组LR、状态、真实ACT短训练/验证/保存/重载 | 长训练收敛或性能提升 |
| 评测模块完成 | 尾部mask、统一选样、指标、离线链接、ZIP内容、失败不发布 | 真机成功率 |
| 集成完成 | 新进程独立启动CLI；无旧仓库import；单卡真实小数据全流程 | 多卡通过 |
| 多卡可用时 | 小数据2/4卡训练和同设备评测，检查失败传播 | 线性加速或结果逐位等价 |

服务器验收只在本工程独立路径、独立环境和空闲GPU运行，使用少量真实数据与短步数，不动旧任务。
若环境安装或GPU条件不允许，在验收文档明确阻塞项，不能把模拟测试替代真实验证。
每完成一板块更新changelog；最终交付列明真实通过项、未通过项和启动命令。

## 11. 实现者可决定的细节

具体模块边界、复用函数与重写范围、HTML配色排版、测试fixture大小、依赖锁文件形式由实现者决定。
这些决定不得改变上面的数据/动作契约与用户已选路线；有行为变化需记录原因和对照验证。
