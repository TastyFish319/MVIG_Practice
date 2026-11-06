# ACT 来源核对（2026-11-05）

## 已核对的来源

| 来源 | 作用 | 结果 |
|---|---|---|
| mvig-training: output/ACT | 当前真实训练代码与运行记录 | 已连通，只读检查 |
| 本地 SuperInference/incubator/act_training | 代码备份 | 包内27个py文件，25个SHA256一致 |
| 相邻 Pi-train-new | 工程约定、目录、预处理思路 | 已读取agents、README、依赖和关键模块 |
| 相邻 ACT_deployment_new | 权重输出兼容目标 | 解析act_action_contract与act_observation_contract |
| LeRobot v0.4.4官方配置 | ACT模型默认与API核对 | 已通过GitHub连接器读取 |

## 本地与服务器差异

| 文件 | 本地 | 服务器 | 处理 |
|---|---|---|---|
| train_server.py | lr=1e-4，dropout沿用官方0.1 | lr=1e-5，dropout=0.05 | 保留服务器兼容配置，默认值显式写入配置 |
| serve.py | SHA256不同 | SHA256不同 | 训练工程不包含旧HTTP硬件服务，无需直接搬入 |
| 其他25个py | SHA256一致 | SHA256一致 | 可从本地核对训练与评测算法 |

`train_server.py` SHA256：

- 本地：`2b626c3922626f55a907ecf6bcc85505877168ec0c112b7f74409227623c00d2`
- 服务器：`5fc2d0e8cce2edf562ce79370c2c0db40e6be85d4af1f1c06ee65c0be0fc647d`

## 真实训练语义

- 时间：20Hz、32步、下一帧作为第一动作目标；与Pi默认30Hz不同。
- 状态顺序：左臂在前、右臂在后；与Pi默认动作臂在前不同。
- v2：state14；v3：state21，追加过去100ms的右臂速度，不是预测速度。
- 右臂关节相对于当前follower位置；leader夹爪为绝对开度。
- 已有部署直接读取上述两份契约；不能在整理时悄悄换维度、顺序或时间定义。
- 存在名为yellow_v3_run1但launch记录实际调用acttrainingv2的训练；目录名不能作为v3证据。

## 环境实测

| 依赖 | 服务器已有act环境版本 |
|---|---|
| lerobot | 0.4.4 |
| torch / torchvision | 2.7.1+cu126 / 0.22.1+cu126 |
| accelerate | 1.15.0 |
| datasets | 4.8.5 |
| numpy | 2.2.6 |
| mcap | 1.6.0 |
| protobuf | 6.33.6 |

rich、tensorboard不在当前act环境；新工程应单独声明与安装，不假设已有。
资源观察只代表核对当时：4张H20；GPU0已有显存占用；共享盘约78%已用。
Pi旧agents提到磁盘99%已用，不应照搬为当前事实。

## 数据与结构耦合

- 旧convert.py插入旧仓库sys.path并import `consumer.io.i2rt_nedf3_rerun`，不是完全独立。
- 旧trainv3.sh调用SuperInference的runtime_profiles.py与私有本机环境注册信息。
- 旧README开头仍描述follower动作标签，和后续leader训练入口不一致，不能直接继承。
- 旧trainv3代码通过受限上下文修改官方trainer的工厂/保存函数；应封装、锁版本、集中适配并验证退出后恢复。
- 已有可用leader数据集（包含蓝积木与黄色任务），旧数据只读复用，不能重划已有实验的split。
- 最初prompt中的本机 `Desktop/ACT/data/blue_block_v2` 当前不存在；审核文件可从服务器现有数据集provenance核对。

## 官方参考

- [ACT项目与论文](https://tonyzhaozh.github.io/aloha/)：图像+关节观测预测动作块；训练CVAE、推理使用prior。
- [LeRobot 0.4.4 ACTConfig](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/policies/act/configuration_act.py)：模型配置、优化器接口。
- [LeRobot 0.4.4许可证](https://github.com/huggingface/lerobot/blob/v0.4.4/LICENSE)：保留外部代码来源说明。

用户已确认本轮暂不公开发布；本页只记录技术来源。
