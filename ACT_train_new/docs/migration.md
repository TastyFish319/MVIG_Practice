# 旧ACT数据与权重迁移

## 只读复用数据

新工程不import旧包。已有完整leader数据集可通过描述文件引用：

```bash
bash scripts/dataset.sh /path/to/existing/leader_dataset dataset/old_task
```

描述文件只记录数据位置和manifest SHA256，旧split、trace和统计完全保留。此文件属于运行数据，不纳入代码版本。跨机器移动时应调整引用或复制整个数据集；不可只复制Parquet。也可直接把旧完整数据集放入dataset/。

入口检查LeRobot0.4.4、20Hz/32步/下一帧、state14/action7、follower观测/leader标签、验证通过记录。旧follower标签数据被拒绝。原始审核数据重新预处理可能生成不同split，不能替换已有实验的输入。

## 完整权重目录

部署/评测需要完整pretrained_model/：config.json、model.safetensors、pre/postprocessor JSON及其状态文件、act_action_contract.json；速度版另有act_observation_contract.json。不只复制模型文件。

现有ACT_deployment_new按这两份契约识别state14/state21和相对关节。新工程保留字段语义，但未对真机部署进行验收；不要仅根据目录名猜模型类型。

## 本轮保留与变化

| 保留 | 变化 |
|---|---|
| LeRobot0.4.4、ACT架构、动作/状态语义 | 单一训练入口，速度作为开关 |
| 旧评分默认值、归一化尺度 | 默认固定至多8条val，--max-episodes 0可评测全部 |
| 服务器默认lr/dropout | 参数显式配置，双组StepLR可分别调整 |
| 完整checkpoint输出 | 增加TensorBoard、best记录、可选训练后精简 |
| 每帧/全chunk离线评测 | 自动识别v2/v3并支持同设备预算并发 |
| 审核与源文件指纹核对 | 独立MCAP读取，解除consumer和runtime_profiles依赖 |

默认后台任务不能resume，失败后换训练名重跑。代码结构兼容不代表改变batch、随机采样、选样或版本后结果逐位相同。
