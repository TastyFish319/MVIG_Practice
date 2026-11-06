# 技术来源与整理范围

本轮暂不公开发布，不整理许可证声明；本页仅记录技术来源与改动，便于后续维护。

| 新模块 | 来源 | 整理内容 |
|---|---|---|
| data/core.py、convert.py、validate.py | 旧yam_act同名文件 | 包内导入、独立记录读取、严格配置、分段split、速度trace、验证后发布、显式异常 |
| model/relative.py、velocity.py、inference.py、velocity_inference.py | 旧yam_act同名文件 | 保留数值公式和权重契约，移至统一包 |
| training/distributed.py、metrics.py、validation.py、schedule.py | 旧distributed/training_metrics/validation_probe/step_lr | 作用域适配、TensorBoard、best记录、验证失败传播 |
| training/worker.py | 旧acttrainingv3.py并参照服务器train_server.py | 统一速度开关，明确lr/dropout，独立缓存 |
| evaluation/runner.py、parallel.py、episodes.py、episode_io.py、metrics.py | 旧evaluate_v3/parallel_evaluation/episode_evaluation/episode_io/evaluation_metrics | 自动选v2/v3、固定样本/offset、可配置评分、P95/F1、过滤另存 |
| data/records.py | NEDF3消息结构、Pi/旧ACT解码方法 | 独立实现必要读取，不包含Rerun/SDK/设备逻辑 |
| CLI、config、pipeline、report | 本轮新增，参考Pi目录与交互约定 | 独立工程启动、后台监督、报告、环境检查 |

旧来源位于SuperInference/incubator/act_training；25个核心Python文件本地与服务器SHA256一致。训练默认以服务器真实配置为准，见source_audit.md。

模型、优化器和官方训练算法均从安装的LeRobot0.4.4调用，没有复制官方整套模型实现。针对版本的工厂/保存/采样器适配集中在上下文管理器，退出后恢复原值。将来升级LeRobot需要单独验证适配接口。
