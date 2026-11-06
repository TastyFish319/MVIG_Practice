# 调参指南

## 自动默认值（随数据集大小 / 卡数变化，已实现）

| 参数 | 规则 | 本数据（约 64 条 / 3.8 万帧）示例 |
|---|---|---|
| 全局 batch | 32，平分到每卡 | 1卡 32 / 2卡 16×2 / 4卡 8×4 |
| steps | ≈15 epoch × 帧数 / 全局batch，限制在 [3000, 30000] | ≈ 18000 |
| lr | 2.5e-5 × √(全局batch/32) | 2.5e-5 |
| warmup | min(1000, 5% steps) | ≈ 900 |
| 验证间隔 | steps / 10 | ≈ 1800 |
| StepLR 间隔 | steps / 3 | ≈ 6000 |

> 卡数增加时默认保持全局 batch 不变 → 每步更快、优化过程等价。想用多卡跑更大 batch，改"每卡 batch"，lr 会按平方根自动建议。

## 看曲线判断

| 现象 | 可能原因 | 调整 |
|---|---|---|
| train loss 降、val_nmae 先降后升 | 过拟合 | 用 best 权重；减少 steps；开图像增强；冻结视觉编码器 |
| train loss 几乎不降 | lr 太小 / 冻结过多 | lr ×2；关闭 train_expert_only |
| loss 剧烈抖动、grad 很大 | lr 太大 | lr ÷2；加长 warmup |
| val_nmae 到结尾还在降 | 训练不足 | steps ×1.5 |
| 显存不足 | batch 太大 | 减小每卡 batch；确认梯度检查点开启；冻结视觉编码器 |
| 夹爪 F1 低 | 夹爪时机没学好 | 检查数据中夹爪动作是否一致；多训练 |
| 衔接跳变大 | chunk 间不连贯 | 部署时减小 n_action_steps（更频繁重新推理） |

## 各参数建议范围

| 参数 | 建议 | 说明 |
|---|---|---|
| lr | 1e-5 ~ 5e-5 | 微调大模型宁小勿大 |
| steps | 5k ~ 30k | 小数据 10~20 epoch |
| StepLR gamma | 0.3 ~ 0.7 | 间隔约 steps/3 |
| keep_top_k | 1 ~ 3 | 每个约 7GB |
| freeze_vision_encoder | 数据 < 30 条时考虑开 | 防过拟合 |
| train_expert_only | 资源紧张时 | 效果通常下降 |
| n_action_steps | 10 ~ 25（30Hz） | 越小越反应快，但推理更频繁 |
| dropout | 不适用 | pi0.5 结构中没有 dropout 参数 |

## 推荐实验顺序

1. 默认参数、不含速度、4 卡 → 基线
2. 同参数、含速度 → 对比综合分
3. 若过拟合：开 freeze_vision_encoder 或减 steps；若欠拟合：加 steps
4. 真机上只部署评测第一名，必要时微调夹爪 floor
