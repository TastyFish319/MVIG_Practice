# ACT 独立工程实施计划

> **For agentic workers:** 使用 superpowers:executing-plans 在当前会话连续实施。用户批准直接继续，验收集中到最后，不做逐项红绿测试、不创建提交、不新增工作树。

**Goal:** 将现有ACT训练方法整理为独立的预处理→训练→评测→统计报告工程。
**Architecture:** 模型/数据契约兼容LeRobot0.4.4；复用核验过的数值与训练适配实现，按data/model/training/evaluation拆分。交互CLI与非交互CLI统一配置与后台流水线。
**Tech Stack:** Python3.12、LeRobot0.4.4、Torch2.7.1、Accelerate、rich、TensorBoard、matplotlib。
**Spec:** ../specs/2026-11-05-act-independent-engineering-design.md

## Global Constraints
- 20Hz、32步、下一帧目标、state14/21、leader相对关节/绝对夹爪。
- 本目录非Git仓库，直接在用户指定的新目录工作；不复制已有数据与权重。
- 用户覆盖技能默认测试节奏：只做最终验收；本轮不开源，不整理许可证/下载命令。
- 不依赖旧仓库、不改旧环境/任务；仅最终独立路径短训练验收。

## Review Focus
- 已有输出/partial拒绝覆盖，坏配置不能启动后台训练。
- 审核嵌套路径、MCAP分片路径越界、审核集合不一致拒绝。
- v2/v3混合checkpoint按契约选择运行时，统一val选样和评分。
- 多卡失败不发布排名，训练/评测使用同一可见设备集合。
- 精简checkpoint不删除final或best；ZIP所有离线链接有效且无权重。

## Task 1: 数据与契约
- [x] 迁移core/relative/velocity到acttrain/data与model，重写包内导入。
- [x] 独立MCAP读取器、审核与分层split、转换和官方读取验证；partial成功发布。
- [x] 严格数据发现与旧数据引用；编写最终验收用例（暂不运行）。
接口：convert(raw_root, config_path, output_override=None) -> Path；validate(root) -> dict；resolve_dataset(path) -> Path。

## Task 2: 训练
- [x] 统一worker、配置、14/21维开关；复用官方trainer与双组StepLR。
- [x] CSV/TensorBoard、prior验证、best与保留策略；后台状态与失败传播。
接口：TrainingConfig.from_dict(dict)、worker.main(argv)、pipeline.run(plan)。

## Task 3: 评测和交互
- [x] 统一v2/v3 evaluator，checkpoint并发worker，固定≤8条val。
- [x] 原始评分兼容、RMSE/P95/F1诊断、逐关节预测与图、离线报告ZIP。
- [x] preprocess/train/evaluate/doctor/report CLI和bash包装。
接口：evaluator.main(argv)、build_report(run,evaluation,light=True) -> Path。

## Task 4: 文档和最终验收
- [x] 中文README、环境/流程/调参/迁移、依赖与配置、changelog。
- [x] 最后一次集中静态/单元/CLI/打包验收；真实数据预处理与真实ACT单卡短流水线；空闲GPU允许时2/4卡短验收。
- [x] 一次独立代码审阅；修复实质问题并对受影响项复验；记录真实结果与未验证边界。
## 最终结果
- 19项集中用例、真实预处理、单卡state21和双卡state14的训练→评测→报告通过；4卡实际运行因GPU0占用未做，已在acceptance.md标明。
- 一次代码审阅的4项问题均修复；未产生Git提交或公开发布。
