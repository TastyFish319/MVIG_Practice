# 最终代码审阅

日期：2026-11-05；一次独立只读审阅，未进行逐模块重复审阅。

| 发现 | 修复 | 最终验收覆盖 |
|---|---|---|
| 已有partial被错误写入失败状态 | 只有本次attempt令牌匹配才更新状态；既有目录提前拒绝 | test_existing_partial_status_is_never_modified |
| MCAP索引可以遗漏审核过的分片 | 索引集合必须完整匹配目录MCAP集合，限定后缀、唯一与路径边界 | test_mcap_inventory_rejects_escape_and_duplicate |
| episode/data目录软链接可越界 | 审核与读取分别检查resolve后的data仍在episode内 | test_data_directory_symlink_cannot_escape_episode |
| Popen返回前停止可遗留独立进程组 | spawning/pending_signal缓冲；登记后停止，清理期间屏蔽二次信号并恢复原处理器 | test_signal_during_spawn_cleans_child_and_restores_handlers |

审阅者核对了LeRobot0.4.4官方trainer/config/sampler接口，未发现默认训练接口阻断；真实运行结论由acceptance.md记录。上述修复进入最后集中验收，不新增逐项审阅轮次。