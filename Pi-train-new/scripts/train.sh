#!/usr/bin/env bash
# 一键训练：选数据集 → 卡数 → 超参 → 命名 → (后台) 训练 + 评测 + 打包 <名称>_statistics.zip
# 额外参数会透传，例如: bash scripts/train.sh --foreground
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
python -m pitrain.cli.train "$@"
