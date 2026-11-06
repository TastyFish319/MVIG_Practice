#!/usr/bin/env bash
# 一键预处理：选择 raw_data 中的数据 → 读审核记录 → 划分(需确认) → 写 dataset/
# 额外参数会透传，例如: bash scripts/preprocess.sh --no-verify-hash --workers 16
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
python -m pitrain.cli.preprocess "$@"
