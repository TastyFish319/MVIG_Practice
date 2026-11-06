#!/usr/bin/env bash
# 一键训练：复用 train.sh 的参数选择，完成后后台排队，空闲 GPU 足够时开始训练 + 评测 + 打包。
# 额外参数会透传，例如: bash scripts/train_wait.sh --idle-memory-mb 1024 --idle-utilization 10
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
python -m pitrain.cli.train_wait "$@"
