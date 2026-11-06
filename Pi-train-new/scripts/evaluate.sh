#!/usr/bin/env bash
# 单独(重新)评测并打包某次训练：bash scripts/evaluate.sh runs/<名称>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
RUN="${1:?用法: bash scripts/evaluate.sh runs/<名称>}"
python -m pitrain.pipeline --run "$RUN" --skip-train
