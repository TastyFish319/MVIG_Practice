#!/usr/bin/env bash
# ACT 部署一键启动（在 mvig-ubuntu 上运行）
#
#   bash start.sh               交互式：选任务 → 选权重 → 选模式 → 选执行方式 → 控制台
#   bash start.sh --check       只读环境自检（解释器 / torch+CUDA / 配置 / 权重 / CAN / 相机）
#   bash start.sh --selftest    离线自测（不动硬件）
#   bash start.sh --rescue      救援：遗留 worker 保持中的手臂 → 平滑回零 → 断电
#   bash start.sh --mode sim    其余参数原样传给 act_deploy.app.main（--task/--checkpoint/--preset ...）
#
# 两个解释器默认取 SI 仓库（本目录的上一级）里已有的环境，可在 local.env 覆盖：
#   DEVICE_PY  设备环境 Py3.10：<SI>/.venv/bin/python
#   MODEL_PY   模型环境 Py3.12：<SI>/.superinference/local/act-inference-py312/bin/python
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
[ -f local.env ] && . ./local.env
SI_ROOT="${SI_ROOT:-$(cd "$HERE/.." && pwd)}"
DEVICE_PY="${DEVICE_PY:-$SI_ROOT/.venv/bin/python}"
MODEL_PY="${MODEL_PY:-$SI_ROOT/.superinference/local/act-inference-py312/bin/python}"
export MODEL_PY HF_HUB_OFFLINE=1 PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -x "$DEVICE_PY" ]; then
  echo "错误：设备环境解释器不存在: $DEVICE_PY （在 local.env 里设置 DEVICE_PY）"; exit 1
fi

case "${1:-}" in
  --rescue)   shift; exec "$DEVICE_PY" -m act_deploy.tools.rescue "$@" ;;
  --check)    exec "$DEVICE_PY" -m act_deploy.tools.check_env --model-python "$MODEL_PY" ;;
  --selftest) exec "$DEVICE_PY" -m act_deploy.tools.selftest ;;
  -h|--help)  sed -n '2,13p' "$0"; exit 0 ;;
esac

if ! "$DEVICE_PY" -m act_deploy.tools.rescue --check-only; then
  echo "检测到上一次运行遗留的机械臂 worker（手臂可能仍通电保持），先进行救援。"
  "$DEVICE_PY" -m act_deploy.tools.rescue || exit 1
fi

"$DEVICE_PY" -m act_deploy.app.main --model-python "$MODEL_PY" "$@"
rc=$?
stty sane 2>/dev/null || true

# The console exited while arm workers are still holding (crash / 3x Ctrl+C) -> rescue them.
if ! "$DEVICE_PY" -m act_deploy.tools.rescue --check-only; then
  echo
  echo "⚠ 控制台已退出，但机械臂 worker 仍在运行（手臂通电保持）→ 进入救援"
  "$DEVICE_PY" -m act_deploy.tools.rescue
fi
exit $rc
