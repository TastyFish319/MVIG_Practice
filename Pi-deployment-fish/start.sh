#!/usr/bin/env bash
# π0.5 部署一键启动（在 mvig-ubuntu 上运行）
#
#   bash start.sh               交互式：选任务 → 选权重 → 选模式 → 选执行方式 → 控制台
#   bash start.sh --check       只读环境自检（解释器 / torch+CUDA / 配置 / 权重 / CAN / 相机）
#   bash start.sh --selftest    离线自测（不动硬件）
#   bash start.sh --rescue      救援：遗留 worker 保持中的手臂 → 平滑回零 → 断电
#   bash start.sh --mode sim    其余参数原样传给 pi_deploy.app.main（--task/--checkpoint/--preset ...）
#
# 两个解释器默认取 SI 仓库（本目录的上一级）里已有的环境，可在 local.env 覆盖：
#   DEVICE_PY  设备环境 Py3.10：<SI>/.venv/bin/python
#   MODEL_PY   模型环境 Py3.12：<本工程>/.venv-pi/bin/python
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
[ -f local.env ] && . ./local.env
SI_ROOT="${SI_ROOT:-$(cd "$HERE/.." && pwd)}"
DEVICE_PY="${DEVICE_PY:-$SI_ROOT/.venv/bin/python}"
MODEL_PY="${MODEL_PY:-$HERE/.venv-pi/bin/python}"
export MODEL_PY HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUTF8=1 PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -x "$DEVICE_PY" ]; then
  echo "错误：设备环境解释器不存在: $DEVICE_PY （在 local.env 里设置 DEVICE_PY）"; exit 1
fi

case "${1:-}" in
  --rescue)   shift; exec "$DEVICE_PY" -m pi_deploy.tools.rescue "$@" ;;
  --check)    shift; exec "$DEVICE_PY" -m pi_deploy.tools.check_env --model-python "$MODEL_PY" "$@" ;;
  --benchmark) shift; exec "$MODEL_PY" -m pi_deploy.tools.benchmark "$@" ;;
  --selftest) exec "$DEVICE_PY" -m pi_deploy.tools.selftest ;;
  -h|--help)  sed -n '2,13p' "$0"; exit 0 ;;
esac

"$DEVICE_PY" -m pi_deploy.app.main --model-python "$MODEL_PY" "$@"
rc=$?
stty sane 2>/dev/null || true

# Rescue is explicit (--rescue). A simulation run must never inspect/control real workers.
exit $rc
