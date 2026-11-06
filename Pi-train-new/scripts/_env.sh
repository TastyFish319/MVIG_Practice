#!/usr/bin/env bash
# 公共环境：激活 conda 环境 pi_fish（可用 PI_CONDA / PI_ENV 覆盖）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI_CONDA="${PI_CONDA:-$HOME/output/miniconda3}"
PI_ENV="${PI_ENV:-pi_fish}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$PI_ENV" ]]; then
  if [[ -f "$PI_CONDA/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$PI_CONDA/bin/activate" "$PI_ENV"
  else
    echo "⚠ 未找到 $PI_CONDA/bin/activate，使用当前 python：$(command -v python)"
  fi
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
