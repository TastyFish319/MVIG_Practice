#!/usr/bin/env bash
# ------------------------------------------------------------
# 下载 pi0.5 预训练权重 + PaliGemma tokenizer 到 pretrained/
#
#   pretrained/pi05_base/            lerobot/pi05_base（~14.5GB，float32）
#   pretrained/paligemma_tokenizer/  只下载 tokenizer 文件（~40MB）
#
# 用法:
#   bash scripts/download_models.sh                # 默认走 hf-mirror.com
#   HF_TOKEN=hf_xxx bash scripts/download_models.sh   # 有 token 时用 google 官方 tokenizer
#   HF_ENDPOINT=https://huggingface.co bash scripts/download_models.sh  # 能直连官网时
# ------------------------------------------------------------
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
PRE="$ROOT/pretrained"
mkdir -p "$PRE"

echo "==> HF_ENDPOINT = $HF_ENDPOINT"
echo "==> [1/2] 下载 lerobot/pi05_base -> $PRE/pi05_base"
hf download lerobot/pi05_base --local-dir "$PRE/pi05_base"

TOK_FILES=(tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json added_tokens.json)
echo "==> [2/2] 下载 PaliGemma tokenizer -> $PRE/paligemma_tokenizer"
if [[ -n "${HF_TOKEN:-}" ]] && hf download google/paligemma-3b-pt-224 "${TOK_FILES[@]}" \
      --local-dir "$PRE/paligemma_tokenizer" 2>/dev/null; then
  echo "    使用 google/paligemma-3b-pt-224 官方 tokenizer"
else
  # google 官方仓库是 gated（需要登录并同意 Gemma 协议）。没有 token 时使用社区镜像
  # leo009/paligemma-3b-pt-224（只取 tokenizer 文件，不含任何可执行代码）。
  echo "    未提供 HF_TOKEN 或官方仓库不可用 -> 改用社区副本 leo009/paligemma-3b-pt-224"
  hf download leo009/paligemma-3b-pt-224 "${TOK_FILES[@]}" --local-dir "$PRE/paligemma_tokenizer"
fi

echo "==> 校验"
python - <<PY
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$PRE/paligemma_tokenizer")
ids = tok("Task: pick the blue block, State: 1 2 3;\nAction: ")["input_ids"]
print("    tokenizer OK, vocab =", len(tok), ", 示例 token 数 =", len(ids))
assert len(tok) >= 257152, "词表大小不对，tokenizer 可能不是 PaliGemma 的"
PY
du -sh "$PRE"/* 2>/dev/null || true
echo "==> 完成"
