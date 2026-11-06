# 环境配置（服务器 mvig-training）

## 服务器现状（2026-11 只读查询）

| 项 | 值 | 影响 |
|---|---|---|
| GPU | 4 × NVIDIA H20 96GB | 支持 1/2/4 卡 |
| 驱动 | 535.183 / CUDA 12.6 | 用 **cu126** 版 torch |
| HuggingFace 官网 | ❌ 不通 | 用 `HF_ENDPOINT=https://hf-mirror.com` |
| PyPI / download.pytorch.org | ✅ | 直接 pip 安装 |
| 磁盘 | 共享盘剩余 ~1.8T | 权重只存 top-k + final |

## 一次性安装

```bash
cd ~/output/Fish/Pi-train-new
source ~/output/miniconda3/bin/activate

# 1) 创建环境（LeRobot 0.6.1 要求 Python >= 3.12）
conda create -n pi_fish python=3.12 -y
conda activate pi_fish

# 2) 先装 GPU 版 torch（必须单独装，否则 pip 会装成 CPU 版或不匹配的 CUDA 版）
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

# 3) 其余依赖
pip install -r requirements.txt

# 4) 验证 GPU
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# 期望输出: 2.8.0+cu126 True 4

# 5) 下载预训练权重与 tokenizer（~15GB，走 hf-mirror）
bash scripts/download_models.sh
```

## 依赖说明（为什么需要它）

| 包 | 用途 |
|---|---|
| `lerobot[pi]` 0.6.1 | pi0.5 模型、数据集格式、归一化、相对动作处理器 |
| `transformers` | PaliGemma / Gemma 网络结构与 tokenizer |
| `accelerate` | 多卡 DDP、bf16 混合精度 |
| `datasets` / `pyarrow` / `pandas` | LeRobot 数据集底层存储（parquet） |
| `mcap` + `protobuf` | 读取原始 NEDF3 数据 |
| `opencv-python-headless` | JPEG 解码、缩放 |
| `tensorboard` | 训练曲线（成熟方案） |
| `matplotlib` | 静态 PNG 曲线、评测图 |
| `scikit-learn` | 评测指标（F1 等） |
| `Jinja2` | HTML 报告模板 |
| `rich` | 终端美化 |

## 常见问题

- **`torch.cuda.is_available()` 为 False**：多半是装成了 CPU 版，重新执行第 2 步。
- **tokenizer 403**：`google/paligemma-3b-pt-224` 是 gated 仓库；脚本会自动退回社区副本。
  若想用官方：在 HF 网站同意 Gemma 协议后 `HF_TOKEN=hf_xxx bash scripts/download_models.sh`。
- **DataLoader 报 shared memory / bus error**：把 `num_workers` 调小或设为 0。
- **本地 Windows 开发环境**：`pi_fish_dev`（CPU 版 torch）只用于跑预处理和单元测试，不训练。
