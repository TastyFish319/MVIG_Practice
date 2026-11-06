# 环境与依赖

目标环境为Linux + Python3.12 + NVIDIA H20（1/2/4卡）。本地Windows用于编辑与纯数值检查，不从Windows启动训练。整个工程不需要SuperInference的环境注册表，也不需要设备SDK。

## 独立环境

在已安装的miniconda中创建一个新环境，避免修改正在使用的act或pi_fish：

```bash
conda create -n act_train_new python=3.12 -y
conda activate act_train_new
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
bash scripts/doctor.sh --gpus 1
```

上述CUDA12.6组合与本次核对的旧ACT运行环境一致；需要GPU驱动支持该wheel。Torch安装后再装本项目，其他依赖集中声明在pyproject.toml；requirements.txt自动使用requirements-validated.txt约束到本次实测的主要版本。采用图像存储模式，不依赖GPU视频解码。

| 核心依赖 | 版本/范围 | 用途 |
|---|---|---|
| Python | 3.12 | 与旧ACT一致 |
| LeRobot | 0.4.4 | 模型、处理器、数据集与trainer |
| Torch / torchvision | 2.7.1 / 0.22.1，cu126 | 张量、GPU、ResNet18 |
| Accelerate | 1.15.0 | 单节点DDP |
| datasets / pyarrow | 4.8.5 / 26.0.0 | LeRobot数据存储 |
| NumPy / MCAP | 2.2.6 / 1.6.0 | 数值与原始数据 |
| protobuf | >=6.31.1,<7 | MCAP内嵌schema解码 |
| rich / TensorBoard | >=14,<16 / >=2.20,<3 | 中文终端与曲线 |
| matplotlib / scikit-learn | >=3.8,<4 / >=1.5,<2 | 离线图与F1诊断 |

训练使用ImageNet ResNet18初始权重，首次启动需缓存可用或网络可达。可用环境变量`TORCH_HOME`指定已有兼容缓存；默认缓存放在新工程的runs/.cache/torch。评测从checkpoint恢复完整参数，不重复请求骨干权重。

## 可选：验收隔离环境

在已有经过验证的act环境上，可以用`python -m venv --system-site-packages <新目录>`建立独立overlay，包的新增仅进入overlay，不改原环境。该方式便于服务器短验收；正式可复现环境建议按上方独立conda配置建立。验收中使用的精确版本记录在acceptance.md及运行产物中。

## 路径与设备

- `ACTTRAIN_PYTHON`：bash入口使用的解释器；默认当前环境的python。
- `ACTTRAIN_ROOT`或CLI的`--root`：工程工作目录；默认源代码所在目录。
- `CUDA_VISIBLE_DEVICES`或训练入口`--devices 1,2`：限定可见物理GPU；自动评测继承完全相同的集合。
- 训练配置的gpus是进程数，batch_size_per_gpu是每卡batch，不是总batch。
- 依赖检查失败直接停止，不会自动切换CPU或更换环境。

最终验收未重复获取服务器已有依赖，而是把同为Python3.12的已有安装分发复制进新overlay；这是验收准备手段，正式环境仍可按上方独立安装。
