"""本地冒烟测试用：把 pi0.5 的网络替换成"同结构、极小尺寸"的版本，在 CPU 上几秒钟跑完一步。

只改尺寸（层数/宽度），不改任何数据流、处理器、损失函数——用于验证整个工程的接线是否正确。
通过环境变量 PITRAIN_TEST_PATCH=tests.smoke_patch 注入（见 tests/smoke_test.py）。
"""

from __future__ import annotations

from lerobot.policies.pi05 import modeling_pi05 as m
from transformers import CONFIG_MAPPING

_real_get = m.get_gemma_config


def tiny_gemma_config(variant: str):
    if variant == "gemma_2b":  # VLM 文本部分：宽度必须保持 2048（与视觉投影维度绑定），只保留 1 层
        return m.GemmaConfig(width=2048, depth=1, mlp_dim=256, num_heads=8, num_kv_heads=1, head_dim=256)
    return m.GemmaConfig(width=256, depth=1, mlp_dim=256, num_heads=8, num_kv_heads=1, head_dim=256)


class _TinyMapping:
    def __getitem__(self, key):
        if key == "paligemma":

            def make():
                cfg = CONFIG_MAPPING["paligemma"]()
                v = cfg.vision_config
                v.hidden_size, v.num_hidden_layers, v.num_attention_heads = 64, 1, 2
                return cfg

            return make
        return CONFIG_MAPPING[key]


m.get_gemma_config = tiny_gemma_config
m.CONFIG_MAPPING = _TinyMapping()
