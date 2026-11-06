"""图像处理：JPEG 解码 + 保持比例缩放 + 黑边补成正方形（letterbox）。

pi0.5 的视觉编码器 SigLIP 输入为 224×224。原图 640×480（4:3），直接拉伸会变形，
所以先等比缩到 224×168，再上下各补 28 行黑边——与 OpenPI 的 resize_with_pad 一致。
"""
from __future__ import annotations

import numpy as np


def decode_jpeg_rgb(data: bytes) -> np.ndarray:
    import cv2

    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("JPEG 解码失败")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def letterbox(rgb: np.ndarray, size: int) -> np.ndarray:
    import cv2

    h, w = rgb.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas
