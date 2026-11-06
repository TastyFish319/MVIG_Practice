"""可选夹爪软下限；只改变推理输出，不修改训练标签或评分。"""
import numpy as np

def soft_floor(values, floor, threshold):
    if not np.isfinite([floor, threshold]).all() or not 0 <= floor < threshold <= 1:
        raise ValueError("夹爪参数需满足0<=floor<threshold<=1")
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("夹爪预测必须为有限值")
    width = threshold - floor
    return np.where(values >= threshold, values,
                    floor + width * np.exp(np.minimum((values-threshold)/width, 0)))
