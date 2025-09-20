import numpy as np


def smape(y_true, y_pred):
    """
    计算 Symmetric Mean Absolute Percentage Error (SMAPE)
    """

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)

    # 定义 smape=0 当 y_true=y_pred=0
    mask = (y_true == 0) & (y_pred == 0)
    smape_values = np.zeros_like(y_true, dtype=float)
    smape_values[~mask] = diff[~mask] / denominator[~mask]

    return 100.0 * np.mean(smape_values)
