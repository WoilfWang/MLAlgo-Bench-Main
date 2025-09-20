import numpy as np


def mae(y_true, y_pred):
    """
    计算 MAE
    """
    return np.mean(np.abs(y_true - y_pred))
