import numpy as np


def rmsle(y_true, y_pred, eps=1e-15):
    """
    实现的 RMSLE
    """

    return np.sqrt(np.mean((np.log1p(y_pred + eps) - np.log1p(y_true + eps)) ** 2))
