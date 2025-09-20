import numpy as np


def rmse(y_true, y_pred):
    """
    实现的 RMSE (Root Mean Squared Error)
    """

    return np.sqrt(((y_true - y_pred) ** 2).mean())
