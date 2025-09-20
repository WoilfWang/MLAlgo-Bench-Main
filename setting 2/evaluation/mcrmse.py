import numpy as np


def mcrmse(y_true, y_pred):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE)

    参数:
        y_true (np.ndarray): shape (n_samples, n_targets)
        y_pred (np.ndarray): shape (n_samples, n_targets)

    返回:
        float: MCRMSE 值
    """

    rmse_per_col = np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))
    return rmse_per_col.mean()
