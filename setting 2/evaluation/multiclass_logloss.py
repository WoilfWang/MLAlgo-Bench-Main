import numpy as np


def multiclass_logloss(y_true, y_pred, eps=1e-15):
    """
    计算多分类对数损失 (Log Loss)

    参数:
        y_true (array-like): shape (N,)，真实类别索引 (int)
        y_pred (array-like): shape (N, M)，预测概率分布，每行归一化
        eps (float): 防止 log(0)，做数值稳定

    返回:
        float: log loss 值
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)  # 防止概率为 0 或 1

    N = y_true.shape[0]
    log_probs = -np.log(y_pred[np.arange(N), y_true])
    loss = np.sum(log_probs) / N
    return loss
