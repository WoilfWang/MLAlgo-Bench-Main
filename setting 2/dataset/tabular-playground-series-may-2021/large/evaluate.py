import pandas as pd
from metrics.multiclass_logloss import multiclass_logloss


def evaluate(label_df, pred_df, metrics_func=multiclass_logloss):
    """
    对齐 id 后计算多分类 log loss

    参数:
        label_df (pd.DataFrame): 包含列 ["id", "target"]，target 为真实类别 (str)
        pred_df (pd.DataFrame): 包含列 ["id", class_0, class_1, ..., class_M-1]，每列是预测概率

    返回:
        float: log loss 值
    """
    merged = pd.merge(label_df, pred_df, on="id", how="inner")

    class_names = [c for c in merged.columns if c not in ["id", "target"]]
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    y_true = merged["target"].map(class_to_idx).values
    y_pred = merged[class_names].values

    return metrics_func(y_true, y_pred)
