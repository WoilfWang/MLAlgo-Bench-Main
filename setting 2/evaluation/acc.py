def acc(y_true, y_pred):
    """
    自己实现的准确率计算函数
    参数:
        y_true (list/array): 真实标签
        y_pred (list/array): 预测标签
    返回:
        float: accuracy 值
    """

    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    return correct / len(y_true)
