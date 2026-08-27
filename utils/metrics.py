import numpy as np
from sklearn.metrics import f1_score


def weighted_f1(y_true, y_pred):
    value = float(f1_score(y_true=y_true, y_pred=y_pred, average="weighted"))
    return value
