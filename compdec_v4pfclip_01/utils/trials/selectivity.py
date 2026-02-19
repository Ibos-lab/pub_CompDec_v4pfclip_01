"""

Author: Camila Losada
"""

import numpy as np
from scipy import stats
from sklearn import metrics
from typing import Tuple


def scale_signal(x, out_range=(-1, 1)):
    if np.sum(x > 1) > 0:
        return
    domain = 0, 1
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def compute_roc_auc(group1, group2, sacale=True):
    roc_score = []
    p = []
    for n_win in np.arange(group1.shape[1]):
        g1 = group1[:, n_win]
        g2 = group2[:, n_win]
        p.append(stats.ttest_ind(g1, g2)[1])
        thresholds = np.unique(np.concatenate([g1, g2]))
        y_g1, y_g2 = np.ones(len(g1)), np.zeros(len(g2))
        score = 0.5
        fpr, tpr = [], []
        for threshold in thresholds:
            g1_y_pred, g2_y_pred = np.zeros(len(g1)), np.zeros(len(g2))
            g1_mask, g2_mask = g1 >= threshold, g2 >= threshold
            g1_y_pred[g1_mask], g2_y_pred[g2_mask] = 1, 1
            tp = sum(np.logical_and(y_g1 == 1, g1_y_pred == 1))
            fn = sum(np.logical_and(y_g1 == 1, g1_y_pred == 0))
            tpr.append(tp / (tp + fn))
            fp = sum(np.logical_and(y_g2 == 0, g2_y_pred == 1))
            tn = sum(np.logical_and(y_g2 == 0, g2_y_pred == 0))
            fpr.append(fp / (fp + tn))
        if len(fpr) > 1:
            fpr, tpr = np.array(fpr), np.array(tpr)
            score = metrics.auc(fpr[fpr.argsort()], tpr[fpr.argsort()])
        roc_score.append(score)
    roc_score = np.array(roc_score)
    if sacale:
        roc_score = scale_signal(np.round(roc_score, 2), out_range=[-1, 1])
    return roc_score, np.array(p)


def find_latency(
    p_value: np.ndarray, win: int, step: int = 1, p_treshold: float = 0.01
) -> np.ndarray:
    sig = np.full(p_value.shape[0], False)
    # sig[p_value < 0.01] = True
    for i_step in np.arange(0, sig.shape[0], step):
        sig[i_step] = np.where(
            np.all(p_value[i_step : i_step + win] < p_treshold), True, False
        )
    latency = np.where(sig)[0]

    if len(latency) != 0:
        endl = np.where(~sig[latency[0] :])[0]
        endl = endl[0] if len(endl) != 0 else -1
        return latency[0], endl + latency[0] + win
    else:
        return np.nan, np.nan


def get_auc_selectivity(
    sp_1, sp_2, win, scores=False, sacale=True
) -> Tuple[float, np.ndarray, np.ndarray]:
    nanarray = np.array([np.nan])
    if np.logical_or(sp_1.ndim < 2, sp_2.ndim < 2):
        return np.nan, nanarray, nanarray
    if np.logical_or(sp_1.shape[0] < 2, sp_2.shape[0] < 2):
        return np.nan, nanarray, nanarray
    roc_score, p_value = compute_roc_auc(sp_1, sp_2, sacale=sacale)
    lat, _ = find_latency(p_value, win=win, step=1)
    if np.isnan(lat):
        roc_score = roc_score if scores else nanarray
        return lat, roc_score, p_value
    roc_score = roc_score if scores else np.array(roc_score[lat])
    return lat, roc_score, p_value
