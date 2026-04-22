import numpy as np


def blend(supervised_scores, unsupervised_scores, sup_weight=0.7):
    sup_avg = np.mean(list(supervised_scores.values()), axis=0)
    unsup_avg = np.mean(list(unsupervised_scores.values()), axis=0)
    return sup_weight * sup_avg + (1 - sup_weight) * unsup_avg


def lgbm_if_blend(supervised_scores, unsupervised_scores, sup_weight=0.7):
    lgbm = supervised_scores["LightGBM"]
    iso = unsupervised_scores["Isolation Forest"]
    return sup_weight * lgbm + (1 - sup_weight) * iso
