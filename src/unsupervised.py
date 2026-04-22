import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor


def _normalize_scores(raw):
    mn, mx = raw.min(), raw.max()
    if mx == mn:
        return np.zeros_like(raw)
    return 1 - (raw - mn) / (mx - mn)


def get_models(contamination):
    return {
        "Isolation Forest": IsolationForest(
            contamination=contamination, random_state=42, n_jobs=-1
        ),
        "One-Class SVM": OneClassSVM(nu=min(contamination * 2, 0.1), kernel="rbf"),
        "Local Outlier Factor": LocalOutlierFactor(
            contamination=contamination, novelty=True, n_jobs=-1
        ),
    }


def train_all(models, X_train_normal):
    ocsvm_limit = 10_000
    X_ocsvm = (
        X_train_normal[:ocsvm_limit]
        if len(X_train_normal) > ocsvm_limit
        else X_train_normal
    )
    trained = {}
    for name, model in models.items():
        if name == "One-Class SVM":
            model.fit(X_ocsvm)
        else:
            model.fit(X_train_normal)
        trained[name] = model
    return trained


def predict_scores(trained_models, X):
    scores = {}
    for name, model in trained_models.items():
        raw = model.score_samples(X)
        scores[name] = _normalize_scores(raw)
    return scores
