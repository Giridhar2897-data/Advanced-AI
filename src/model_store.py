import os
import joblib


MODELS_DIR = "models"


def save_models(supervised, unsupervised, scaler, feature_cols):
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model in supervised.items():
        path = _model_path("sup", name)
        joblib.dump(model, path)
        print(f"  Saved {name} → {path}")
    for name, model in unsupervised.items():
        path = _model_path("unsup", name)
        joblib.dump(model, path)
        print(f"  Saved {name} → {path}")
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.joblib"))
    print(f"  Saved scaler and feature columns → {MODELS_DIR}/")


def load_models(supervised_names, unsupervised_names):
    supervised = {name: joblib.load(_model_path("sup", name)) for name in supervised_names}
    unsupervised = {name: joblib.load(_model_path("unsup", name)) for name in unsupervised_names}
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.joblib"))
    return supervised, unsupervised, scaler, feature_cols


def _model_path(prefix, name):
    safe = name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "").replace("+", "_")
    return os.path.join(MODELS_DIR, f"{prefix}_{safe}.joblib")
