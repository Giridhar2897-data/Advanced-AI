from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_models(fraud_weight):
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=fraud_weight,
            random_state=42,
            eval_metric="aucpr",
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
        ),
    }


def train_all(models, X_train, y_train):
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def predict_scores(trained_models, X):
    return {name: model.predict_proba(X)[:, 1] for name, model in trained_models.items()}
