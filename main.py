import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data_loader import load_data, prepare_splits
from src.supervised import get_models as get_sup_models, train_all as train_sup, predict_scores as sup_scores
from src.unsupervised import get_models as get_unsup_models, train_all as train_unsup, predict_scores as unsup_scores
from src.ensemble import lgbm_if_blend
from src.evaluation import compute_metrics, best_threshold
from src.plots import (
    plot_class_distribution,
    plot_amount_time,
    plot_feature_distributions,
    plot_correlation,
    plot_all_pr_roc,
    plot_model_comparison,
    plot_confusion_matrices,
    plot_threshold_analysis,
    plot_cost_analysis,
    save,
)
from src.explainability import run_shap


def print_results_table(results):
    df = pd.DataFrame(results).T
    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS (Test Set)")
    print("=" * 70)
    print(df[["AUPRC", "ROC-AUC", "F1", "Precision", "Recall"]].to_string())
    print("=" * 70 + "\n")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Dataset: {len(df):,} transactions | Fraud rate: {df['Class'].mean()*100:.3f}%")

    plot_class_distribution(df)
    plot_amount_time(df)
    plot_feature_distributions(df)
    plot_correlation(df)
    print("EDA plots saved.")

    print("\nPreparing splits...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler = prepare_splits(df)
    print(f"Train: {len(y_train):,} | Val: {len(y_val):,} | Test: {len(y_test):,}")
    print(f"Train fraud: {y_train.sum()} | Val fraud: {y_val.sum()} | Test fraud: {y_test.sum()}")

    fraud_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("\nTraining supervised models...")
    sup_models = train_sup(get_sup_models(fraud_weight), X_train, y_train)
    sup_test_scores = sup_scores(sup_models, X_test)
    sup_val_scores = sup_scores(sup_models, X_val)

    print("Training unsupervised models (on normal training data only)...")
    X_train_normal = X_train[y_train == 0]
    contamination = y_train.mean()
    unsup_models = train_unsup(get_unsup_models(contamination), X_train_normal)
    unsup_test_scores = unsup_scores(unsup_models, X_test)

    ensemble_scores = lgbm_if_blend(sup_test_scores, unsup_test_scores)
    all_test_scores = {**sup_test_scores, **unsup_test_scores, "Ensemble (LightGBM+IF)": ensemble_scores}

    print("\nEvaluating all models...")
    results = {}
    for name, scores in all_test_scores.items():
        results[name] = compute_metrics(y_test, scores)
    print_results_table(results)

    print("Generating evaluation plots...")
    plot_all_pr_roc(y_test, all_test_scores)
    plot_model_comparison(results)

    preds_for_cm = {}
    for name, scores in {"LightGBM": sup_test_scores["LightGBM"],
                         "Isolation Forest": unsup_test_scores["Isolation Forest"],
                         "Ensemble": ensemble_scores}.items():
        t, _ = best_threshold(y_test, scores)
        preds_for_cm[name] = (scores >= t).astype(int)
    plot_confusion_matrices(y_test, preds_for_cm)

    key_models = {k: v for k, v in all_test_scores.items()
                  if k in ["LightGBM", "XGBoost", "Isolation Forest", "Ensemble (LightGBM+IF)"]}
    plot_threshold_analysis(y_test, key_models)
    plot_cost_analysis(y_test, key_models)

    print("\nRunning SHAP explainability...")
    run_shap(sup_models["LightGBM"], X_train, X_test, y_test, feature_cols)

    print("\nAll figures saved to reports/figures/")
    print("Done.")


if __name__ == "__main__":
    main()
