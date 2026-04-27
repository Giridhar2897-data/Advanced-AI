import numpy as np
import shap
import matplotlib.pyplot as plt
import os

os.makedirs("reports/figures", exist_ok=True)


def run_shap(model, X_train, X_test, y_test, feature_names):
    explainer = shap.TreeExplainer(model)

    sample_size = min(500, len(X_test))
    idx = np.random.choice(len(X_test), sample_size, replace=False)
    X_sample = X_test[idx]

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
    plt.title("SHAP Summary Plot — LightGBM")
    plt.tight_layout()
    plt.savefig("reports/figures/12_shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=15)
    plt.title("SHAP Feature Importance — LightGBM")
    plt.tight_layout()
    plt.savefig("reports/figures/13_shap_importance.png", bbox_inches="tight", dpi=150)
    plt.close()

    fraud_idx = np.where(y_test[idx] == 1)[0]
    if len(fraud_idx) > 0:
        example_idx = fraud_idx[0]
        plt.figure(figsize=(12, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[example_idx],
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                data=X_sample[example_idx],
                feature_names=feature_names,
            ),
            show=False,
            max_display=12,
        )
        plt.title("SHAP Waterfall — Example Fraud Transaction")
        plt.tight_layout()
        plt.savefig("reports/figures/14_shap_waterfall.png", bbox_inches="tight", dpi=150)
        plt.close()

    return shap_values, feature_names
