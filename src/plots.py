import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
import os

os.makedirs("reports/figures", exist_ok=True)

PALETTE = sns.color_palette("tab10")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


def save(name):
    plt.tight_layout()
    plt.savefig(f"reports/figures/{name}.png", bbox_inches="tight")
    plt.close()


def plot_class_distribution(df):
    counts = df["Class"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(["Legitimate", "Fraud"], counts.values, color=[PALETTE[0], PALETTE[3]])
    axes[0].set_title("Transaction Class Counts")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f"{v:,}", ha="center", fontsize=10)

    axes[1].pie(
        counts.values,
        labels=["Legitimate (99.83%)", "Fraud (0.17%)"],
        colors=[PALETTE[0], PALETTE[3]],
        autopct="%1.2f%%",
        startangle=90,
    )
    axes[1].set_title("Class Distribution")
    save("01_class_distribution")


def plot_amount_time(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    axes[0].hist(legit["Amount"].clip(upper=500), bins=60, alpha=0.6, label="Legitimate", color=PALETTE[0])
    axes[0].hist(fraud["Amount"].clip(upper=500), bins=60, alpha=0.8, label="Fraud", color=PALETTE[3])
    axes[0].set_title("Transaction Amount Distribution (clipped at $500)")
    axes[0].set_xlabel("Amount ($)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].scatter(legit["Time"] / 3600, legit["Amount"].clip(upper=500), alpha=0.05, s=2, label="Legitimate", color=PALETTE[0])
    axes[1].scatter(fraud["Time"] / 3600, fraud["Amount"].clip(upper=500), alpha=0.6, s=10, label="Fraud", color=PALETTE[3])
    axes[1].set_title("Amount vs Time")
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("Amount ($)")
    axes[1].legend()
    save("02_amount_time")


def plot_feature_distributions(df):
    top_features = ["V1", "V2", "V3", "V4", "V9", "V10", "V11", "V12", "V14", "V17"]
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    for i, feat in enumerate(top_features):
        axes[i].hist(legit[feat], bins=50, alpha=0.5, label="Legitimate", color=PALETTE[0], density=True)
        axes[i].hist(fraud[feat], bins=50, alpha=0.7, label="Fraud", color=PALETTE[3], density=True)
        axes[i].set_title(feat)
        axes[i].set_ylabel("Density")
        if i == 0:
            axes[i].legend()
    fig.suptitle("Feature Distributions: Fraud vs Legitimate (Top 10 Features)", fontsize=13)
    save("03_feature_distributions")


def plot_correlation(df):
    corr_with_class = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
    top_feats = corr_with_class.head(15).index.tolist()
    corr_matrix = df[top_feats + ["Class"]].corr()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.heatmap(corr_matrix, ax=axes[0], cmap="coolwarm", center=0, annot=False, linewidths=0.3)
    axes[0].set_title("Correlation Matrix (Top 15 Features + Class)")

    corr_with_class.head(15).plot(kind="barh", ax=axes[1], color=PALETTE[0])
    axes[1].set_title("|Correlation with Class| — Top 15 Features")
    axes[1].set_xlabel("|Pearson Correlation|")
    save("04_correlation")


def plot_pr_curves(y_true, scores_dict, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, scores) in enumerate(scores_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = np.trapezoid(precision[::-1], recall[::-1])
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=PALETTE[i])
    ax.axhline(y=y_true.mean(), color="black", linestyle="--", label="Random baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curves{title_suffix}")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    save(f"05_pr_curves{'_' + title_suffix.strip() if title_suffix else ''}")


def plot_roc_curves(y_true, scores_dict, title_suffix=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, scores) in enumerate(scores_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=PALETTE[i])
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves{title_suffix}")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    save(f"06_roc_curves{'_' + title_suffix.strip() if title_suffix else ''}")


def plot_all_pr_roc(y_true, all_scores):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, (name, scores) in enumerate(all_scores.items()):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = np.trapezoid(precision[::-1], recall[::-1])
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(recall, precision, label=f"{name} (AP={ap:.3f})", color=PALETTE[i])
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=PALETTE[i])

    axes[0].axhline(y=y_true.mean(), color="black", linestyle="--", label="Baseline")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curves (All Models)")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot([0, 1], [0, 1], "k--", label="Baseline")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curves (All Models)")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(True, alpha=0.3)
    save("07_all_models_pr_roc")


def plot_model_comparison(results):
    import pandas as pd
    df = pd.DataFrame(results).T
    metrics = ["AUPRC", "ROC-AUC", "F1", "Precision", "Recall"]
    df = df[metrics].astype(float)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.15
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, df[metric], width, label=metric, color=PALETTE[i])
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df.index, rotation=15, ha="right")
    ax.set_ylim([0, 1.1])
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    save("08_model_comparison")


def plot_confusion_matrices(y_true, preds_dict):
    n = len(preds_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, y_pred) in zip(axes, preds_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        ax.set_title(name)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    save("09_confusion_matrices")


def plot_threshold_analysis(y_true, scores_dict):
    thresholds = np.linspace(0.01, 0.99, 200)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (name, scores) in enumerate(scores_dict.items()):
        precisions, recalls, f1s = [], [], []
        for t in thresholds:
            y_pred = (scores >= t).astype(int)
            if y_pred.sum() == 0:
                precisions.append(0)
                recalls.append(0)
                f1s.append(0)
                continue
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        axes[0].plot(thresholds, f1s, label=name, color=PALETTE[i])
        axes[1].plot(thresholds, precisions, linestyle="-", color=PALETTE[i], label=f"{name} Precision")
        axes[1].plot(thresholds, recalls, linestyle="--", color=PALETTE[i], label=f"{name} Recall")

    axes[0].set_title("F1 Score vs Decision Threshold")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("F1 Score")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Precision & Recall vs Threshold")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Score")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)
    save("10_threshold_analysis")


def plot_cost_analysis(y_true, scores_dict, fraud_loss=500, review_cost=5, decline_cost=50):
    from src.evaluation import cost_analysis
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, scores) in enumerate(scores_dict.items()):
        thresholds, costs = cost_analysis(y_true, scores, fraud_loss, review_cost, decline_cost)
        ax.plot(thresholds, costs / 1000, label=name, color=PALETTE[i])
        best_t = thresholds[np.argmin(costs)]
        ax.axvline(x=best_t, color=PALETTE[i], linestyle=":", alpha=0.5)
    ax.set_title(f"Cost Analysis (fraud_loss=${fraud_loss}, review=${review_cost}, decline=${decline_cost})")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Total Cost (thousands $)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    save("11_cost_analysis")
