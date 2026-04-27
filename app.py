import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st
from src.model_store import load_models
from src.ensemble import lgbm_if_blend
from src.unsupervised import _normalize_scores

SUPERVISED_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
UNSUPERVISED_NAMES = ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]
MODELS_DIR = "models"


@st.cache_resource
def load_all_models():
    return load_models(SUPERVISED_NAMES, UNSUPERVISED_NAMES)


def models_ready():
    return os.path.exists(os.path.join(MODELS_DIR, "scaler.joblib"))


def engineer_features(df_raw):
    df = df_raw.copy()
    if "log_amount" not in df.columns:
        df["log_amount"] = np.log1p(df["Amount"])
    if "time_hour" not in df.columns:
        df["time_hour"] = (df["Time"] / 3600) % 24
    return df


def run_inference(df_raw, sup_models, unsup_models, scaler, feature_cols):
    df = engineer_features(df_raw)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return None

    X = scaler.transform(df[feature_cols])

    sup_scores = {name: model.predict_proba(X)[:, 1] for name, model in sup_models.items()}
    unsup_scores = {}
    for name, model in unsup_models.items():
        raw = model.score_samples(X)
        unsup_scores[name] = _normalize_scores(raw)

    ensemble = lgbm_if_blend(sup_scores, unsup_scores)

    all_scores = {**sup_scores, **unsup_scores, "Ensemble (LightGBM+IF)": ensemble}
    return all_scores


def score_color(score):
    if score >= 0.7:
        return "🔴"
    if score >= 0.4:
        return "🟡"
    return "🟢"


# ── UI ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")
st.title("🔍 Credit Card Fraud Detection")

if not models_ready():
    st.error(
        "No trained models found in `models/`. "
        "Run `python main.py` first to train and save the models."
    )
    st.stop()

sup_models, unsup_models, scaler, feature_cols = load_all_models()

threshold = st.sidebar.slider("Decision threshold", 0.1, 0.9, 0.5, 0.01)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Legend**  \n🟢 Low risk  \n🟡 Medium risk  \n🔴 High risk"
)

tab_upload, tab_manual = st.tabs(["Upload CSV", "Manual Input"])


# ── Tab 1: CSV upload ────────────────────────────────────────────────────────

with tab_upload:
    st.markdown(
        "Upload a CSV with columns `Time`, `Amount`, `V1`–`V28` "
        "(same format as `creditcard.csv`, `Class` column optional)."
    )
    file = st.file_uploader("Choose a CSV file", type="csv")

    if file:
        df_raw = pd.read_csv(file)
        st.write(f"**{len(df_raw):,} transactions loaded.**")

        with st.spinner("Running inference..."):
            all_scores = run_inference(df_raw, sup_models, unsup_models, scaler, feature_cols)

        if all_scores is not None:
            ensemble = all_scores["Ensemble (LightGBM+IF)"]
            predictions = (ensemble >= threshold).astype(int)

            n_fraud = predictions.sum()
            st.metric("Predicted fraudulent", f"{n_fraud} / {len(predictions)}")

            results_df = df_raw.copy()
            for name, scores in all_scores.items():
                results_df[name] = scores.round(4)
            results_df["Prediction"] = np.where(predictions == 1, "FRAUD", "Legitimate")

            st.dataframe(
                results_df[
                    ["Prediction", "Ensemble (LightGBM+IF)", "LightGBM", "XGBoost",
                     "Random Forest", "Logistic Regression",
                     "Isolation Forest", "One-Class SVM", "Local Outlier Factor"]
                ],
                use_container_width=True,
            )

            csv_out = results_df.to_csv(index=False).encode()
            st.download_button("Download results CSV", csv_out, "fraud_predictions.csv", "text/csv")


# ── Tab 2: Manual single transaction ────────────────────────────────────────

with tab_manual:
    st.markdown("Enter a single transaction's raw values.")

    col1, col2 = st.columns(2)
    with col1:
        time_val = st.number_input("Time (seconds)", value=0.0, step=1.0)
        amount_val = st.number_input("Amount ($)", value=0.0, min_value=0.0, step=0.01)

    st.markdown("**PCA features V1 – V28**")
    v_cols = st.columns(7)
    v_vals = {}
    for i in range(1, 29):
        col_idx = (i - 1) % 7
        with v_cols[col_idx]:
            v_vals[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v{i}", label_visibility="visible")

    if st.button("Predict", type="primary"):
        row = {"Time": time_val, "Amount": amount_val, **v_vals}
        df_single = pd.DataFrame([row])

        with st.spinner("Running inference..."):
            all_scores = run_inference(df_single, sup_models, unsup_models, scaler, feature_cols)

        if all_scores is not None:
            ensemble_score = float(all_scores["Ensemble (LightGBM+IF)"][0])
            is_fraud = ensemble_score >= threshold

            st.markdown("---")
            verdict = "🔴 FRAUD" if is_fraud else "🟢 Legitimate"
            st.subheader(f"Verdict: {verdict}")
            st.metric("Ensemble score", f"{ensemble_score:.4f}", delta=None)

            st.markdown("**Per-model scores**")
            rows = []
            for name, scores in all_scores.items():
                s = float(scores[0])
                rows.append({"Model": name, "Score": round(s, 4), "Risk": score_color(s)})
            score_df = pd.DataFrame(rows).set_index("Model")
            st.dataframe(score_df, use_container_width=True)
