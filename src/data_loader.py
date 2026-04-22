import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path="data/creditcard.csv"):
    df = pd.read_csv(path)
    df = df.sort_values("Time").reset_index(drop=True)
    df["log_amount"] = np.log1p(df["Amount"])
    df["time_hour"] = (df["Time"] / 3600) % 24
    return df


def temporal_split(df):
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def get_features(df):
    drop_cols = ["Class", "Time", "Amount"]
    return [c for c in df.columns if c not in drop_cols]


def prepare_splits(df):
    train_df, val_df, test_df = temporal_split(df)
    feature_cols = get_features(df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_val = scaler.transform(val_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])

    y_train = train_df["Class"].values
    y_val = val_df["Class"].values
    y_test = test_df["Class"].values

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler
