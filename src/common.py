"""Common utilities for the drug activity coursework.

The dataset contains three activity columns: IC50, CC50 and SI.
For all predictive tasks these three columns are removed from the feature matrix
so that the model does not use target leakage. SI is calculated from IC50 and
CC50, therefore using one of these columns to predict another would make the
problem unrealistically easy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["IC50, mM", "CC50, mM", "SI"]
RANDOM_STATE = 42


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = project_root() / "data" / "drug_activity.csv"
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = [c for c in TARGET_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target]
    return X, y


def make_binary_target(y: pd.Series, mode: str) -> pd.Series:
    if mode == "median":
        threshold = y.median()
    elif mode == "si_gt_8":
        threshold = 8.0
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return (y > threshold).astype(int)


def regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred),
    }


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        result["ROC_AUC"] = roc_auc_score(y_true, y_proba)
    return result
