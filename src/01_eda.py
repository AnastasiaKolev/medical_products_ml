"""Exploratory data analysis for antiviral compound dataset."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import TARGET_COLUMNS, load_data, project_root

sns.set_theme(style="whitegrid")
ROOT = project_root()
FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = load_data()
    print("Shape:", df.shape)
    print(df.head())
    print(df.info())

    # Basic statistics and missing values.
    df.describe().T.to_csv(RES_DIR / "eda_describe.csv")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_share": df.isna().mean(),
    }).sort_values("missing_share", ascending=False)
    missing.to_csv(RES_DIR / "eda_missing.csv")

    # Target distribution: raw scale and log scale.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, TARGET_COLUMNS):
        ax.hist(df[col].dropna(), bins=40)
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_distributions_raw.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, TARGET_COLUMNS):
        ax.hist(np.log1p(df[col].dropna()), bins=40)
        ax.set_title(f"log1p({col})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_distributions_log.png", dpi=160)
    plt.close()

    # Boxplots show strong outliers in biological activity columns.
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=np.log1p(df[TARGET_COLUMNS]))
    plt.title("Target columns on log1p scale")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_boxplots_log.png", dpi=160)
    plt.close()

    # Correlation of targets and strongest descriptor correlations with targets.
    corr = df.corr(numeric_only=True)
    corr.to_csv(RES_DIR / "eda_correlation_matrix.csv")
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr.loc[TARGET_COLUMNS, TARGET_COLUMNS], annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation between IC50, CC50 and SI")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_correlation.png", dpi=160)
    plt.close()

    target_corr = corr[TARGET_COLUMNS].drop(index=TARGET_COLUMNS, errors="ignore")
    top_rows = []
    for target in TARGET_COLUMNS:
        top = target_corr[target].abs().sort_values(ascending=False).head(15)
        for feature, value in top.items():
            top_rows.append({"target": target, "feature": feature, "abs_corr": value, "corr": corr.loc[feature, target]})
    pd.DataFrame(top_rows).to_csv(RES_DIR / "eda_top_target_correlations.csv", index=False)

    # High correlation pairs among descriptors.
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    feature_corr = corr.loc[feature_cols, feature_cols].abs()
    upper = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
    high_pairs = upper.stack().sort_values(ascending=False).head(50).reset_index()
    high_pairs.columns = ["feature_1", "feature_2", "abs_corr"]
    high_pairs.to_csv(RES_DIR / "eda_highly_correlated_feature_pairs.csv", index=False)

    # Top correlation barplots.
    for target in TARGET_COLUMNS:
        top = target_corr[target].sort_values(key=lambda s: s.abs(), ascending=False).head(15).sort_values()
        plt.figure(figsize=(8, 6))
        top.plot(kind="barh")
        plt.title(f"Top descriptor correlations with {target}")
        plt.xlabel("Pearson correlation")
        plt.tight_layout()
        safe = target.replace(", mM", "").lower()
        plt.savefig(FIG_DIR / f"top_corr_{safe}.png", dpi=160)
        plt.close()

    print("EDA artifacts saved to", RES_DIR, "and", FIG_DIR)


if __name__ == "__main__":
    main()
