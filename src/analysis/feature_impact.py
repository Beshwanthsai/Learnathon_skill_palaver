"""Compute SHAP feature importances for a saved model and dataset."""
import argparse
from pathlib import Path
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_X(df: pd.DataFrame):
    df2 = df.copy()
    # Add temporal features to match training pipeline
    df2['quarter_sin'] = np.sin(2 * np.pi * df2['quarter'] / 4)
    df2['quarter_cos'] = np.cos(2 * np.pi * df2['quarter'] / 4)
    df2 = pd.get_dummies(df2, columns=["brand", "os", "quarter"], drop_first=True)
    # Drop target-related columns to prevent data leakage in SHAP
    leakage_cols = ["sales_volume", "revenue", "predicted_revenue"]
    df2 = df2.drop(columns=[c for c in leakage_cols if c in df2.columns], errors="ignore")
    return df2.fillna(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    X = prepare_X(df)

    model = joblib.load(args.model)

    # Use a smaller sample for faster SHAP computation
    sample_size = min(1000, len(X))  # Limit to 1000 samples max
    X_sample = X.sample(n=sample_size, random_state=42)

    # Use TreeExplainer for faster computation on tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # mean absolute SHAP per feature
    mean_abs = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
    mean_abs = mean_abs.sort_values("mean_abs_shap", ascending=False)
    mean_abs.to_csv(outdir / "feature_impact.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=mean_abs.head(25), x="mean_abs_shap", y="feature")
    plt.title("Top 25 feature impacts (mean |SHAP|)")
    plt.tight_layout()
    plt.savefig(outdir / "feature_impact.png")
    print(f"Wrote feature impact csv and plot to {outdir}")


if __name__ == "__main__":
    main()
