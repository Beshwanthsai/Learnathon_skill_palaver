"""Load a saved model and run predictions with confidence intervals."""
import argparse
from pathlib import Path
import pandas as pd
import joblib
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=False)
    args = p.parse_args()

    model = joblib.load(args.model)
    
    # Load residual std for confidence intervals (if available)
    residual_std_path = Path(args.model).parent / "residual_std.joblib"
    try:
        std_residuals = joblib.load(residual_std_path)
    except:
        std_residuals = None
    
    df = pd.read_csv(args.input)

    # Add temporal features (must match training)
    df_temp = df.copy()
    if 'quarter' in df_temp.columns:
        df_temp['quarter_sin'] = np.sin(2 * np.pi * df_temp['quarter'] / 4)
        df_temp['quarter_cos'] = np.cos(2 * np.pi * df_temp['quarter'] / 4)
    
    # Preprocessing (must match training)
    X = pd.get_dummies(df_temp, columns=["brand", "os", "quarter"], drop_first=True).fillna(0)
    target_cols = ["sales_volume", "revenue"]
    X = X.drop(columns=[col for col in target_cols if col in X.columns], errors='ignore')

    # Predictions
    preds = model.predict(X)
    
    # Confidence intervals (95%)
    if std_residuals is not None:
        z_score = 1.96  # 95% confidence
        df_out = df.copy()
        df_out["predicted_revenue"] = preds
        df_out["prediction_lower_95"] = preds - (z_score * std_residuals)
        df_out["prediction_upper_95"] = preds + (z_score * std_residuals)
        df_out["prediction_std_dev"] = std_residuals
        print(f"✅ Predictions with 95% confidence intervals")
    else:
        df_out = df.copy()
        df_out["predicted_revenue"] = preds
        print(f"⚠️ No confidence intervals available")

    if args.out:
        df_out.to_csv(args.out, index=False)
        print(f"✍️ Wrote predictions to {args.out}")
    else:
        print(df_out.head(10))


if __name__ == "__main__":
    main()
