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
    
    # Load heteroscedastic CI params saved during training
    residual_std_path = Path(args.model).parent / "residual_std.joblib"
    try:
        ci_params = joblib.load(residual_std_path)
        # Support old format (plain float) and new format (dict)
        if isinstance(ci_params, dict):
            ci_alpha = ci_params["alpha"]
        else:
            ci_alpha = None  # old fixed-std format — ignore
    except:
        ci_alpha = None
    
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
    
    # Heteroscedastic confidence intervals (95%)
    # Width scales with the prediction magnitude: CI = pred ± 1.96 * alpha * pred
    df_out = df.copy()
    df_out["predicted_revenue"] = preds
    if ci_alpha is not None:
        z_score = 1.96
        half_width = z_score * ci_alpha * np.abs(preds)   # proportional to prediction
        df_out["prediction_lower_95"] = preds - half_width
        df_out["prediction_upper_95"] = preds + half_width
        df_out["prediction_ci_width"]  = 2 * half_width
        print("✅ Predictions with heteroscedastic 95% confidence intervals")
    else:
        print("⚠️ No confidence interval parameters found — point predictions only")

    if args.out:
        df_out.to_csv(args.out, index=False)
        print(f"✍️ Wrote predictions to {args.out}")
    else:
        print(df_out.head(10))


if __name__ == "__main__":
    main()
