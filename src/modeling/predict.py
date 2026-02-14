"""Load a saved model and run predictions on an input CSV or single row."""
import argparse
from pathlib import Path
import pandas as pd
import joblib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=False)
    args = p.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.input)

    # simple preprocessing similar to training - exclude target columns
    X = pd.get_dummies(df, columns=["brand", "os", "quarter"], drop_first=True).fillna(0)
    # Remove target columns if they exist
    target_cols = ["sales_volume", "revenue"]
    X = X.drop(columns=[col for col in target_cols if col in X.columns], errors='ignore')

    preds = model.predict(X)
    df_out = df.copy()
    df_out["predicted_revenue"] = preds

    if args.out:
        df_out.to_csv(args.out, index=False)
        print(f"Wrote predictions to {args.out}")
    else:
        print(df_out.head())


if __name__ == "__main__":
    main()
