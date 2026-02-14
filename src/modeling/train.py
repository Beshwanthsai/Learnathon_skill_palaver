"""Train a Random Forest regressor on sales data and save model and metrics."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def prepare_features(df: pd.DataFrame):
    df2 = df.copy()
    # simple encoding
    df2 = pd.get_dummies(df2, columns=["brand", "os", "quarter"], drop_first=True)
    # target: revenue (or sales_volume) - choose revenue
    X = df2.drop(["sales_volume", "revenue"], axis=1, errors="ignore")
    # numeric fill and ensure all dtypes are numeric
    X = X.fillna(0).astype("float64")
    y = df2["revenue"].fillna(0).astype(float)
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    model = RandomForestRegressor(n_estimators=50, random_state=args.seed, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    joblib.dump(model, outdir / "model.joblib")
    pd.DataFrame({"mse": [mse], "r2": [r2]}).to_csv(outdir / "metrics.csv", index=False)
    print(f"Saved model and metrics to {outdir} (mse={mse:.2f}, r2={r2:.3f})")


if __name__ == "__main__":
    main()
