"""Train RF + XGBoost ensemble with time-series cross-validation."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_validate, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def prepare_features(df: pd.DataFrame):
    df2 = df.copy()

    # Sinusoidal encoding keeps Q4 and Q1 adjacent on a circle
    df2['quarter_sin'] = np.sin(2 * np.pi * df2['quarter'] / 4)
    df2['quarter_cos'] = np.cos(2 * np.pi * df2['quarter'] / 4)

    # Convert text columns (brand, os, quarter) to numbers
    df2 = pd.get_dummies(df2, columns=["brand", "os", "quarter"], drop_first=True)

    # Drop target and leakage columns — model must not see the answer
    X = df2.drop(["sales_volume", "revenue"], axis=1, errors="ignore")
    X = X.fillna(0).astype("float64")
    y = df2["revenue"].fillna(0).astype(float) if "revenue" in df2.columns else pd.Series(0.0, index=df2.index)
    return X, y


# Black dots = historical quarterly actuals
# Blue line = Prophet's fitted + forecasted expected revenue
# Light-blue band = 95% confidence interval (widens into the future because uncertainty grows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Sort by quarter so folds always train on past, test on future
    df_sorted = pd.read_csv(args.data).sort_values("quarter").reset_index(drop=True)
    X, y = prepare_features(df_sorted)

    # 5-fold time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    rf_cv = RandomForestRegressor(n_estimators=100, max_depth=15,
                                  random_state=args.seed, n_jobs=-1)
    cv_scores = cross_validate(rf_cv, X, y, cv=tscv,
                               scoring=['r2', 'neg_mean_squared_error',
                                        'neg_mean_absolute_error'])
    cv_r2  = cv_scores['test_r2'].mean()
    cv_mse = -cv_scores['test_neg_mean_squared_error'].mean()
    cv_mae = -cv_scores['test_neg_mean_absolute_error'].mean()
    print(f"CV  R²={cv_r2:.4f}  MSE={cv_mse:.0f}  MAE={cv_mae:.0f}")

    # 80/20 train-test split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed)

    # Build model — add XGBoost to ensemble if installed
    estimators = [('rf', RandomForestRegressor(n_estimators=100, max_depth=15,
                                               random_state=args.seed, n_jobs=-1))]
    if HAS_XGBOOST:
        estimators.append(('xgb', XGBRegressor(n_estimators=100, max_depth=10,
                                               learning_rate=0.1,
                                               random_state=args.seed, verbosity=0)))
        model = VotingRegressor(estimators=estimators)
    else:
        model = estimators[0][1]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2  = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f"Test R²={r2:.4f}  MSE={mse:.0f}  MAE={mae:.0f}")

    # Heteroscedastic CI: error scales proportionally with prediction size
    # alpha = relative std of residuals → used in predict.py as CI width factor
    residuals = y_test - preds
    alpha = float(np.std(residuals / np.maximum(preds, 1)))

    # Save artifacts
    joblib.dump(model, outdir / "model.joblib")
    joblib.dump({"alpha": alpha, "global_std": float(np.std(residuals))},
                outdir / "residual_std.joblib")

    pd.DataFrame({
        "metric": ["r2", "mse", "mae", "cv_r2", "cv_mse", "cv_mae", "ci_alpha"],
        "value":  [r2, mse, mae, cv_r2, cv_mse, cv_mae, alpha],
    }).to_csv(outdir / "metrics.csv", index=False)

    print(f"Saved model, metrics, and CI params to {outdir}/")


if __name__ == "__main__":
    main()
