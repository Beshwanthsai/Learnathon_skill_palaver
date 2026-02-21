"""Train ensemble models (RF + XGBoost) on sales data with cross-validation."""
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
    print("⚠️ XGBoost not installed. Run: pip install xgboost")


def prepare_features(df: pd.DataFrame):
    df2 = df.copy()
    
    # Add temporal features (for time-series capability)
    df2['quarter_sin'] = np.sin(2 * np.pi * df2['quarter'] / 4)
    df2['quarter_cos'] = np.cos(2 * np.pi * df2['quarter'] / 4)
    
    # One-hot encoding
    df2 = pd.get_dummies(df2, columns=["brand", "os", "quarter"], drop_first=True)
    
    # target: revenue
    X = df2.drop(["sales_volume", "revenue"], axis=1, errors="ignore")
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

    # Sort by quarter so TimeSeriesSplit respects temporal order
    df_sorted = pd.read_csv(args.data).sort_values("quarter").reset_index(drop=True)
    X, y = prepare_features(df_sorted)

    # 5-Fold Time-Series Cross-Validation (respects temporal order)
    print("🔄 Running 5-fold TimeSeriesSplit cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)

    # Base model for CV
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15,
                                     random_state=args.seed, n_jobs=-1)

    # Cross-validation scores
    cv_scores = cross_validate(rf_model, X, y, cv=tscv,
                               scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
                               return_train_score=True)
    
    cv_r2 = cv_scores['test_r2'].mean()
    cv_mse = -cv_scores['test_neg_mean_squared_error'].mean()
    cv_mae = -cv_scores['test_neg_mean_absolute_error'].mean()
    
    print(f"✅ Cross-Validation Results (5-fold):")
    print(f"   R² (Mean ± Std): {cv_scores['test_r2'].mean():.4f} ± {cv_scores['test_r2'].std():.4f}")
    print(f"   MSE: {cv_mse:.2f}")
    print(f"   MAE: {cv_mae:.2f}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    # Build ensemble: Random Forest + XGBoost (if available)
    models = [('rf', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=args.seed, n_jobs=-1))]
    
    if HAS_XGBOOST:
        print("🚀 XGBoost available - using ensemble!")
        models.append(('xgb', XGBRegressor(n_estimators=100, max_depth=10, 
                                          learning_rate=0.1, random_state=args.seed, verbosity=0)))
        ensemble = VotingRegressor(estimators=models)
    else:
        ensemble = models[0][1]
        print("⚠️ XGBoost not available - using Random Forest only")

    # Train ensemble
    ensemble.fit(X_train, y_train)
    
    # Predictions
    preds = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    # Heteroscedastic confidence intervals:
    # Error grows proportionally with prediction magnitude.
    # alpha = std(residual / prediction) — the relative error rate.
    residuals = y_test - preds
    relative_errors = residuals / np.maximum(preds, 1)          # avoid div-by-zero
    alpha = float(np.std(relative_errors))                      # relative std
    global_std = float(np.std(residuals))                       # kept for reference

    print(f"\n📊 Test Set Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MSE: {mse:.2f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   Relative error α (heteroscedastic): {alpha:.4f}")

    # Save models
    joblib.dump(ensemble, outdir / "model.joblib")
    # Save both for confidence interval computation
    joblib.dump({"alpha": alpha, "global_std": global_std}, outdir / "residual_std.joblib")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        "metric": ["r2", "mse", "mae", "cv_r2", "cv_mse", "cv_mae", "ci_alpha"],
        "value": [r2, mse, mae, cv_r2, cv_mse, cv_mae, alpha]
    })
    metrics_df.to_csv(outdir / "metrics.csv", index=False)
    
    print(f"\n✨ Saved ensemble model, metrics, and residuals to {outdir}")


if __name__ == "__main__":
    main()
