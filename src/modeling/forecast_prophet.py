"""
Time-series revenue forecasting using Facebook Prophet.

Prophet models trend + seasonality + holidays automatically.
This script aggregates daily/quarterly sales into a time-series
and forecasts the next N periods per brand.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from prophet import Prophet


def build_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate synthetic data into a quarterly time-series.
    Prophet expects columns: ds (date), y (target value).
    """
    # Map quarter number -> a representative date so Prophet can work with it
    quarter_to_month = {1: "01", 2: "04", 3: "07", 4: "10"}
    df = df.copy()
    df["year"] = 2023  # Base year for synthetic data
    df["month"] = df["quarter"].map(quarter_to_month)
    df["ds"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"] + "-01")
    return df


def forecast_total(df: pd.DataFrame, periods: int, outdir: Path):
    """Forecast total revenue across all brands for next N quarters."""
    ts = build_time_series(df)
    agg = ts.groupby("ds")["revenue"].sum().reset_index()
    agg.columns = ["ds", "y"]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
    )
    model.fit(agg)

    # Forecast into the future (quarterly = ~90 days each)
    future = model.make_future_dataframe(periods=periods, freq="QS")
    forecast = model.predict(future)

    # Save forecast CSV
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(
        outdir / "prophet_forecast_total.csv", index=False
    )

    # Plot forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                    alpha=0.2, color="steelblue", label="95% Confidence Band")
    ax.plot(forecast["ds"], forecast["yhat"], color="steelblue", linewidth=2.5, label="Forecast")
    ax.scatter(agg["ds"], agg["y"], color="black", zorder=5, s=60, label="Actual")
    ax.set_title("Total Revenue Forecast (All Brands) - Prophet", fontsize=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "prophet_forecast_total.png", dpi=150)
    plt.close(fig)
    print(f"  → Total forecast saved")
    return forecast


def forecast_by_brand(df: pd.DataFrame, periods: int, outdir: Path):
    """Forecast revenue separately for each brand."""
    ts = build_time_series(df)
    brands = df["brand"].unique()
    brand_results = {}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, brand in enumerate(sorted(brands)):
        brand_df = ts[ts["brand"] == brand].groupby("ds")["revenue"].sum().reset_index()
        brand_df.columns = ["ds", "y"]

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
        )
        model.fit(brand_df)

        future = model.make_future_dataframe(periods=periods, freq="QS")
        forecast = model.predict(future)
        brand_results[brand] = forecast

        ax = axes[idx]
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                        alpha=0.2, color="coral", label="95% Band")
        ax.plot(forecast["ds"], forecast["yhat"], color="coral", linewidth=2.5, label="Forecast")
        ax.scatter(brand_df["ds"], brand_df["y"], color="black", zorder=5, s=50, label="Actual")
        ax.set_title(f"{brand} Revenue Forecast", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Revenue ($)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Revenue Forecast by Brand - Prophet", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(outdir / "prophet_forecast_by_brand.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save all brand forecasts as one CSV
    frames = []
    for brand, fc in brand_results.items():
        fc_out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        fc_out["brand"] = brand
        frames.append(fc_out)
    pd.concat(frames).to_csv(outdir / "prophet_forecast_by_brand.csv", index=False)
    print(f"  → Per-brand forecasts saved")
    return brand_results


def print_forecast_summary(forecast: pd.DataFrame, periods: int):
    """Print a human-readable forecast summary."""
    future_rows = forecast.tail(periods)
    print(f"\n📅 Next {periods} Quarter Forecast (Total Revenue):")
    print(f"{'Quarter':<12} {'Expected ($)':>15} {'Low ($)':>15} {'High ($)':>15}")
    print("-" * 60)
    for i, (_, row) in enumerate(future_rows.iterrows(), 1):
        print(f"  Q{i} {str(row['ds'])[:10]:<8}  {row['yhat']:>14,.0f}  "
              f"{row['yhat_lower']:>14,.0f}  {row['yhat_upper']:>14,.0f}")


def main():
    p = argparse.ArgumentParser(description="Prophet time-series sales forecasting")
    p.add_argument("--data", required=True, help="Path to sales CSV")
    p.add_argument("--outdir", default="artifacts", help="Output directory")
    p.add_argument("--periods", type=int, default=4, help="Quarters to forecast ahead")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"📊 Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"   {len(df):,} records loaded across {df['brand'].nunique()} brands\n")

    print("🔮 Forecasting total revenue...")
    total_fc = forecast_total(df, args.periods, outdir)

    print("🏢 Forecasting by brand...")
    brand_fc = forecast_by_brand(df, args.periods, outdir)

    print_forecast_summary(total_fc, args.periods)

    print(f"\n✅ All Prophet outputs saved to: {outdir}/")
    print(f"   • prophet_forecast_total.csv")
    print(f"   • prophet_forecast_total.png")
    print(f"   • prophet_forecast_by_brand.csv")
    print(f"   • prophet_forecast_by_brand.png")


if __name__ == "__main__":
    main()
