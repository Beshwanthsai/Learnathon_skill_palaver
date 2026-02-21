"""
Sales Forecasting Dashboard — clean minimal UI.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.modeling.train import prepare_features

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (LeetCode-inspired: light, clean, no decorations) ──────────────
st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] { background: #ffffff; }
[data-testid="stSidebar"]          { background: #f7f8fa; border-right: 1px solid #e1e4e8; }

/* Typography */
h1 { font-size: 1.4rem !important; font-weight: 700 !important; color: #1a1a1a !important; }
h2 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1a1a1a !important;
     border-bottom: 1px solid #e1e4e8; padding-bottom: .35rem; margin-top: 1.4rem !important; }
h3 { font-size: .95rem !important; font-weight: 600 !important; color: #444 !important; }
p, li, label, .stMarkdown { font-size: .875rem !important; color: #444 !important; }

/* Sidebar nav text */
[data-testid="stSidebar"] label { font-size: .85rem !important; color: #333 !important; }

/* Metric cards */
[data-testid="stMetricValue"]  { font-size: 1.5rem !important; font-weight: 700; color: #1a1a1a; }
[data-testid="stMetricLabel"]  { font-size: .78rem !important; color: #666; text-transform: uppercase; letter-spacing: .04em; }
[data-testid="stMetricDelta"]  { font-size: .78rem !important; }

/* Remove default Streamlit header colour blocks */
[data-testid="stHeader"] { background: transparent; }

/* Table */
[data-testid="stDataFrame"] table { font-size: .8rem !important; }

/* Thin horizontal rule */
hr { border: none; border-top: 1px solid #e1e4e8; margin: 1rem 0; }

/* Hide Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data / model loaders ──────────────────────────────────────────────────────
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def build_predictions(_model, _X, df):
    preds = _model.predict(_X)
    out = df.copy()
    out["predicted_revenue"] = preds
    out["error"]     = out["revenue"] - preds
    out["error_pct"] = (out["error"] / out["revenue"]) * 100
    return out

@st.cache_data
def load_feature_impact():
    try:
        return pd.read_csv("artifacts/feature_impact.csv")
    except:
        return pd.DataFrame()

@st.cache_data
def load_metrics():
    try:
        m = pd.read_csv("artifacts/metrics.csv").set_index("metric")["value"]
        return m
    except:
        return None


# ── Chart style helpers ───────────────────────────────────────────────────────
ACCENT  = "#2563eb"   # blue accent (single colour throughout)
GRAY    = "#6b7280"
LIGHT   = "#f3f4f6"

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d1d5db")
    ax.tick_params(colors="#6b7280", labelsize=8)
    ax.set_xlabel(xlabel, fontsize=8, color="#6b7280")
    ax.set_ylabel(ylabel, fontsize=8, color="#6b7280")
    if title:
        ax.set_title(title, fontsize=9, fontweight="600", color="#1a1a1a", pad=8)
    ax.grid(axis="y", color="#f3f4f6", linewidth=0.8)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown("### Sales Forecasting")
        st.markdown("<hr style='margin:.4rem 0 .8rem'>", unsafe_allow_html=True)
        page = st.radio(
            "Navigate to",
            ["Summary", "Dataset", "Predictions", "Feature Impact", "Forecast", "Model Details"],
            label_visibility="collapsed",
        )
        st.markdown("<hr style='margin:.8rem 0 .4rem'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:.75rem;color:#9ca3af;'>Mobile Phone Sales · Random Forest + Prophet</p>",
            unsafe_allow_html=True,
        )

    # ── Load shared resources ─────────────────────────────────────────────────
    try:
        df      = load_data("data/synthetic_sales.csv")
        model   = load_model("artifacts/model.joblib")
        X, y    = prepare_features(df)
        df_pred = build_predictions(model, X, df)
        fi      = load_feature_impact()
        m       = load_metrics()
    except Exception as e:
        st.error(f"Failed to load data or model: {e}")
        st.code(
            "python3 data/generate_synthetic.py --output data/synthetic_sales.csv --n 5000\n"
            "python3 src/modeling/train.py --data data/synthetic_sales.csv --outdir artifacts",
            language="bash",
        )
        return

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE: Summary
    # ═════════════════════════════════════════════════════════════════════════
    if page == "Summary":
        st.markdown("## Summary")
        st.markdown(
            "Revenue forecasting for mobile phone sales across four brands. "
            "The model predicts quarterly revenue from product specifications and market signals."
        )

        # ── KPI row ──────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records",  f"{len(df):,}")
        c2.metric("Brands",         df["brand"].nunique())
        c3.metric("Avg Revenue",    f"${df['revenue'].mean():,.0f}")
        c4.metric("Avg Price",      f"${df['price'].mean():,.0f}")

        st.markdown("<hr>", unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        # ── Model metrics ─────────────────────────────────────────────────────
        with col_left:
            st.markdown("## Model Performance")
            if m is not None:
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("R²",         f"{m['r2']:.4f}")
                mc2.metric("CV R² (5-fold)", f"{m['cv_r2']:.4f}")
                mc3.metric("MAE",        f"${m['mae']:,.0f}")
            else:
                st.info("Run training to generate metrics.")

            st.markdown("## Dataset Info")
            info = pd.DataFrame({
                "Field":  ["Brands", "Operating Systems", "Quarters", "Price range"],
                "Value": [
                    ", ".join(sorted(df["brand"].unique())),
                    ", ".join(sorted(df["os"].unique())),
                    "Q1 – Q4",
                    f"${df['price'].min():.0f} – ${df['price'].max():.0f}",
                ],
            })
            st.dataframe(info, hide_index=True, use_container_width=True)

        # ── Top features mini chart ───────────────────────────────────────────
        with col_right:
            st.markdown("## Top 5 Features")
            if not fi.empty:
                top5 = fi.head(5).sort_values("mean_abs_shap")
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.barh(top5["feature"], top5["mean_abs_shap"], color=ACCENT, height=0.55)
                _style_ax(ax, xlabel="Mean |SHAP|")
                fig.tight_layout(pad=1.2)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("Run feature impact analysis to populate this chart.")

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE: Dataset
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "Dataset":
        st.markdown("## Dataset")
        st.markdown(f"{len(df):,} records · {df.shape[1]} columns · 4 brands · quarters 1–4")

        st.markdown("## Preview")
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("## Revenue Distribution")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(df["revenue"], bins=35, color=ACCENT, edgecolor="white", linewidth=0.4)
            _style_ax(ax, xlabel="Revenue ($)", ylabel="Count")
            fig.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c2:
            st.markdown("## Revenue by Brand")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            brand_rev = df.groupby("brand")["revenue"].sum().sort_values()
            ax.barh(brand_rev.index, brand_rev.values, color=ACCENT, height=0.55)
            _style_ax(ax, xlabel="Total Revenue ($)")
            fig.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("## Price vs Sales Volume")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.scatter(df["price"], df["sales_volume"], alpha=0.3, s=12, color=ACCENT)
            _style_ax(ax, xlabel="Price ($)", ylabel="Sales Volume")
            fig.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c2:
            st.markdown("## Battery vs Revenue")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.scatter(df["battery"], df["revenue"], alpha=0.3, s=12, color=GRAY)
            _style_ax(ax, xlabel="Battery (mAh)", ylabel="Revenue ($)")
            fig.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE: Predictions
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "Predictions":
        st.markdown("## Predictions")
        st.markdown("Actual revenue vs model predictions across the full dataset.")

        # Scatter: actual vs predicted
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(df_pred["revenue"], df_pred["predicted_revenue"],
                   alpha=0.35, s=12, color=ACCENT, label="Samples")
        lo = df_pred["revenue"].min(); hi = df_pred["revenue"].max()
        ax.plot([lo, hi], [lo, hi], color="#ef4444", linewidth=1.2,
                linestyle="--", label="Perfect prediction")
        _style_ax(ax, xlabel="Actual Revenue ($)", ylabel="Predicted Revenue ($)")
        ax.legend(fontsize=8, framealpha=0)
        fig.tight_layout(pad=1.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Filters
        col_b, col_o = st.columns(2)
        brands = col_b.multiselect("Brand", sorted(df["brand"].unique()),
                                   default=sorted(df["brand"].unique()))
        oses   = col_o.multiselect("OS",    sorted(df["os"].unique()),
                                   default=sorted(df["os"].unique()))

        filt = df_pred[df_pred["brand"].isin(brands) & df_pred["os"].isin(oses)]

        if filt.empty:
            st.warning("No rows match the selected filters.")
        else:
            st.markdown(f"## Records  ({len(filt):,})")
            st.dataframe(
                filt[["brand", "os", "price", "ram", "storage",
                       "battery", "revenue", "predicted_revenue", "error_pct"]]
                .rename(columns={"error_pct": "error %"})
                .head(30),
                use_container_width=True, hide_index=True,
            )

            st.markdown("## Prediction Error Distribution")
            fig, ax = plt.subplots(figsize=(8, 3.5))
            ax.hist(filt["error_pct"].clip(-50, 50), bins=35,
                    color=ACCENT, edgecolor="white", linewidth=0.4)
            ax.axvline(0, color="#ef4444", linewidth=1.2, linestyle="--")
            _style_ax(ax, xlabel="Error (%)", ylabel="Count")
            fig.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE: Feature Impact
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "Feature Impact":
        st.markdown("## Feature Impact")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) measures how much each feature "
            "shifts the model's output away from the mean prediction. "
            "Values shown are mean absolute SHAP across 1,000 sampled rows."
        )

        if fi.empty:
            st.info(
                "No feature impact data found. Run:\n\n"
                "`python3 src/analysis/feature_impact.py "
                "--data data/synthetic_sales.csv --model artifacts/model.joblib --outdir artifacts`"
            )
        else:
            top_n = st.slider("Features to show", 5, min(20, len(fi)), 10)

            c1, c2 = st.columns([1, 1])

            with c1:
                st.markdown("## Ranking Table")
                display = fi.head(top_n).copy()
                display.index = range(1, len(display) + 1)
                display.columns = ["Feature", "Mean |SHAP|"]
                display["Mean |SHAP|"] = display["Mean |SHAP|"].map("{:,.0f}".format)
                st.dataframe(display, use_container_width=True)

            with c2:
                st.markdown("## Impact Chart")
                plot_df = fi.head(top_n).sort_values("mean_abs_shap")
                fig, ax = plt.subplots(figsize=(6, top_n * 0.42 + 0.8))
                ax.barh(plot_df["feature"], plot_df["mean_abs_shap"],
                        color=ACCENT, height=0.6)
                _style_ax(ax, xlabel="Mean |SHAP|")
                fig.tight_layout(pad=1.2)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.markdown("<hr>", unsafe_allow_html=True)
            top1 = fi.iloc[0]
            st.markdown(
                f"**Dominant feature:** `{top1['feature']}` with a mean |SHAP| of "
                f"**{top1['mean_abs_shap']:,.0f}** — "
                "meaning it shifts the revenue prediction by that amount on average."
            )

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE: Forecast (Prophet)
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "Forecast":
        st.markdown("## Quarterly Revenue Forecast")
        st.markdown(
            "Uses Facebook Prophet to project total and per-brand revenue "
            "for the next four quarters. Bands represent 95% prediction intervals."
        )

        total_csv = Path("artifacts/prophet_forecast_total.csv")
        brand_csv = Path("artifacts/prophet_forecast_by_brand.csv")
        total_png = Path("artifacts/prophet_forecast_total.png")
        brand_png = Path("artifacts/prophet_forecast_by_brand.png")

        if not total_csv.exists():
            st.info(
                "Forecast files not found. Generate them with:\n\n"
                "`python3 src/modeling/forecast_prophet.py "
                "--data data/synthetic_sales.csv --periods 4`"
            )
            return

        st.markdown("## Total Revenue — All Brands")
        if total_png.exists():
            st.image(str(total_png), use_container_width=True)

        total_fc   = pd.read_csv(total_csv, parse_dates=["ds"])
        future_fc  = total_fc.tail(4)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        future_fc.columns = ["Quarter Start", "Expected ($)", "Lower ($)", "Upper ($)"]
        for col in ["Expected ($)", "Lower ($)", "Upper ($)"]:
            future_fc[col] = future_fc[col].map("${:,.0f}".format)
        st.dataframe(future_fc.reset_index(drop=True), use_container_width=True, hide_index=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("## Per-Brand Forecast")

        if brand_png.exists():
            st.image(str(brand_png), use_container_width=True)

        if brand_csv.exists():
            brand_fc  = pd.read_csv(brand_csv, parse_dates=["ds"])
            selected  = st.selectbox("Brand", sorted(brand_fc["brand"].unique()))
            bdf = brand_fc[brand_fc["brand"] == selected].tail(8)[
                ["ds", "yhat", "yhat_lower", "yhat_upper"]
            ].copy()
            bdf.columns = ["Date", "Expected ($)", "Lower ($)", "Upper ($)"]
            st.dataframe(bdf.reset_index(drop=True), use_container_width=True, hide_index=True)

        st.markdown(
            "_Shaded bands widen further into the future reflecting increasing uncertainty._"
        )

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE: Model Details
    # ═════════════════════════════════════════════════════════════════════════
    elif page == "Model Details":
        st.markdown("## Model Details")
        st.markdown(
            "Technical overview of the training pipeline, validation strategy, "
            "and design decisions."
        )

        st.markdown("## Pipeline")
        pipeline = pd.DataFrame({
            "Step":   ["Data generation", "Feature engineering", "Validation",
                       "Training", "Prediction", "Interpretability", "Forecasting"],
            "Method": [
                "Synthetic quarterly mobile phone sales (5,000 rows)",
                "One-hot encoding (brand, OS, quarter) + sinusoidal quarter features",
                "5-fold TimeSeriesSplit cross-validation",
                "Random Forest (100 trees, max depth 15); XGBoost if installed",
                "Heteroscedastic 95% confidence intervals (width scales with prediction)",
                "SHAP TreeExplainer — mean |SHAP| per feature",
                "Facebook Prophet — additive, no yearly seasonality, tight changepoint prior",
            ],
        })
        st.dataframe(pipeline, use_container_width=True, hide_index=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("## Metrics")

        if m is not None:
            metrics_display = pd.DataFrame({
                "Metric":      ["R² (test)", "CV R² (5-fold)", "MAE (test)", "CI alpha"],
                "Value":       [
                    f"{m['r2']:.4f}",
                    f"{m.get('cv_r2', float('nan')):.4f}",
                    f"${m['mae']:,.0f}",
                    f"{m.get('ci_alpha', float('nan')):.4f}",
                ],
                "Interpretation": [
                    "Fraction of revenue variance explained by the model",
                    "Average R² across 5 time-ordered folds",
                    "Average absolute dollar error per prediction",
                    "Relative error std used for confidence interval width",
                ],
            })
            st.dataframe(metrics_display, use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("R²",            f"{m['r2']:.4f}")
            c2.metric("CV R² (5-fold)", f"{m.get('cv_r2', 0):.4f}")
            c3.metric("MAE",           f"${m['mae']:,.0f}")
        else:
            st.info("Run training to generate metrics.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("## Design Decisions")
        decisions = pd.DataFrame({
            "Decision": [
                "TimeSeriesSplit instead of KFold",
                "Heteroscedastic confidence intervals",
                "Prophet with yearly_seasonality=False",
                "Sinusoidal quarter encoding",
                "revenue and sales_volume excluded from SHAP features",
            ],
            "Reason": [
                "Shuffling folds leaks future data into training — TimeSeriesSplit preserves order",
                "Fixed-width CIs over- or under-estimate uncertainty depending on prediction magnitude",
                "Only 4 distinct time points available — enabling yearly seasonality overfits",
                "Preserves cyclical distance between Q4 and Q1 which linear encoding breaks",
                "Using the target variable as a SHAP input inflates importance scores artificially",
            ],
        })
        st.dataframe(decisions, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
