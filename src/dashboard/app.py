"""
Streamlit dashboard for Sales Forecasting and Feature Impact Analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import sys
import shap
import base64
from io import BytesIO

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Import from our project modules
from src.modeling.train import prepare_features

# â”€â”€ Page config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="SalesIQ AI Forecasting",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€ Dark Design System â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* â”€â”€ Tokens â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
:root {
  --bg-base:     #0d1117;
  --bg-surface:  #161b22;
  --bg-raised:   #1e252e;
  --bg-overlay:  #252d38;
  --border:      #30363d;
  --border-subtle: #21262d;
  --accent:      #4f8ef7;
  --accent-dim:  #1d3461;
  --teal:        #2dd4bf;
  --teal-dim:    #134e4a;
  --amber:       #f59e0b;
  --red:         #f87171;
  --green:       #4ade80;
  --txt-primary: #e6edf3;
  --txt-secondary:#8b949e;
  --txt-muted:   #484f58;
  --radius-sm:   5px;
  --radius-md:   8px;
  --radius-lg:   12px;
}

/* â”€â”€ Reset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
html, body, [class*="css"] {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  color: var(--txt-primary);
}

/* â”€â”€ App shell â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.stApp { background-color: var(--bg-base); }
.main .block-container {
  padding: 28px 40px 56px 40px;
  max-width: 1440px;
}

/* â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
[data-testid="stSidebar"] {
  background-color: var(--bg-surface) !important;
  border-right: 1px solid var(--border-subtle) !important;
}
[data-testid="stSidebar"] * { color: var(--txt-secondary) !important; }
[data-testid="stSidebar"] .stRadio label {
  font-size: 13px;
  font-weight: 400;
  padding: 5px 0;
  letter-spacing: 0.01em;
  transition: color .12s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: var(--txt-primary) !important; }
[data-testid="stSidebar"] hr { border-color: var(--border-subtle) !important; margin: 14px 0; }

/* â”€â”€ Page heading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.pg-title {
  font-size: 20px; font-weight: 600; color: var(--txt-primary);
  letter-spacing: -0.025em; margin: 0 0 4px;
}
.pg-sub {
  font-size: 13px; color: var(--txt-secondary); margin: 0 0 32px;
  font-weight: 400; line-height: 1.5;
}

/* â”€â”€ Section label â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.sec-label {
  font-size: 11px; font-weight: 600; color: var(--txt-muted);
  text-transform: uppercase; letter-spacing: 0.09em;
  margin: 28px 0 12px; padding-bottom: 8px;
  border-bottom: 1px solid var(--border-subtle);
}

/* â”€â”€ KPI cards â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.kpi {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 18px 20px;
}
.kpi-lbl {
  font-size: 10.5px; font-weight: 600; color: var(--txt-muted);
  text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;
}
.kpi-val {
  font-size: 26px; font-weight: 700; color: var(--txt-primary);
  letter-spacing: -0.03em; line-height: 1.1;
}
.kpi-note { font-size: 11px; color: var(--txt-secondary); margin-top: 4px; }
.kpi-note.pos { color: var(--green); }

/* â”€â”€ Feature / innovation cards â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.feat-card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: var(--radius-md);
  padding: 18px 20px;
  margin-bottom: 14px;
}
.feat-card h4 {
  font-size: 13px; font-weight: 600; color: var(--txt-primary); margin: 0 0 8px;
}
.feat-card ul {
  margin: 0; padding-left: 16px; color: var(--txt-secondary);
  font-size: 12.5px; line-height: 1.75;
}

/* â”€â”€ Notice banners â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.notice {
  border-radius: var(--radius-md); padding: 12px 16px;
  font-size: 13px; line-height: 1.6; margin: 16px 0;
}
.notice.info   { background: #0d1f3c; border: 1px solid var(--accent-dim); color: #93c5fd; }
.notice.warn   { background: #1c1200; border: 1px solid #78350f; color: #fcd34d; }
.notice.error  { background: #1c0a0a; border: 1px solid #7f1d1d; color: #fca5a5; }
.notice.success{ background: #052e16; border: 1px solid #14532d; color: #86efac; }

/* â”€â”€ Divider â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
.div { border: none; border-top: 1px solid var(--border-subtle); margin: 24px 0; }

/* â”€â”€ Streamlit metric overrides â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
[data-testid="metric-container"] {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] {
  font-size: 10.5px !important; font-weight: 600 !important;
  text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--txt-muted) !important;
}
[data-testid="stMetricValue"] {
  font-size: 22px !important; font-weight: 700 !important;
  color: var(--txt-primary) !important; letter-spacing: -0.02em;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* â”€â”€ Dataframe â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
[data-testid="stDataFrame"] iframe {
  border-radius: var(--radius-md) !important;
}

/* â”€â”€ Inputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
  background: var(--bg-raised) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--txt-primary) !important;
}

/* â”€â”€ Hide Streamlit chrome â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
#MainMenu, footer, header { visibility: hidden; }

/* ── Suggestion cards ────────────────────────────────────────────────────── */
.sugg-card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-top: 3px solid var(--green);
  border-radius: var(--radius-md);
  padding: 20px 18px;
  text-align: center;
}
.sugg-icon  { font-size: 22px; color: var(--green); margin-bottom: 6px; }
.sugg-label { font-size: 12.5px; font-weight: 600; color: var(--txt-primary); margin-bottom: 10px; }
.sugg-gain  { font-size: 22px; font-weight: 700; color: var(--green); letter-spacing: -0.03em; }
.sugg-new   { font-size: 11px; color: var(--txt-secondary); margin-top: 4px; }
</style>
"""

# â”€â”€ Chart system â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_BG      = "#161b22"
_GRID    = "#21262d"
_TXT     = "#8b949e"
_TTXT    = "#e6edf3"
_PAL     = ["#4f8ef7", "#2dd4bf", "#a78bfa", "#f59e0b", "#f87171", "#4ade80"]

import matplotlib.ticker as _mtick

def _style(ax, title="", xlabel="", ylabel="", legend=False):
    ax.set_facecolor(_BG)
    ax.figure.patch.set_facecolor(_BG)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_TXT, labelsize=9.5)
    ax.xaxis.label.set(color=_TXT, size=10)
    ax.yaxis.label.set(color=_TXT, size=10)
    if title:  ax.set_title(title, fontsize=11.5, fontweight="600", color=_TTXT, pad=12, loc="left")
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if legend: ax.legend(frameon=False, fontsize=9, labelcolor=_TXT)

def _dollar(ax, axis="x"):
    def _f(v, _):
        if abs(v) >= 1e9: return f"${v/1e9:.1f}B"
        if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
        if abs(v) >= 1e3: return f"${v/1e3:.0f}K"
        return f"${v:.0f}"
    fmt = _mtick.FuncFormatter(_f)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)

# â”€â”€ HTML helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _kpi(label, value, note="", pos=False):
    note_cls = "kpi-note pos" if pos else "kpi-note"
    n_html   = f'<div class="{note_cls}">{note}</div>' if note else ""
    return (f'<div class="kpi"><div class="kpi-lbl">{label}</div>'
            f'<div class="kpi-val">{value}</div>{n_html}</div>')

def _sec(text):
    return f'<p class="sec-label">{text}</p>'

def _notice(html, kind="info"):
    return f'<div class="notice {kind}">{html}</div>'

def _feat(title, bullets):
    items = "".join(f"<li>{b}</li>" for b in bullets)
    return (f'<div class="feat-card"><h4>{title}</h4>'
            f'<ul>{items}</ul></div>')



@st.cache_data
def load_data(data_path):
    """Load and return the sales data."""
    return pd.read_csv(data_path)


@st.cache_resource
def load_model(model_path):
    """Load and return the trained model."""
    return joblib.load(model_path)


@st.cache_data
def prepare_data(_model, _X, _y, df):
    """Prepare predictions and metrics."""
    predictions = _model.predict(_X)
    df_with_preds = df.copy()
    df_with_preds["predicted_revenue"] = predictions
    df_with_preds["prediction_error"] = df_with_preds["revenue"] - df_with_preds["predicted_revenue"]
    df_with_preds["error_pct"] = (df_with_preds["prediction_error"] / df_with_preds["revenue"]) * 100
    return df_with_preds


@st.cache_data
def get_feature_impact_simple(df):
    """Get feature impact from CSV file (already computed)."""
    try:
        feature_imp = pd.read_csv("artifacts/feature_impact.csv")
        return feature_imp
    except:
        return pd.DataFrame({
            "feature": ["price", "brand_B", "revenue"],
            "mean_abs_shap": [88000, 53000, 52000]
        })


def get_image_download_link(fig, filename="plot.png", text="Download Plot"):
    """Generate a link to download a matplotlib plot as a PNG."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'


def _load_metrics():
    """Read metrics.csv â€” handles both wide and tidy (long) formats."""
    try:
        m = pd.read_csv("artifacts/metrics.csv")
        if "metric" in m.columns and "value" in m.columns:
            return dict(zip(m["metric"], m["value"]))
        return m.iloc[0].to_dict()
    except Exception:
        return {}


def main():
    st.markdown(_CSS, unsafe_allow_html=True)

    # â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with st.sidebar:
        st.markdown("""
        <div style="padding:4px 0 20px">
          <div style="font-size:15px;font-weight:700;
                      color:#e6edf3;letter-spacing:-0.02em">SalesIQ</div>
          <div style="font-size:10.5px;color:#484f58;letter-spacing:0.06em;
                      text-transform:uppercase;margin-top:2px">
            AI Forecasting Platform
          </div>
        </div>
        <hr/>
        """, unsafe_allow_html=True)

        st.markdown(
            '<p style="font-size:10px;color:#484f58;font-weight:600;'
            'text-transform:uppercase;letter-spacing:0.09em;margin:0 0 8px">Menu</p>',
            unsafe_allow_html=True,
        )

        _NAV_DISPLAY = [
            "Overview",
            "Data Explorer",
            "Sales Predictions",
            "Feature Impact",
            "Time-Series Forecast",
            "Product Advisor",
            "Model Insights",
        ]
        _NAV_INTERNAL = {
            "Overview":             "Overview",
            "Data Explorer":        "Data Explorer",
            "Sales Predictions":    "Sales Predictions",
            "Feature Impact":       "Feature Impact",
            "Time-Series Forecast": "📅 Time-Series Forecast",
            "Product Advisor":      "Product Advisor",
            "Model Insights":       "Model Insights",
        }
        sel   = st.radio("Navigation", _NAV_DISPLAY, label_visibility="collapsed")
        page  = _NAV_INTERNAL[sel]

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:11px;color:#484f58;line-height:1.6">'
            'Source: <code style="color:#4f8ef7;font-size:10px">'
            'synthetic_sales.csv</code></p>',
            unsafe_allow_html=True,
        )

    # â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    data_path  = "data/synthetic_sales.csv"
    model_path = "artifacts/model.joblib"

    try:
        with st.spinner("Loadingâ€¦"):
            df            = load_data(data_path)
            model         = load_model(model_path)
            X, y          = prepare_features(df)
            df_with_preds = prepare_data(model, X, y, df)

        feature_impact = get_feature_impact_simple(df)

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # OVERVIEW
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        if page == "Overview":
            st.markdown(
                '<p class="pg-title">Overview</p>'
                '<p class="pg-sub">Model performance summary, dataset stats, and top revenue drivers.</p>',
                unsafe_allow_html=True,
            )

            metrics = _load_metrics()
            r2  = metrics.get("r2")
            mae = metrics.get("mae")
            cvr = metrics.get("cv_r2")

            # KPI row
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(
                    _kpi("Total Records", f"{len(df):,}", f"{df['brand'].nunique()} brands"),
                    unsafe_allow_html=True)
            with k2:
                st.markdown(
                    _kpi("Avg Revenue", f"${df['revenue'].mean()/1e6:.2f}M", "per product record"),
                    unsafe_allow_html=True)
            with k3:
                st.markdown(
                    _kpi("Model R2", f"{r2:.4f}" if r2 else "â€”",
                         "Excellent fit" if r2 and r2 > 0.9 else "", pos=bool(r2 and r2 > 0.9)),
                    unsafe_allow_html=True)
            with k4:
                st.markdown(
                    _kpi("Mean Abs Error", f"${mae:,.0f}" if mae else "â€”", "prediction tolerance"),
                    unsafe_allow_html=True)

            st.markdown("<hr class='div'/>", unsafe_allow_html=True)

            left, right = st.columns([1, 1.1], gap="large")

            with left:
                st.markdown(_sec("Dataset Summary"), unsafe_allow_html=True)
                st.dataframe(pd.DataFrame({
                    "Attribute": ["Brands", "Operating Systems", "Avg Price",
                                  "Price Range", "Quarters"],
                    "Value": [
                        ", ".join(sorted(df["brand"].unique())),
                        ", ".join(sorted(df["os"].unique())),
                        f"${df['price'].mean():.2f}",
                        f"${df['price'].min():.0f} to ${df['price'].max():.0f}",
                        "Q1 to Q4",
                    ],
                }), use_container_width=True, hide_index=True)

                if metrics:
                    st.markdown(_sec("Performance Metrics"), unsafe_allow_html=True)
                    lmap = {
                        "r2":           ("Test R2",             "{:.4f}"),
                        "mse":          ("Test MSE ($²)",       "{:,.0f}"),
                        "mae":          ("Test MAE",            "${:,.0f}"),
                        "rmse":         ("RMSE",                "${:,.0f}"),
                        "mape":         ("MAPE",                "{:.2f}%"),
                        "cv_r2":        ("CV R2 (5-fold)",      "{:.4f}"),
                        "cv_mse":       ("CV MSE ($²)",         "{:,.0f}"),
                        "cv_mae":       ("CV MAE",              "${:,.0f}"),
                        "std_residuals":("Residual Std Dev",    "${:,.0f}"),
                    }
                    rows = [{"Metric": lmap[k][0], "Value": lmap[k][1].format(v)}
                            for k, v in metrics.items() if k in lmap]
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with right:
                st.markdown(_sec("Top Feature Drivers"), unsafe_allow_html=True)
                if len(feature_impact) > 0:
                    fig, ax = plt.subplots(figsize=(7, 4.2))
                    top5   = feature_impact.head(5).copy()
                    clrs   = [_PAL[0]] * len(top5); clrs[0] = _PAL[1]
                    ax.barh(range(len(top5)), top5["mean_abs_shap"],
                            color=clrs, edgecolor="none", height=0.52)
                    ax.set_yticks(range(len(top5)))
                    ax.set_yticklabels(top5["feature"], fontsize=9.5, color=_TXT)
                    ax.invert_yaxis()
                    _style(ax, title="Top 5 Revenue Drivers (Mean |SHAP|)")
                    _dollar(ax, "x")
                    ax.grid(axis="x", color=_GRID, linewidth=0.8)
                    ax.grid(axis="y", visible=False)
                    fig.tight_layout(pad=1.2)
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.markdown(_notice("Feature impact data unavailable.", "warn"),
                                unsafe_allow_html=True)

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # DATA EXPLORER
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        elif page == "Data Explorer":
            st.markdown(
                '<p class="pg-title">Data Explorer</p>'
                '<p class="pg-sub">Inspect the dataset and analyse distribution patterns.</p>',
                unsafe_allow_html=True,
            )

            st.markdown(_sec("Dataset Preview"), unsafe_allow_html=True)
            st.dataframe(df.head(15), use_container_width=True, hide_index=True)

            st.markdown("<hr class='div'/>", unsafe_allow_html=True)
            st.markdown(_sec("Distribution Analysis"), unsafe_allow_html=True)

            c1, c2 = st.columns(2, gap="large")
            with c1:
                fig, ax = plt.subplots(figsize=(7, 3.8))
                ax.hist(df["revenue"], bins=30, color=_PAL[0],
                        edgecolor=_BG, linewidth=0.4, alpha=0.9)
                _style(ax, title="Revenue Distribution",
                       xlabel="Revenue ($)", ylabel="Frequency")
                _dollar(ax, "x")
                fig.tight_layout(pad=1.2)
                st.pyplot(fig, use_container_width=True)

            with c2:
                fig, ax = plt.subplots(figsize=(7, 3.8))
                br = df.groupby("brand")["revenue"].sum().sort_values(ascending=True)
                ax.barh(br.index, br.values,
                        color=_PAL[:len(br)], edgecolor="none", height=0.48)
                _style(ax, title="Total Revenue by Brand", xlabel="Revenue ($)")
                _dollar(ax, "x")
                ax.grid(axis="x", color=_GRID, linewidth=0.8)
                ax.grid(axis="y", visible=False)
                fig.tight_layout(pad=1.2)
                st.pyplot(fig, use_container_width=True)

            c3, c4 = st.columns(2, gap="large")
            with c3:
                fig, ax = plt.subplots(figsize=(7, 3.8))
                ax.scatter(df["price"], df["sales_volume"],
                           alpha=0.3, s=18, color=_PAL[0], edgecolors="none")
                _style(ax, title="Price vs Sales Volume",
                       xlabel="Price ($)", ylabel="Sales Volume")
                fig.tight_layout(pad=1.2)
                st.pyplot(fig, use_container_width=True)

            with c4:
                fig, ax = plt.subplots(figsize=(7, 3.8))
                ax.scatter(df["battery"], df["revenue"],
                           alpha=0.3, s=18, color=_PAL[1], edgecolors="none")
                _style(ax, title="Battery Capacity vs Revenue",
                       xlabel="Battery (mAh)", ylabel="Revenue ($)")
                _dollar(ax, "y")
                fig.tight_layout(pad=1.2)
                st.pyplot(fig, use_container_width=True)

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # SALES PREDICTIONS
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        elif page == "Sales Predictions":
            st.markdown(
                '<p class="pg-title">Sales Predictions</p>'
                '<p class="pg-sub">Actual vs predicted revenue, error analysis, and filtered record view.</p>',
                unsafe_allow_html=True,
            )

            st.markdown(_sec("Actual vs Predicted Revenue"), unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(df_with_preds["revenue"], df_with_preds["predicted_revenue"],
                       alpha=0.3, s=16, color=_PAL[0], edgecolors="none")
            lo = df_with_preds["revenue"].min(); hi = df_with_preds["revenue"].max()
            ax.plot([lo, hi], [lo, hi], color=_PAL[4], linewidth=1.4,
                    linestyle="--", label="Perfect prediction")
            _style(ax, title="Actual vs Predicted Revenue",
                   xlabel="Actual Revenue ($)", ylabel="Predicted Revenue ($)", legend=True)
            _dollar(ax, "x"); _dollar(ax, "y")
            fig.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)

            st.markdown("<hr class='div'/>", unsafe_allow_html=True)
            st.markdown(_sec("Filter Predictions"), unsafe_allow_html=True)

            fc1, fc2 = st.columns(2, gap="large")
            with fc1:
                brand_filter = st.multiselect(
                    "Brand", df["brand"].unique(),
                    default=list(df["brand"].unique())[:2])
            with fc2:
                os_filter = st.multiselect(
                    "Operating System", df["os"].unique(),
                    default=list(df["os"].unique()))

            filtered_df = df_with_preds[
                df_with_preds["brand"].isin(brand_filter) &
                df_with_preds["os"].isin(os_filter)
            ]

            if len(filtered_df) > 0:
                tbl = filtered_df[[
                    "brand", "os", "price", "ram", "storage", "battery",
                    "revenue", "predicted_revenue", "error_pct"
                ]].head(25).rename(columns={
                    "brand": "Brand", "os": "OS", "price": "Price ($)",
                    "ram": "RAM", "storage": "Storage", "battery": "Battery",
                    "revenue": "Actual ($)", "predicted_revenue": "Predicted ($)",
                    "error_pct": "Error (%)",
                })
                st.dataframe(tbl, use_container_width=True, hide_index=True)

                st.markdown(_sec("Error Distribution"), unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 3.8))
                ax.hist(filtered_df["error_pct"].clip(-50, 50), bins=30,
                        color=_PAL[2], edgecolor=_BG, linewidth=0.4, alpha=0.9)
                ax.axvline(0, color=_PAL[4], linewidth=1.4, linestyle="--",
                           label="Zero error")
                _style(ax, title="Prediction Error Distribution (clipped Â±50 %)",
                       xlabel="Error (%)", ylabel="Frequency", legend=True)
                fig.tight_layout(pad=1.2)
                st.pyplot(fig, use_container_width=True)
            else:
                st.markdown(_notice("No records match the selected filters.", "warn"),
                            unsafe_allow_html=True)

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # FEATURE IMPACT
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        elif page == "Feature Impact":
            st.markdown(
                '<p class="pg-title">Feature Impact Analysis</p>'
                '<p class="pg-sub">'
                'SHAP (SHapley Additive exPlanations) â€” each feature\'s marginal '
                'contribution to predicted revenue.'
                '</p>',
                unsafe_allow_html=True,
            )

            if len(feature_impact) > 0:
                tbl_col, chart_col = st.columns([1, 1.6], gap="large")

                with tbl_col:
                    st.markdown(_sec("Feature Ranking"), unsafe_allow_html=True)
                    fi_display = feature_impact.copy()
                    fi_display.columns = ["Feature", "Mean |SHAP|"]
                    fi_display["Mean |SHAP|"] = fi_display["Mean |SHAP|"].map("${:,.0f}".format)
                    st.dataframe(fi_display, use_container_width=True, hide_index=True)

                with chart_col:
                    st.markdown(_sec("Importance Chart"), unsafe_allow_html=True)
                    top_n = st.slider(
                        "Features to display",
                        min_value=5, max_value=min(20, len(feature_impact)), value=10)
                    top_imp = feature_impact.head(top_n).sort_values("mean_abs_shap", ascending=True)
                    clrs    = [_PAL[0]] * len(top_imp); clrs[-1] = _PAL[1]
                    fig, ax = plt.subplots(figsize=(7, max(3.2, top_n * 0.4)))
                    ax.barh(top_imp["feature"], top_imp["mean_abs_shap"],
                            color=clrs, edgecolor="none", height=0.52)
                    _style(ax, title=f"Top {top_n} Features (Mean |SHAP|)",
                           xlabel="Mean |SHAP| Value")
                    _dollar(ax, "x")
                    ax.grid(axis="x", color=_GRID, linewidth=0.8)
                    ax.grid(axis="y", visible=False)
                    fig.tight_layout(pad=1.2)
                    st.pyplot(fig, use_container_width=True)

                st.markdown("<hr class='div'/>", unsafe_allow_html=True)
                top_f = feature_impact.iloc[0]["feature"]
                top_v = feature_impact.iloc[0]["mean_abs_shap"]
                st.markdown(
                    _notice(f"<strong>{top_f}</strong> is the dominant revenue driver "
                            f"(mean |SHAP| = ${top_v:,.0f}). Brand, camera specs, battery, "
                            f"and promotional activity follow as the next strongest signals."),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    _notice("Feature impact data unavailable. Run the analysis script first.", "warn"),
                    unsafe_allow_html=True,
                )

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # TIME-SERIES FORECAST
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        elif page == "📅 Time-Series Forecast":
            st.markdown(
                '<p class="pg-title">Time-Series Revenue Forecast</p>'
                '<p class="pg-sub">'
                'Facebook Prophet decomposes revenue into trend + seasonality. '
                'Every forecast period includes a 95 % confidence interval.'
                '</p>',
                unsafe_allow_html=True,
            )

            total_csv = Path("artifacts/prophet_forecast_total.csv")
            brand_csv = Path("artifacts/prophet_forecast_by_brand.csv")
            total_png = Path("artifacts/prophet_forecast_total.png")
            brand_png = Path("artifacts/prophet_forecast_by_brand.png")

            if not total_csv.exists():
                st.markdown(
                    _notice(
                        "<strong>Forecast artifacts not found.</strong> Generate them by running:<br/><br/>"
                        "<code>python src/modeling/forecast_prophet.py "
                        "--data data/synthetic_sales.csv --periods 4</code>",
                        "warn",
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(_sec("Total Revenue â€” All Brands"), unsafe_allow_html=True)
                if total_png.exists():
                    st.image(str(total_png), use_container_width=True)

                total_fc    = pd.read_csv(total_csv, parse_dates=["ds"])
                future_rows = total_fc[total_fc["ds"] > total_fc["ds"].iloc[-5]]

                st.markdown(_sec("Forward Quarterly Estimates"), unsafe_allow_html=True)
                dfc = future_rows[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                dfc.columns = ["Date", "Expected Revenue", "Lower Bound", "Upper Bound"]
                for c in ["Expected Revenue", "Lower Bound", "Upper Bound"]:
                    dfc[c] = pd.to_numeric(
                        dfc[c].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
                    ).map("${:,.0f}".format)
                st.dataframe(dfc.reset_index(drop=True),
                             use_container_width=True, hide_index=True)

                st.markdown("<hr class='div'/>", unsafe_allow_html=True)
                st.markdown(_sec("Revenue Forecast by Brand"), unsafe_allow_html=True)
                if brand_png.exists():
                    st.image(str(brand_png), use_container_width=True)

                if brand_csv.exists():
                    brand_fc      = pd.read_csv(brand_csv, parse_dates=["ds"])
                    brand_selected = st.selectbox(
                        "Brand forecast detail", sorted(brand_fc["brand"].unique()))
                    bdf = brand_fc[brand_fc["brand"] == brand_selected].tail(8).copy()
                    bdf = bdf[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                    bdf.columns = ["Date", "Expected ($)", "Lower ($)", "Upper ($)"]
                    st.dataframe(bdf.reset_index(drop=True),
                                 use_container_width=True, hide_index=True)

                st.markdown(
                    _notice(
                        "<strong>Reading the chart:</strong> Solid line = expected revenue. "
                        "Shaded band = 95 % confidence interval. "
                        "Dots = historical actuals. Wider bands indicate greater uncertainty "
                        "further from the training window."
                    ),
                    unsafe_allow_html=True,
                )

        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # MODEL INSIGHTS  (was: Innovation Showcase)
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # ─────────────────────────────────────────────────────────
        # PRODUCT ADVISOR
        # ─────────────────────────────────────────────────────────
        elif page == "Product Advisor":
            st.markdown(
                '<p class="pg-title">Product Advisor</p>'
                '<p class="pg-sub">Configure a phone spec and get an instant revenue prediction '
                'plus ranked suggestions for which spec changes would increase revenue the most.</p>',
                unsafe_allow_html=True,
            )

            # Load CI alpha from saved artifact
            _ci_alpha = 0.06
            try:
                _rs = joblib.load("artifacts/residual_std.joblib")
                _ci_alpha = _rs.get("alpha", 0.06)
            except Exception:
                pass

            def _predict_spec(spec_dict):
                """Predict revenue for a single phone spec dict, with 95% CI."""
                # Add dummy revenue so prepare_features can find the target column
                row_df = pd.DataFrame([{**spec_dict, "revenue": 0.0, "sales_volume": 0}])
                combined = pd.concat(
                    [df.drop(columns=["predicted_revenue"], errors="ignore"),
                     row_df],
                    ignore_index=True,
                )
                X_all, _ = prepare_features(combined)
                X_row    = X_all.tail(1).reindex(columns=model.feature_names_in_, fill_value=0)
                p        = float(model.predict(X_row)[0])
                hw       = 1.96 * _ci_alpha * abs(p)
                return p, p - hw, p + hw

            col_in, col_out = st.columns([1, 1.1], gap="large")

            with col_in:
                st.markdown(_sec("Phone Specification"), unsafe_allow_html=True)
                _brands    = sorted(df["brand"].unique().tolist())
                _oses      = sorted(df["os"].unique().tolist())
                _brand     = st.selectbox("Brand", _brands)
                _os_choice = st.selectbox("Operating System", _oses)
                _price     = st.slider("Price ($)", int(df["price"].min()), int(df["price"].max()),
                                       int(df["price"].median()))
                _ram       = st.select_slider("RAM (GB)", options=[2, 4, 6, 8, 12, 16, 32], value=8)
                _storage   = st.select_slider("Storage (GB)", options=[32, 64, 128, 256, 512], value=128)
                _battery   = st.slider("Battery (mAh)", int(df["battery"].min()), int(df["battery"].max()),
                                       int(df["battery"].median()))
                _camera    = st.slider("Camera (MP)", int(df["camera_mp"].min()), int(df["camera_mp"].max()),
                                       int(df["camera_mp"].median()))
                _promo     = 1 if st.checkbox("Promotional Discount Active") else 0
                _quarter   = st.selectbox("Quarter", [1, 2, 3, 4])

            _base_spec = {
                "brand": _brand, "os": _os_choice, "price": _price, "ram": _ram,
                "storage": _storage, "battery": _battery, "camera_mp": _camera,
                "promo": _promo, "sentiment": 0.5, "quarter": _quarter,
            }

            _pred, _lower, _upper = _predict_spec(_base_spec)

            with col_out:
                st.markdown(_sec("Revenue Forecast"), unsafe_allow_html=True)
                st.markdown(
                    _kpi("Predicted Revenue", f"${_pred:,.0f}",
                         note=f"95% CI: ${_lower:,.0f} \u2014 ${_upper:,.0f}"),
                    unsafe_allow_html=True,
                )
                st.markdown("<br/>", unsafe_allow_html=True)
                st.markdown(
                    _kpi("Confidence Range Width", f"${_upper - _lower:,.0f}",
                         note="Narrower = higher model certainty"),
                    unsafe_allow_html=True,
                )

            # Suggestions
            st.markdown("<hr class='div'/>", unsafe_allow_html=True)
            st.markdown(_sec("Suggestions to Boost Revenue"), unsafe_allow_html=True)

            _candidates = [
                {"label": "Add promotional discount",    "field": "promo",     "new_val": 1,    "ok": _promo == 0},
                {"label": "Upgrade RAM to 16 GB",         "field": "ram",       "new_val": 16,   "ok": _ram < 16},
                {"label": "Upgrade storage to 256 GB",    "field": "storage",   "new_val": 256,  "ok": _storage < 256},
                {"label": "Upgrade storage to 512 GB",    "field": "storage",   "new_val": 512,  "ok": _storage < 512},
                {"label": "Boost camera to 48 MP",        "field": "camera_mp", "new_val": 48,   "ok": _camera < 48},
                {"label": "Increase battery to 5000 mAh", "field": "battery",   "new_val": 5000, "ok": _battery < 5000},
                {"label": "Launch in Q4 (peak quarter)",  "field": "quarter",   "new_val": 4,    "ok": _quarter != 4},
            ]

            _suggestions = []
            for _c in _candidates:
                if not _c["ok"]:
                    continue
                _mod_spec = {**_base_spec, _c["field"]: _c["new_val"]}
                _mod_pred, _, _ = _predict_spec(_mod_spec)
                _gain = _mod_pred - _pred
                if _gain > 0:
                    _suggestions.append({"label": _c["label"], "gain": _gain, "new_pred": _mod_pred})

            _suggestions.sort(key=lambda x: x["gain"], reverse=True)

            if _suggestions:
                _top   = _suggestions[:3]
                _scols = st.columns(len(_top), gap="medium")
                for _i, _s in enumerate(_top):
                    with _scols[_i]:
                        st.markdown(
                            f'<div class="sugg-card">'
                            f'<div class="sugg-icon">&#8593;</div>'
                            f'<div class="sugg-label">{_s["label"]}</div>'
                            f'<div class="sugg-gain">+${_s["gain"]:,.0f}</div>'
                            f'<div class="sugg-new">New forecast: ${_s["new_pred"]:,.0f}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                if len(_suggestions) > 3:
                    st.markdown("<br/>", unsafe_allow_html=True)
                    _more = pd.DataFrame(_suggestions[3:])
                    _more["gain"]     = _more["gain"].map("+${:,.0f}".format)
                    _more["new_pred"] = _more["new_pred"].map("${:,.0f}".format)
                    _more.columns = ["Suggestion", "Revenue Gain", "New Forecast"]
                    st.dataframe(_more, use_container_width=True, hide_index=True)
            else:
                st.markdown(
                    _notice("This configuration is already near-optimal. No further gains detected.", "success"),
                    unsafe_allow_html=True,
                )

        elif page == "Model Insights":
            st.markdown(
                '<p class="pg-title">Model Insights</p>'
                '<p class="pg-sub">Technical innovations, validation methodology, and full metrics report.</p>',
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown(
                    _feat("5-Fold Cross-Validation", [
                        "Evaluated across 5 independent train / test splits",
                        "R2, MSE, and MAE reported with standard deviation",
                        "Confirms model generalises beyond a single holdout",
                    ]),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _feat("Ensemble Learning — RF + XGBoost", [
                        "VotingRegressor averages both models' predictions",
                        "Reduces individual model bias and variance",
                        "XGBoost activates automatically when installed",
                    ]),
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    _feat("95 % Prediction Confidence Intervals", [
                        "Upper and lower bounds derived from residual std dev",
                        "Every prediction includes a measurable uncertainty range",
                        "Enables risk-aware planning and scenario analysis",
                    ]),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    _feat("Sinusoidal Temporal Feature Engineering", [
                        "Quarter encoded as sin / cos to preserve cyclical continuity",
                        "Prevents discontinuity at Q4 → Q1 boundary",
                        "Foundation for downstream time-series models",
                    ]),
                    unsafe_allow_html=True,
                )

            st.markdown("<hr class='div'/>", unsafe_allow_html=True)

            metrics = _load_metrics()
            if metrics:
                st.markdown(_sec("Model Performance"), unsafe_allow_html=True)
                m1, m2, m3, m4, m5 = st.columns(5, gap="large")
                with m1:
                    r2v = metrics.get("r2")
                    st.metric("Test R2",       f"{r2v:.4f}" if r2v else "—",
                              delta="Excellent" if r2v and r2v > 0.9 else None)
                with m2:
                    cvr = metrics.get("cv_r2")
                    st.metric("CV R2 (5-fold)", f"{cvr:.4f}" if cvr else "—",
                              delta="Robust" if cvr else None)
                with m3:
                    mae = metrics.get("mae")
                    st.metric("Test MAE ($)",   f"${mae:,.0f}" if mae else "—")
                with m4:
                    rmse = metrics.get("rmse")
                    st.metric("RMSE ($)",       f"${rmse:,.0f}" if rmse else "—")
                with m5:
                    mape = metrics.get("mape")
                    st.metric("MAPE (%)",       f"{mape:.2f}%" if mape else "—")

                st.markdown(_sec("Full Metrics Report"), unsafe_allow_html=True)
                lmap = {
                    "r2":           "Test R2",
                    "mse":          "Test MSE ($²)",
                    "mae":          "Test MAE ($)",
                    "rmse":         "RMSE ($)",
                    "mape":         "MAPE (%)",
                    "cv_r2":        "CV R2 (5-fold)",
                    "cv_mse":       "CV MSE ($²)",
                    "cv_mae":       "CV MAE ($)",
                    "std_residuals":"Residual Std Dev ($)",
                }
                rows = [{"Metric": lmap.get(k, k), "Value": v}
                        for k, v in metrics.items() if k in lmap]
                if rows:
                    st.dataframe(pd.DataFrame(rows),
                                 use_container_width=True, hide_index=True)
            else:
                st.markdown(
                    _notice("Run the training script to generate metrics.", "warn"),
                    unsafe_allow_html=True,
                )

            st.markdown("<hr class='div'/>", unsafe_allow_html=True)
            st.markdown(_sec("Capability Summary"), unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Capability":      ["Ensemble Models",    "Cross-Validation",
                                    "Confidence Intervals","Temporal Features",
                                    "SHAP Explainability", "Feature Impact Analysis"],
                "Benefit":         ["RF + XGBoost voting","Robust evaluation",
                                    "Uncertainty bounds",  "Quarterly seasonality",
                                    "Interpretable outputs","Driver identification"],
                "Business Impact": ["Higher accuracy",    "Production-safe",
                                    "Risk-aware planning", "Seasonal forecasting",
                                    "Stakeholder trust",   "Product strategy"],
            }), use_container_width=True, hide_index=True)

            st.markdown("<hr class='div'/>", unsafe_allow_html=True)
            st.markdown(_sec("Recommended Future Enhancements"), unsafe_allow_html=True)
            st.markdown(
                _notice(
                    "<strong>Potential improvements for production:</strong><br/>"
                    "TimeSeriesSplit validation &nbsp;·&nbsp; Multi-year real data &nbsp;·&nbsp; "
                    "LSTM / Transformer forecasting &nbsp;·&nbsp; MLflow experiment tracking &nbsp;·&nbsp; "
                    "Real-time prediction API &nbsp;·&nbsp; Model drift monitoring &nbsp;·&nbsp; "
                    "Hierarchical brand forecasting"
                ),
                unsafe_allow_html=True,
            )

    # â”€â”€ Error state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    except Exception as e:
        st.markdown(_CSS, unsafe_allow_html=True)
        st.markdown(
            _notice(f"<strong>Setup required.</strong> {e}", "error"),
            unsafe_allow_html=True,
        )
        st.markdown(
            _notice(
                "<strong>Getting started:</strong><br/><br/>"
                "1. Generate data:<br/>"
                "<code>python data/generate_synthetic.py --output data/synthetic_sales.csv --n 5000</code><br/><br/>"
                "2. Train the model:<br/>"
                "<code>python src/modeling/train.py --data data/synthetic_sales.csv --outdir artifacts</code><br/><br/>"
                "3. Refresh this page."
            ),
            unsafe_allow_html=True,
        )
        


if __name__ == "__main__":
    main()
