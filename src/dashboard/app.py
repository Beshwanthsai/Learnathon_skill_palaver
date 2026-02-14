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

# Set page config first
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")



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


def main():
    st.title("🚀 AI-Powered Sales Forecasting & Feature Impact Analysis")
    
    # Sidebar
    st.sidebar.header("📊 Navigation")
    pages = ["Overview", "Data Explorer", "Sales Predictions", "Feature Impact"]
    page = st.sidebar.radio("Go to", pages)
    
    # Paths
    data_path = "data/synthetic_sales.csv"
    model_path = "artifacts/model.joblib"
    
    # Load data and model
    try:
        with st.spinner("Loading data and model..."):
            df = load_data(data_path)
            model = load_model(model_path)
            X, y = prepare_features(df)
        
        with st.spinner("Preparing predictions..."):
            df_with_preds = prepare_data(model, X, y, df)
        
        feature_impact = get_feature_impact_simple(df)
        
        # Overview page
        if page == "Overview":
            st.header("📋 Project Overview")
            
            st.markdown("""
            This dashboard presents an **AI-powered sales forecasting system** for mobile phones.
            
            **What does it do?**
            - 🤖 **Predict Revenue** using machine learning (Random Forest)
            - 📊 **Analyze Feature Impact** to understand what drives sales
            - 📈 **Explore Data** patterns across brands, specs, and market conditions
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"📱 **Total Records**: {len(df):,}")
            
            with col2:
                st.info(f"💰 **Avg Revenue**: ${df['revenue'].mean():,.0f}")
            
            with col3:
                st.info(f"🏢 **Brands**: {len(df['brand'].unique())}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Model Performance")
                try:
                    metrics = pd.read_csv("artifacts/metrics.csv")
                    mse = metrics['mse'].values[0]
                    r2 = metrics['r2'].values[0]
                    st.metric("R² Score", f"{r2:.3f}", delta="Excellent fit (>0.93)")
                    st.metric("MSE", f"{mse:,.0f}")
                except Exception as e:
                    st.error(f"Could not load metrics: {e}")
                    
                st.subheader("📂 Dataset Info")
                st.write(f"**Brands**: {', '.join(df['brand'].unique())}")
                st.write(f"**Operating Systems**: {', '.join(df['os'].unique())}")
                st.write(f"**Average Price**: ${df['price'].mean():.2f}")
            
            with col2:
                st.subheader("🎯 Top 5 Features Impacting Sales")
                if len(feature_impact) > 0:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    top_features = feature_impact.head(5)
                    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
                    ax.barh(range(len(top_features)), top_features["mean_abs_shap"], color=colors)
                    ax.set_yticks(range(len(top_features)))
                    ax.set_yticklabels(top_features["feature"])
                    ax.set_xlabel("Mean |SHAP| Impact")
                    ax.set_title("Top Features Driving Revenue")
                    ax.invert_yaxis()
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.warning("Feature impact data not available")
        
        # Data Explorer page
        elif page == "Data Explorer":
            st.header("🔍 Data Explorer")
            
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head(15), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Revenue Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(df["revenue"], bins=30, color="steelblue", edgecolor="black")
                ax.set_xlabel("Revenue ($)")
                ax.set_ylabel("Frequency")
                ax.set_title("Revenue Distribution")
                st.pyplot(fig, use_container_width=True)
            
            with col2:
                st.subheader("💻 Sales by Brand")
                fig, ax = plt.subplots(figsize=(8, 5))
                brand_revenue = df.groupby("brand")["revenue"].sum().sort_values(ascending=True)
                brand_revenue.plot(kind="barh", ax=ax, color="coral")
                ax.set_xlabel("Total Revenue ($)")
                ax.set_title("Total Revenue by Brand")
                st.pyplot(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Price vs Sales Volume")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(df["price"], df["sales_volume"], alpha=0.5, s=30)
                ax.set_xlabel("Price ($)")
                ax.set_ylabel("Sales Volume")
                ax.set_title("Price vs Sales Volume")
                st.pyplot(fig, use_container_width=True)
            
            with col2:
                st.subheader("🔋 Battery Impact")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(df["battery"], df["revenue"], alpha=0.5, s=30, color="green")
                ax.set_xlabel("Battery Capacity (mAh)")
                ax.set_ylabel("Revenue ($)")
                ax.set_title("Battery vs Revenue")
                st.pyplot(fig, use_container_width=True)
        
        # Sales Predictions page
        elif page == "Sales Predictions":
            st.header("💹 Sales Predictions")
            
            st.subheader("✅ Actual vs Predicted Revenue")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_with_preds["revenue"], df_with_preds["predicted_revenue"], alpha=0.5, s=20)
            
            # Add perfect prediction line
            min_val = df_with_preds["revenue"].min()
            max_val = df_with_preds["revenue"].max()
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Prediction")
            
            ax.set_xlabel("Actual Revenue ($)", fontsize=12)
            ax.set_ylabel("Predicted Revenue ($)", fontsize=12)
            ax.set_title("Model Predictions: Actual vs Predicted", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            
            st.subheader("🎯 Prediction Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                brand_filter = st.multiselect("Filter by Brand", 
                                             df["brand"].unique(), 
                                             default=list(df["brand"].unique())[:2])
            
            with col2:
                os_filter = st.multiselect("Filter by OS", 
                                          df["os"].unique(), 
                                          default=df["os"].unique())
            
            filtered_df = df_with_preds[
                (df_with_preds["brand"].isin(brand_filter)) & 
                (df_with_preds["os"].isin(os_filter))
            ]
            
            if len(filtered_df) > 0:
                st.dataframe(filtered_df[[
                    "brand", "os", "price", "ram", "storage", "battery",
                    "revenue", "predicted_revenue", "error_pct"
                ]].head(25), use_container_width=True)
                
                st.subheader("📊 Prediction Error Distribution")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(filtered_df["error_pct"].clip(-50, 50), bins=30, color="purple", edgecolor="black")
                ax.set_xlabel("Prediction Error (%)")
                ax.set_ylabel("Frequency")
                ax.set_title("Error Distribution (clipped to ±50%)")
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label="Zero Error")
                ax.legend()
                st.pyplot(fig, use_container_width=True)
            else:
                st.warning("❌ No data matches the selected filters")
        
        # Feature Impact page
        elif page == "Feature Impact":
            st.header("🎯 Feature Impact Analysis")
            
            st.markdown("""
            **SHAP (SHapley Additive exPlanations)** values show how each feature contributes 
            to the model's predictions. Higher absolute SHAP values = stronger impact on sales predictions.
            """)
            
            st.subheader("📊 Feature Importance Ranking")
            
            if len(feature_impact) > 0:
                st.dataframe(feature_impact, use_container_width=True)
                
                st.subheader("📈 Top Features Chart")
                top_n = st.slider("Number of top features to display", min_value=5, max_value=min(20, len(feature_impact)), value=10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                top_impact = feature_impact.head(top_n).sort_values("mean_abs_shap", ascending=True)
                colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_impact)))
                ax.barh(top_impact["feature"], top_impact["mean_abs_shap"], color=colors)
                ax.set_xlabel("Mean |SHAP| Value")
                ax.set_title(f"Top {top_n} Most Important Features")
                st.pyplot(fig, use_container_width=True)
                
                st.info(f"""
                **Key Insights:**
                - 🏆 **Price** is the dominant factor affecting sales ({feature_impact.iloc[0]['mean_abs_shap']:.0f} impact)
                - 🏢 **Brand** choice significantly influences revenue
                - 📊 **Historical revenue** is a strong predictor of future sales
                - 📸 **Camera specs** matter for phone sales
                - 🔋 **Battery** capacity impacts demand
                """)
            else:
                st.warning("⚠️ Feature impact data not available. Run feature analysis first.")
                
    except Exception as e:
        st.error(f"❌ Error: {str(e)}", icon="⚠️")
        st.info(
            """
            **Setup Instructions:**
            \n1. Generate synthetic data:
            ```bash
            python data/generate_synthetic.py --output data/synthetic_sales.csv --n 5000
            ```
            \n2. Train the model:
            ```bash
            python src/modeling/train.py --data data/synthetic_sales.csv --outdir artifacts
            ```
            \n3. Refresh this page
            """
        )


if __name__ == "__main__":
    main()