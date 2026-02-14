AI-Powered Sales Forecasting & Feature Impact Analysis
====================================================

This repository provides a minimal, end-to-end example for building an AI-driven
sales forecasting and feature impact analysis pipeline for consumer mobile phones.

What you'll find here
- `data/generate_synthetic.py` — produces a synthetic dataset mimicking
  quarterly mobile-phone sales with product features and market signals.
- `src/modeling/train.py` — trains a Random Forest regressor and saves a model
  artifact and evaluation metrics.
- `src/modeling/predict.py` — simple prediction helper to load the saved model
  and predict on new rows or CSV files.
- `src/analysis/feature_impact.py` — computes SHAP values for interpretability
  and writes a summary plot and CSV of feature impacts.
- `requirements.txt` — Python dependencies.

Quick start

1. Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Generate synthetic data and train:

```bash
python data/generate_synthetic.py --output data/synthetic_sales.csv --n 5000
python src/modeling/train.py --data data/synthetic_sales.csv --outdir artifacts
```

4. Run feature impact analysis:

```bash
python src/analysis/feature_impact.py --data data/synthetic_sales.csv --model artifacts/model.joblib --outdir artifacts
```

5. Make a sample prediction (see `src/modeling/predict.py` for usage):

```bash
python src/modeling/predict.py --model artifacts/model.joblib --input data/synthetic_sales.csv --out predictions.csv
```

Notes and next steps
- Replace the synthetic data generator with your real datasets (historical sales,
  product specs, pricing, promos, sentiment, competitor prices, macro indicators).
- Consider time-series models (Prophet, ARIMA, deep learning) and cross-product
  hierarchical forecasting for multi-product portfolios.
- Add monitoring and retraining on new quarterly data.
