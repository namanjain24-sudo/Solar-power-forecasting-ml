# Solar Power Forecasting — ML Project

A supervised learning project that predicts solar panel DC power output using weather and time-based features. Built with **RandomForest Regression** and deployed as an interactive **Streamlit** dashboard.

---

## Project Overview

| Detail | Value |
|---|---|
| **Problem** | Predict solar DC power output from weather data |
| **Type** | Supervised Learning — Regression |
| **Model** | RandomForest Regressor (100 decision trees) |
| **Target** | DC Power (Watts) |
| **Evaluation** | MAE, RMSE, R², MAPE |
| **Deployment** | Streamlit web dashboard |

---

## ML Concepts Demonstrated

| Syllabus Concept | Implementation in Project |
|---|---|
| Supervised Learning | Regression — predicting continuous DC Power from labelled data |
| Data Preprocessing | Feature engineering (hour, month), categorical encoding (SOURCE_KEY) |
| Decision Trees | RandomForest is an ensemble of decision trees using Gini impurity |
| Model Evaluation | MAE, RMSE, R² score, MAPE — four standard regression metrics |
| Cross-Validation | TimeSeriesSplit (k=5) — prevents temporal data leakage |
| Feature Importance | Gini-based feature ranking for model explainability |
| Deployment | Interactive Streamlit dashboard with prediction and visualization |

---

## Project Structure

```
solar-power-forecasting-ml/
│
├── app/
│   └── streamlit_app.py          # Streamlit dashboard
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── load_data.py          # Data loading utility
│   ├── preprocessing/
│   │   └── preprocessing.py      # Feature encoding & train-test split
│   ├── modeling/
│   │   └── train.py              # Model training pipeline
│   ├── evaluation/
│   │   └── metrics.py            # MAE, RMSE, R², MAPE functions
│   └── utils/
│       └── helpers.py            # Plot styling & UI helpers
│
├── data/
│   └── processed/
│       └── solar_final.csv       # Preprocessed dataset (gitignored)
│
├── models/
│   └── solar_model.pkl           # Trained model (gitignored)
│
├── reports/                      # Generated plots from training
├── notebooks/                    # EDA notebooks (optional)
├── tests/                        # Unit tests (optional)
│
├── training_log.json             # Training metrics & hyperparameters
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Features Used

| Feature | Description |
|---|---|
| `SOURCE_KEY` | Inverter ID (encoded to numeric) |
| `AMBIENT_TEMPERATURE` | Ambient air temperature (°C) |
| `MODULE_TEMPERATURE` | Solar panel surface temperature (°C) |
| `IRRADIATION` | Solar irradiance (kW/m²) |
| `hour` | Hour of the day (0–23) |
| `month` | Month of the year (1–12) |

---

## Model Performance

### Holdout Evaluation (80/20 Time-Based Split)

| Metric | Value |
|---|---|
| MAE | 460.65 W |
| RMSE | 933.25 W |
| R² | 0.9296 |

### Cross-Validation (TimeSeriesSplit, k=5)

| Metric | Mean ± Std |
|---|---|
| MAE | 744.68 ± 268.68 |
| RMSE | 1626.86 ± 506.68 |
| R² | 0.8096 ± 0.0909 |

---

## How to Run

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python -m src.modeling.train
```

This will:
- Load and preprocess the solar dataset
- Train a RandomForest model (100 trees)
- Evaluate on holdout set + 5-fold TimeSeriesSplit
- Save model to `models/solar_model.pkl`
- Save metrics to `training_log.json`
- Generate plots in `reports/`

### 3. Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Dashboard tabs:
- **Predict** — Single & batch solar power prediction
- **Data Analysis** — EDA with seasonal and daily trends
- **Model Evaluation** — Metrics, scatter plots, residuals, feature importance
- **Forecast** — Future hourly power forecast
- **Logs & Export** — Training logs and result export

---

## Dataset

- **Source:** Solar power plant generation data
- **Period:** May 15, 2020 – June 17, 2020
- **Size:** ~77,000 records
- **Content:** Weather measurements + inverter-level power output

---

## Tech Stack

- Python 3.x
- scikit-learn
- pandas / numpy
- matplotlib
- Streamlit
- joblib
