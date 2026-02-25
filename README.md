# Solar Power Forecasting — ML Project

A supervised learning project that predicts solar panel DC power output using weather and time-based features. Built with **RandomForest Regression** and deployed as an interactive **Streamlit** dashboard.

---

## Project Overview

This project demonstrates core Machine Learning concepts applied to a real-world problem:

- **Problem Type:** Supervised Learning — Regression
- **Model:** RandomForest Regressor (tree-based)
- **Target Variable:** DC Power output (Watts)
- **Input Features:** Irradiation, ambient temperature, module temperature, inverter ID, hour, month

---

## ML Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Supervised Learning | Regression task — predicting continuous DC Power from labelled data |
| Data Preprocessing | Feature engineering (hour, month extraction), categorical encoding (SOURCE_KEY) |
| Decision Trees | RandomForest is an ensemble of decision trees using Gini impurity |
| Model Evaluation | MAE, RMSE, R² score, MAPE |
| Cross-Validation | TimeSeriesSplit (k=5) — prevents temporal data leakage |
| Feature Importance | Gini-based feature ranking for model explainability |
| Deployment | Interactive Streamlit web dashboard |

---

## Project Structure

```
solar-ai/
├── app/
│   └── streamlit_app.py      # Streamlit dashboard
├── data/
│   └── processed/
│       └── solar_final.csv    # Preprocessed dataset
├── ml/
│   ├── __init__.py
│   ├── load_data.py           # Data loading utility
│   ├── preprocessing.py       # Feature encoding & train-test split
│   └── train.py               # Model training pipeline
├── models/
│   ├── solar_model.pkl        # Trained RandomForest model
│   └── training_log.json      # Training metrics log
├── requirements.txt
└── README.md
```

---

## Setup & Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Train the model

```bash
python -m ml.train
```

This will:
- Load and preprocess the solar dataset
- Train a RandomForest model with 100 trees
- Evaluate using holdout metrics (MAE, RMSE, R², MAPE)
- Run 5-fold TimeSeriesSplit cross-validation
- Save the model to `models/solar_model.pkl`
- Save training logs to `models/training_log.json`

### Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

The dashboard provides:
- **Predict** — Single and batch solar power prediction
- **Data Analysis** — Exploratory data analysis with seasonal and daily trends
- **Model Evaluation** — Metrics, scatter plots, residual analysis, feature importance
- **Forecast** — Future hourly power forecast with solar physics simulation
- **Logs & Export** — Training logs and result export

---

## Model Performance

| Metric | Holdout (80/20) |
|---|---|
| MAE | ~460 W |
| RMSE | ~933 W |
| R² | 0.9296 |
| MAPE | ~110% |

Cross-validation (TimeSeriesSplit, k=5): R² = 0.81 ± 0.09

---

## Dataset

- **Source:** Solar power plant generation data
- **Period:** May 15, 2020 – June 17, 2020
- **Size:** ~77,000 records
- **Features:** Weather measurements (temperature, irradiation) + time features

---

## Technology Stack

- Python 3.x
- scikit-learn (RandomForest, metrics, TimeSeriesSplit)
- pandas, numpy
- matplotlib
- Streamlit
- joblib
