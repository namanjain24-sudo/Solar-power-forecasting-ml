"""
Model training pipeline for the Solar Power Forecasting project.

Pipeline:
  1. Load and preprocess data
  2. Train RandomForest Regressor
  3. Evaluate on holdout set (MAE, RMSE, R², MAPE)
  4. Run TimeSeriesSplit cross-validation (k=5)
  5. Save model, metrics log, and plots

Usage:
  python -m src.modeling.train
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import joblib
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from src.data.load_data import load_data
from src.preprocessing.preprocessing import encode_features, split_data, FEATURES
from src.evaluation.metrics import evaluate_model, compute_mape, print_metrics


def train_model():
    """Full training pipeline: load → preprocess → train → evaluate → save."""

    print("Loading data...")
    df = load_data()
    df = encode_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    # ── Train RandomForest ──
    print("\nTraining RandomForest...")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # ── Holdout evaluation ──
    preds = np.clip(model.predict(X_test), 0, None)
    holdout_metrics = evaluate_model(y_test, preds)
    print_metrics(holdout_metrics, "Holdout Evaluation (80/20 Time-Series Split)")

    # ── TimeSeriesSplit cross-validation (k=5) ──
    print("\nRunning TimeSeriesSplit Cross-Validation (k=5)...")

    df_sorted = df.sort_values("DATE_TIME")
    features = list(X_train.columns)
    X_full = df_sorted[features]
    y_full = df_sorted["DC_POWER"]

    tscv = TimeSeriesSplit(n_splits=5)
    cv_mae, cv_rmse, cv_r2, cv_mape = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_full), 1):
        X_cv_train = X_full.iloc[train_idx]
        X_cv_val = X_full.iloc[val_idx]
        y_cv_train = y_full.iloc[train_idx]
        y_cv_val = y_full.iloc[val_idx]

        cv_model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        cv_model.fit(X_cv_train, y_cv_train)
        cv_preds = np.clip(cv_model.predict(X_cv_val), 0, None)

        fold_metrics = evaluate_model(y_cv_val, cv_preds)
        cv_mae.append(fold_metrics["MAE"])
        cv_rmse.append(fold_metrics["RMSE"])
        cv_r2.append(fold_metrics["R2"])
        cv_mape.append(fold_metrics["MAPE"])

        print(f"  Fold {fold}: MAE={fold_metrics['MAE']:,.1f}  "
              f"RMSE={fold_metrics['RMSE']:,.1f}  "
              f"R²={fold_metrics['R2']:.4f}  "
              f"MAPE={fold_metrics['MAPE']:.1f}%")

    print(f"\n{'='*45}")
    print(f"  CV Average (k=5)")
    print(f"{'='*45}")
    print(f"  MAE  = {np.mean(cv_mae):,.2f} ± {np.std(cv_mae):,.2f}")
    print(f"  RMSE = {np.mean(cv_rmse):,.2f} ± {np.std(cv_rmse):,.2f}")
    print(f"  R²   = {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
    print(f"  MAPE = {np.nanmean(cv_mape):.2f} ± {np.nanstd(cv_mape):.2f} %")
    print(f"{'='*45}")

    # ── Save model ──
    joblib.dump(model, "models/solar_model.pkl")
    print("\nModel saved -> models/solar_model.pkl")

    # ── Save training log ──
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "model": "RandomForestRegressor",
        "hyperparameters": {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
        },
        "features": features,
        "target": "DC_POWER",
        "dataset": {
            "total_rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "date_range": f"{df['DATE_TIME'].min()} -> {df['DATE_TIME'].max()}",
        },
        "holdout_metrics": holdout_metrics,
        "cv_metrics_k5": {
            "MAE_mean": round(np.mean(cv_mae), 2),
            "MAE_std": round(np.std(cv_mae), 2),
            "RMSE_mean": round(np.mean(cv_rmse), 2),
            "RMSE_std": round(np.std(cv_rmse), 2),
            "R2_mean": round(np.mean(cv_r2), 4),
            "R2_std": round(np.std(cv_r2), 4),
            "MAPE_mean": round(np.nanmean(cv_mape), 2),
            "MAPE_std": round(np.nanstd(cv_mape), 2),
        },
    }

    with open("training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)
    print("Training log saved -> training_log.json")

    # ── Feature importance plot ──
    print("\nGenerating plots...")

    importance = model.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance,
    }).sort_values("Importance", ascending=False)

    print("\nFeature Importance:\n", imp_df)

    plt.figure(figsize=(8, 5))
    colors = ["#43A047", "#66BB6A", "#81C784", "#A5D6A7", "#C8E6C9", "#E8F5E9"]
    plt.barh(imp_df["Feature"], imp_df["Importance"],
             color=colors[:len(imp_df)], edgecolor="white", linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.title("Feature Importance — RandomForest", fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png", dpi=150)
    print("Feature importance plot saved -> reports/feature_importance.png")

    # ── Prediction vs actual plot ──
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values[:300], label="Actual", color="#FFA726",
             linewidth=1.5, alpha=0.8)
    plt.plot(preds[:300], label="Predicted", color="#43A047",
             linewidth=1.5, alpha=0.8)
    plt.legend(fontsize=10)
    plt.title("Prediction vs Actual Solar Power", fontsize=13, fontweight="bold")
    plt.xlabel("Test Sample Index")
    plt.ylabel("DC Power (W)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("reports/prediction_vs_actual.png", dpi=150)
    print("Prediction vs Actual plot saved -> reports/prediction_vs_actual.png")

    print("\nTraining complete!")


if __name__ == "__main__":
    train_model()
