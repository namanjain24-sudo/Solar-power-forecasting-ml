from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from .preprocessing import encode_features, split_data
from .load_data import load_data


def compute_mape(y_true, y_pred, threshold=10):
    """Compute MAPE, excluding near-zero actuals to avoid inf."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = np.abs(y_true) > threshold
    if mask.sum() == 0:
        return float("nan")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_model():
    print("Loading data...")

    # Load + preprocess
    df = load_data()
    df = encode_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    print("Training RandomForest...")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Model trained successfully.")

    # Predictions (clipped to >= 0)
    preds = np.clip(model.predict(X_test), 0, None)

    # ══════════════════════════════════════
    # EVALUATION (MAE, RMSE, R², MAPE)
    # ══════════════════════════════════════
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mape = compute_mape(y_test, preds)

    print(f"\n{'='*45}")
    print(f"  Holdout Evaluation (80/20 Time-Series Split)")
    print(f"{'='*45}")
    print(f"  MAE  = {mae:,.2f} W")
    print(f"  RMSE = {rmse:,.2f} W")
    print(f"  R²   = {r2:.4f}")
    print(f"  MAPE = {mape:.2f} %")
    print(f"{'='*45}")

    # ══════════════════════════════════════
    # TIMESERIES CROSS-VALIDATION (k=5)
    # ══════════════════════════════════════
    print("\nRunning TimeSeriesSplit Cross-Validation (k=5)...")

    # Use full sorted data for CV
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

        fold_mae = mean_absolute_error(y_cv_val, cv_preds)
        fold_rmse = np.sqrt(mean_squared_error(y_cv_val, cv_preds))
        fold_r2 = r2_score(y_cv_val, cv_preds)
        fold_mape = compute_mape(y_cv_val, cv_preds)

        cv_mae.append(fold_mae)
        cv_rmse.append(fold_rmse)
        cv_r2.append(fold_r2)
        cv_mape.append(fold_mape)

        print(f"  Fold {fold}: MAE={fold_mae:,.1f}  RMSE={fold_rmse:,.1f}  "
              f"R²={fold_r2:.4f}  MAPE={fold_mape:.1f}%")

    print(f"\n{'='*45}")
    print(f"  CV Average (k=5)")
    print(f"{'='*45}")
    print(f"  MAE  = {np.mean(cv_mae):,.2f} ± {np.std(cv_mae):,.2f}")
    print(f"  RMSE = {np.mean(cv_rmse):,.2f} ± {np.std(cv_rmse):,.2f}")
    print(f"  R²   = {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")
    print(f"  MAPE = {np.nanmean(cv_mape):.2f} ± {np.nanstd(cv_mape):.2f} %")
    print(f"{'='*45}")

    # ══════════════════════════════════════
    # SAVE MODEL
    # ══════════════════════════════════════
    joblib.dump(model, "models/solar_model.pkl")
    print("\nModel saved -> models/solar_model.pkl")

    # ══════════════════════════════════════
    # LOG HYPERPARAMETERS & METRICS -> JSON
    # ══════════════════════════════════════
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
        "holdout_metrics": {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "MAPE": round(mape, 2),
        },
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

    with open("models/training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)
    print("Training log saved -> models/training_log.json")

    # ══════════════════════════════════════
    # FEATURE IMPORTANCE
    # ══════════════════════════════════════
    print("\nCalculating feature importance...")

    importance = model.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
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
    plt.savefig("models/feature_importance.png", dpi=150)
    print("Feature importance plot saved")

    # ══════════════════════════════════════
    # PREDICTION vs ACTUAL
    # ══════════════════════════════════════
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
    plt.savefig("models/prediction_vs_actual.png", dpi=150)
    print("Prediction vs Actual plot saved")

    print("\nTraining complete!")


if __name__ == "__main__":
    train_model()