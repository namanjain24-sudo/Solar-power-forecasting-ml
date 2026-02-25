"""
Evaluation metrics for the Solar Power Forecasting project.

Provides standard regression metrics:
  - MAE  (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R²   (Coefficient of Determination)
  - MAPE (Mean Absolute Percentage Error)
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_mape(y_true, y_pred, threshold=10):
    """Compute MAPE, excluding near-zero actuals to avoid division by zero.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        threshold: Minimum absolute value for inclusion.

    Returns:
        MAPE as a percentage (float).
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = np.abs(y_true) > threshold
    if mask.sum() == 0:
        return float("nan")
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(y_true, y_pred):
    """Compute all regression metrics for a model.

    Args:
        y_true: Actual target values.
        y_pred: Predicted values (should already be clipped to >= 0).

    Returns:
        Dictionary with MAE, RMSE, R2, and MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = compute_mape(y_true, y_pred)

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
    }


def print_metrics(metrics, title="Evaluation"):
    """Pretty-print a metrics dictionary."""
    print(f"\n{'='*45}")
    print(f"  {title}")
    print(f"{'='*45}")
    print(f"  MAE  = {metrics['MAE']:,.2f} W")
    print(f"  RMSE = {metrics['RMSE']:,.2f} W")
    print(f"  R²   = {metrics['R2']:.4f}")
    print(f"  MAPE = {metrics['MAPE']:.2f} %")
    print(f"{'='*45}")
