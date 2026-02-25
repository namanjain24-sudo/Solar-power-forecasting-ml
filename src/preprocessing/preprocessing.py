"""
Data preprocessing for the Solar Power Forecasting project.

Handles:
  - Categorical encoding (SOURCE_KEY → numeric)
  - Time-based train-test split (no data leakage)
"""

import pandas as pd
from src.data.load_data import load_data

# Features used by the model
FEATURES = [
    "SOURCE_KEY",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "hour",
    "month",
]

TARGET = "DC_POWER"


def encode_features(df):
    """Encode categorical columns to numeric labels.

    Converts SOURCE_KEY (inverter ID) from string to integer category codes.
    This is required because decision tree models expect numeric input.

    Args:
        df: DataFrame with a SOURCE_KEY column.

    Returns:
        DataFrame with SOURCE_KEY encoded as integers.
    """
    print("Encoding SOURCE_KEY...")
    if df["SOURCE_KEY"].dtype == "object" or str(df["SOURCE_KEY"].dtype) == "category":
        df["SOURCE_KEY"] = df["SOURCE_KEY"].astype("category").cat.codes
    print("Encoding done")
    return df


def split_data(df, train_ratio=0.8):
    """Time-based train-test split.

    Sorts by DATE_TIME and splits chronologically to prevent
    future data from leaking into training (temporal integrity).

    Args:
        df: Preprocessed DataFrame.
        train_ratio: Fraction of data used for training.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    print("Splitting data (time-series)...")

    df = df.sort_values("DATE_TIME")

    X = df[FEATURES]
    y = df[TARGET]

    split_index = int(len(df) * train_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
