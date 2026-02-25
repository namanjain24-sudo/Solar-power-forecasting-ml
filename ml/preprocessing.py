import pandas as pd
from .load_data import load_data


def encode_features(df):
    """Encode categorical features (SOURCE_KEY) to numerical labels."""
    print("Encoding SOURCE_KEY...")
    df["SOURCE_KEY"] = df["SOURCE_KEY"].astype("category").cat.codes
    print("Encoding done")
    return df


def split_data(df):
    """Time-based 80/20 train-test split (no data leakage)."""
    print("Splitting data (time-series)...")

    # Sort by time
    df = df.sort_values("DATE_TIME")

    features = [
        "SOURCE_KEY",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
        "hour",
        "month",
    ]

    target = "DC_POWER"

    X = df[features]
    y = df[target]

    # Time-based split
    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("Split complete")
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)