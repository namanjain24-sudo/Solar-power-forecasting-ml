"""
Data loading utilities for the Solar Power Forecasting project.
"""

import pandas as pd


def load_data(path="data/processed/solar_final.csv"):
    """Load the preprocessed solar generation dataset.

    Args:
        path: Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset with all columns.
    """
    df = pd.read_csv(path)

    print(f"Data loaded — {df.shape[0]} rows, {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    df = load_data()
    print("Columns:", list(df.columns))
