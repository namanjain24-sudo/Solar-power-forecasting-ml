import pandas as pd


def load_data():
    df = pd.read_csv("data/processed/solar_final.csv")

    print("Data loaded successfully")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns)

    return df


if __name__ == "__main__":
    load_data()