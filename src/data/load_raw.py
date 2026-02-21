# src/data/load_raw.py
#
# Purpose:
#   Centralized loader for raw UJIIndoorLoc CSV files.
#   No cleaning, no transformations — just structured loading.

from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")


def load_training_data() -> pd.DataFrame:
    """
    Load the UJIIndoorLoc training CSV as a DataFrame.
    """
    return pd.read_csv(RAW_DIR / "trainingData.csv")


def load_validation_data() -> pd.DataFrame:
    """
    Load the UJIIndoorLoc validation CSV as a DataFrame.
    """
    return pd.read_csv(RAW_DIR / "validationData.csv")


def main():
    train_df = load_training_data()
    val_df = load_validation_data()

    print(f"Training samples:   {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")


if __name__ == "__main__":
    main()