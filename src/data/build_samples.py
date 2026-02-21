# src/data/build_samples.py
#
# Purpose:
#   Prepare query samples for training and validation.
#   Each sample contains:
#     - domain_id
#     - x, y (local metric coordinates)
#     - aps (AP-wise fingerprint)
#
# Note:
#   - Training samples will be used to learn.
#   - Validation samples will be used to evaluate.
#   - Neither modifies the RP graph.

from pathlib import Path
import json
import pandas as pd

INTERIM_CLEAN_DIR = Path("data/interim/clean")
SAMPLES_DIR = Path("data/processed/samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def load_clean_split(split_name: str) -> pd.DataFrame:
    """
    Load a cleaned split (train or val) from disk.
    """
    return pd.read_parquet(INTERIM_CLEAN_DIR / f"{split_name}_clean.parquet")


def main():
    for split in ["train", "val"]:
        df = load_clean_split(split)

        # Domain = building_floor (same as for RPs/graphs)
        df["domain_id"] = df["building"].astype(str) + "_" + df["floor"].astype(str)

        # Keep only what the model needs for queries
        samples = df[["domain_id", "x", "y", "aps"]].copy()
        out_path = SAMPLES_DIR / f"{split}_samples.parquet"
        samples.to_parquet(out_path, index=False)
        print(f"[build_samples] Saved {split} samples to {out_path}")


if __name__ == "__main__":
    main()