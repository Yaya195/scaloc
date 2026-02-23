# src/data/clean_preprocess.py
#
# Purpose:
#   - Convert UTM (already in meters) to local metric coordinates (x, y) per DOMAIN.
#     A domain = (BUILDINGID, FLOOR).
#   - Clean RSSI values and convert fingerprints to AP-wise sets.
#   - Save cleaned train/val splits in a consistent coordinate system.
#
# Key idea:
#   - Compute a reference (x0, y0) per DOMAIN from the TRAINING set.
#   - Apply the SAME transform to both training and validation.
#   - This ensures consistent metric coordinates across splits and domains.

from pathlib import Path
from typing import Dict, Any, Tuple
import re
import sys

import numpy as np
import pandas as pd

from .load_raw import load_training_data, load_validation_data

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
if str(CONFIGS_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIGS_DIR))
from load_config import load_config

INTERIM_DIR = Path("data/interim/clean")
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

N_APS = 520
AP_COLS = [f"WAP{i}" for i in range(1, N_APS + 1)]


def get_ap_cols(df: pd.DataFrame) -> list[str]:
    """Detect AP columns dynamically (supports WAP1..WAP520 and WAP001..WAP520)."""
    ap_pattern = re.compile(r"^WAP(\d+)$")
    cols = [col for col in df.columns if ap_pattern.match(str(col))]
    cols.sort(key=lambda col: int(ap_pattern.match(str(col)).group(1)))

    if not cols:
        raise ValueError("No WAP columns found. Expected columns like WAP1 or WAP001.")
    return cols


def compute_domain_refs(train_df: pd.DataFrame) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """
    Compute a reference (x0, y0) per DOMAIN = (BUILDINGID, FLOOR)
    from the TRAINING set. These references are reused for validation.
    """
    refs: Dict[Tuple[int, int], Tuple[float, float]] = {}

    for (b, f) in train_df[["BUILDINGID", "FLOOR"]].drop_duplicates().itertuples(index=False):
        sub = train_df[(train_df["BUILDINGID"] == b) & (train_df["FLOOR"] == f)]
        x0 = sub["LONGITUDE"].mean()   # UTM meters
        y0 = sub["LATITUDE"].mean()    # UTM meters
        refs[(int(b), int(f))] = (x0, y0)

    return refs


def utm_to_local_xy(df: pd.DataFrame, refs: Dict[Tuple[int, int], Tuple[float, float]]) -> pd.DataFrame:
    """
    Convert UTM coordinates (already in meters) to local (x, y) per DOMAIN.
    No trig, no radians, no Earth radius.
    """
    df = df.copy()

    xs = np.zeros(len(df))
    ys = np.zeros(len(df))

    for (b, f) in df[["BUILDINGID", "FLOOR"]].drop_duplicates().itertuples(index=False):
        mask = (df["BUILDINGID"] == b) & (df["FLOOR"] == f)

        x0, y0 = refs[(int(b), int(f))]

        xs[mask] = df.loc[mask, "LONGITUDE"] - x0
        ys[mask] = df.loc[mask, "LATITUDE"] - y0

    df["x"] = xs
    df["y"] = ys
    return df


def clean_rssi(df: pd.DataFrame, ap_cols: list[str], min_rssi: int = -104) -> pd.DataFrame:
    """Clean RSSI values: replace 100 with NA, drop very weak signals."""
    df = df.copy()
    for col in ap_cols:
        df[col] = df[col].replace(100, pd.NA)
        df.loc[df[col].notna() & (df[col] < min_rssi), col] = pd.NA
    return df


def row_to_apwise_set(row: pd.Series, ap_cols: list[str]) -> Dict[str, Any]:
    """Convert a row into an AP-wise fingerprint representation."""
    aps = []
    for ap in ap_cols:
        rssi = row[ap]
        if pd.isna(rssi):
            continue
        aps.append({"ap_id": ap, "rssi": float(rssi)})

    return {
        "building": int(row["BUILDINGID"]),
        "floor": int(row["FLOOR"]),
        "space_id": row.get("SPACEID", None),
        "x": float(row["x"]),
        "y": float(row["y"]),
        "aps": aps,
    }


def process_split(
    df: pd.DataFrame,
    split_name: str,
    refs: Dict[Tuple[int, int], Tuple[float, float]],
    min_rssi_threshold: int,
) -> None:
    """Apply coordinate transform + RSSI cleaning + AP-wise conversion to a split."""
    ap_cols = get_ap_cols(df)
    df = utm_to_local_xy(df, refs)
    df = clean_rssi(df, ap_cols=ap_cols, min_rssi=min_rssi_threshold)

    records = [row_to_apwise_set(row, ap_cols=ap_cols) for _, row in df.iterrows()]
    
    # Filter out samples with empty AP lists (no visible APs after cleaning)
    original_count = len(records)
    records = [r for r in records if len(r["aps"]) > 0]
    filtered_count = original_count - len(records)
    
    if filtered_count > 0:
        print(f"[clean_preprocess] Filtered {filtered_count} samples with no visible APs (from {original_count})")
    
    out_path = INTERIM_DIR / f"{split_name}_clean.parquet"
    pd.DataFrame(records).to_parquet(out_path, index=False)
    print(f"[clean_preprocess] Saved cleaned {split_name} to {out_path} ({len(records)} samples)")


def main():
    train_raw = load_training_data()
    val_raw = load_validation_data()

    data_cfg = load_config("data_config")
    min_rssi_threshold = data_cfg.get("preprocess", {}).get("min_rssi_threshold", -104)

    refs = compute_domain_refs(train_raw)

    process_split(train_raw, "train", refs, min_rssi_threshold)
    process_split(val_raw, "val", refs, min_rssi_threshold)


if __name__ == "__main__":
    main()