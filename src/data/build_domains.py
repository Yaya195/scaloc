# src/data/build_domains.py
#
# Purpose:
#   Split cleaned data into domains defined as (building, floor).
#   Domains are used as federated clients later.

from pathlib import Path
import pandas as pd

INTERIM_CLEAN_DIR = Path("data/interim/clean")
DOMAINS_DIR = Path("data/interim/domains")
DOMAINS_DIR.mkdir(parents=True, exist_ok=True)


def load_clean_split(split_name: str) -> pd.DataFrame:
    """
    Load a cleaned split (train or val) from disk.
    """
    return pd.read_parquet(INTERIM_CLEAN_DIR / f"{split_name}_clean.parquet")


def main():
    for split in ["train", "val"]:
        df = load_clean_split(split)

        # Domain = building_floor (e.g., "0_2")
        df["domain_id"] = df["building"].astype(str) + "_" + df["floor"].astype(str)

        for domain_id, group in df.groupby("domain_id"):
            out_path = DOMAINS_DIR / f"{split}_domain_{domain_id}.parquet"
            group.to_parquet(out_path, index=False)
            print(f"[build_domains] Saved {split} domain {domain_id} with {len(group)} samples")


if __name__ == "__main__":
    main()