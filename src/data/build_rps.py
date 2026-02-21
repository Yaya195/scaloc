# src/data/build_rps.py

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DOMAINS_DIR = Path("data/interim/domains")
RPS_DIR = Path("data/processed/rps")
RPS_DIR.mkdir(parents=True, exist_ok=True)


def list_train_domains() -> List[Path]:
    # Also look for old .pkl files for backward compatibility
    return sorted(DOMAINS_DIR.glob("train_domain_*.parquet"))


def encode_rp_fingerprints(samples: pd.DataFrame):
    """
    Aggregate all samples in the cluster into a single RP fingerprint:
        - group by ap_id
        - average RSSI per AP

    Returns:
        ap_ids: list[int]
        rssi:   list[float]
    """
    ap_to_rssi = {}

    for _, row in samples.iterrows():
        for a in row["aps"]:
            ap_id = a["ap_id"]
            rssi = a["rssi"]
            ap_to_rssi.setdefault(ap_id, []).append(rssi)

    ap_ids = []
    rssi_vals = []
    for ap_id, vals in ap_to_rssi.items():
        # Extract numeric part from ap_id (e.g., 'WAP007' -> 7)
        ap_num = int(ap_id.replace('WAP', ''))
        ap_ids.append(ap_num)
        rssi_vals.append(float(np.mean(vals)))

    return ap_ids, rssi_vals


def adaptive_num_rps(n_unique_positions: int, n_spaces: int) -> int:
    n = int(round(0.5 * np.sqrt(n_unique_positions) * np.log1p(n_spaces)))
    return max(5, n)


def build_rps_for_domain(domain_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(domain_path)

    unique_positions = df[["x", "y"]].drop_duplicates().reset_index(drop=True)
    n_unique = len(unique_positions)
    n_spaces = df["space_id"].nunique()

    k = adaptive_num_rps(n_unique, n_spaces)
    k = min(k, n_unique)

    km = KMeans(n_clusters=k, random_state=0)
    labels_unique = km.fit_predict(unique_positions[["x", "y"]])
    unique_positions["cluster_id"] = labels_unique

    merged = df.merge(unique_positions, on=["x", "y"], how="left")

    rps = []
    for cid in range(k):
        cluster_samples = merged[merged["cluster_id"] == cid]

        cx = cluster_samples["x"].mean()
        cy = cluster_samples["y"].mean()
        ap_ids, rssi = encode_rp_fingerprints(cluster_samples)

        rps.append(
            {
                "domain_id": cluster_samples["domain_id"].iloc[0],
                "rp_id": f"rp_{cid}",
                "x": cx,
                "y": cy,
                "ap_ids": ap_ids,
                "rssi": rssi,
            }
        )

    return pd.DataFrame(rps)


def main():
    for domain_path in list_train_domains():
        rps_df = build_rps_for_domain(domain_path)
        domain_id = domain_path.stem.replace("train_domain_", "")
        out_path = RPS_DIR / f"train_rps_{domain_id}.parquet"
        rps_df.to_parquet(out_path, index=False)
        print(f"[adaptive_rps] Saved RPs for train {domain_id} to {out_path}")


if __name__ == "__main__":
    main()