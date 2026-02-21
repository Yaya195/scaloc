"""
Debug script to identify NaN sources in the FL experiment.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RPS_DIR = Path("data/processed/rps")
SAMPLES_DIR = Path("data/processed/samples")


def check_rps():
    """Check for empty or problematic RP fingerprints."""
    print("=" * 60)
    print("CHECKING RP DATA")
    print("=" * 60)
    
    for rp_path in sorted(RPS_DIR.glob("train_rps_*.parquet")):
        domain_id = rp_path.stem.replace("train_rps_", "")
        rp_table = pd.read_parquet(rp_path)
        
        print(f"\nDomain {domain_id}:")
        print(f"  Total RPs: {len(rp_table)}")
        
        # Check for empty AP lists
        empty_aps = 0
        min_aps = float('inf')
        max_aps = 0
        
        for _, row in rp_table.iterrows():
            ap_ids = row["ap_ids"]
            rssi = row["rssi"]
            
            if len(ap_ids) == 0:
                empty_aps += 1
                print(f"  WARNING: RP {row['rp_id']} has EMPTY AP list!")
            
            min_aps = min(min_aps, len(ap_ids))
            max_aps = max(max_aps, len(ap_ids))
            
            # Check for NaN in RSSI
            if any(np.isnan(rssi)):
                print(f"  WARNING: RP {row['rp_id']} has NaN in RSSI!")
        
        print(f"  AP count range: {min_aps} - {max_aps}")
        if empty_aps > 0:
            print(f"  Empty AP lists: {empty_aps}")


def check_queries():
    """Check for empty or problematic query fingerprints."""
    print("\n" + "=" * 60)
    print("CHECKING QUERY DATA")
    print("=" * 60)
    
    train_samples = pd.read_parquet(SAMPLES_DIR / "train_samples.parquet")
    
    for domain_id in train_samples["domain_id"].unique():
        q_df = train_samples[train_samples["domain_id"] == domain_id]
        
        print(f"\nDomain {domain_id}:")
        print(f"  Total queries: {len(q_df)}")
        
        empty_aps = 0
        min_aps = float('inf')
        max_aps = 0
        
        for _, row in q_df.iterrows():
            aps = row["aps"]
            
            # Handle JSON serialization
            if isinstance(aps, str):
                import json
                aps = json.loads(aps)
            
            if len(aps) == 0:
                empty_aps += 1
                print(f"  WARNING: Query has EMPTY AP list!")
            
            min_aps = min(min_aps, len(aps))
            max_aps = max(max_aps, len(aps))
        
        print(f"  AP count range: {min_aps} - {max_aps}")
        if empty_aps > 0:
            print(f"  Empty AP lists: {empty_aps}")


def check_coordinates():
    """Check for invalid coordinates."""
    print("\n" + "=" * 60)
    print("CHECKING COORDINATE DATA")
    print("=" * 60)
    
    for rp_path in sorted(RPS_DIR.glob("train_rps_*.parquet")):
        domain_id = rp_path.stem.replace("train_rps_", "")
        rp_table = pd.read_parquet(rp_path)
        
        coords = rp_table[["x", "y"]].to_numpy()
        
        if np.any(np.isnan(coords)):
            print(f"  WARNING: Domain {domain_id} has NaN coordinates!")
        
        if np.any(np.isinf(coords)):
            print(f"  WARNING: Domain {domain_id} has infinite coordinates!")


if __name__ == "__main__":
    check_rps()
    check_queries()
    check_coordinates()
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
