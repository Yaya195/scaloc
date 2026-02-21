"""
Check raw validation data for samples with no valid AP readings.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA = Path("data/raw/validationData.csv")

def analyze_validation_data():
    print("=" * 70)
    print("ANALYZING VALIDATION DATA FOR EMPTY/INVALID SAMPLES")
    print("=" * 70)
    
    # Load raw data
    df = pd.read_csv(RAW_DATA)
    
    # Get WAP columns
    wap_cols = [col for col in df.columns if col.startswith('WAP')]
    print(f"\nTotal samples in validation data: {len(df)}")
    print(f"Total WAP columns: {len(wap_cols)}")
    
    # Analyze each row
    empty_samples = []
    weak_only_samples = []
    min_rssi_threshold = -95
    
    for idx, row in df.iterrows():
        wap_values = row[wap_cols].values
        
        # Count how many APs are detected (not 100)
        detected = wap_values[wap_values != 100]
        
        # Count how many are above the threshold
        strong_enough = detected[detected >= min_rssi_threshold]
        
        if len(detected) == 0:
            empty_samples.append({
                'index': idx,
                'building': row['BUILDINGID'],
                'floor': row['FLOOR'],
                'detected_count': 0,
                'strong_count': 0
            })
        elif len(strong_enough) == 0:
            weak_only_samples.append({
                'index': idx,
                'building': row['BUILDINGID'],
                'floor': row['FLOOR'],
                'detected_count': len(detected),
                'strong_count': 0,
                'strongest_rssi': detected.max()
            })
    
    total_invalid = len(empty_samples) + len(weak_only_samples)
    
    print(f"\nSamples with NO APs detected: {len(empty_samples)}")
    print(f"Samples with ONLY weak APs (all < {min_rssi_threshold} dBm): {len(weak_only_samples)}")
    print(f"\n" + "=" * 70)
    print(f"TOTAL INVALID SAMPLES: {total_invalid} / {len(df)} ({100*total_invalid/len(df):.2f}%)")
    print("=" * 70)

if __name__ == "__main__":
    analyze_validation_data()
