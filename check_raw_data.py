"""
Check raw training data for samples with no valid AP readings.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA = Path("data/raw/trainingData.csv")

def analyze_raw_data():
    print("=" * 70)
    print("ANALYZING RAW DATA FOR EMPTY/INVALID SAMPLES")
    print("=" * 70)
    
    # Load raw data
    df = pd.read_csv(RAW_DATA)
    
    # Get WAP columns
    wap_cols = [col for col in df.columns if col.startswith('WAP')]
    print(f"\nTotal samples in raw data: {len(df)}")
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
            # ALL APs are 100 (not detected)
            empty_samples.append({
                'index': idx,
                'building': row['BUILDINGID'],
                'floor': row['FLOOR'],
                'detected_count': 0,
                'strong_count': 0,
                'reason': 'All APs = 100 (not detected)'
            })
        elif len(strong_enough) == 0:
            # Some APs detected but ALL below threshold
            weak_only_samples.append({
                'index': idx,
                'building': row['BUILDINGID'],
                'floor': row['FLOOR'],
                'detected_count': len(detected),
                'strong_count': 0,
                'strongest_rssi': detected.max(),
                'reason': f'All detected APs < {min_rssi_threshold} dBm'
            })
    
    # Report findings
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nSamples with NO APs detected: {len(empty_samples)}")
    if empty_samples:
        print("\nFirst 10 examples:")
        for sample in empty_samples[:10]:
            print(f"  Row {sample['index']}: Building {sample['building']}, Floor {sample['floor']}")
            print(f"    → {sample['reason']}")
    
    print(f"\nSamples with ONLY weak APs (all < {min_rssi_threshold} dBm): {len(weak_only_samples)}")
    if weak_only_samples:
        print("\nFirst 10 examples:")
        for sample in weak_only_samples[:10]:
            print(f"  Row {sample['index']}: Building {sample['building']}, Floor {sample['floor']}")
            print(f"    → Detected: {sample['detected_count']} APs, Strongest: {sample['strongest_rssi']:.1f} dBm")
    
    total_invalid = len(empty_samples) + len(weak_only_samples)
    print(f"\n" + "=" * 70)
    print(f"TOTAL INVALID SAMPLES: {total_invalid} / {len(df)} ({100*total_invalid/len(df):.2f}%)")
    print("=" * 70)
    
    # Show distribution by building/floor
    if total_invalid > 0:
        print("\nDistribution by building/floor:")
        all_invalid = empty_samples + weak_only_samples
        invalid_df = pd.DataFrame(all_invalid)
        distribution = invalid_df.groupby(['building', 'floor']).size().sort_values(ascending=False)
        print(distribution.to_string())
    
    # Show example of a problematic row
    if empty_samples:
        print("\n" + "=" * 70)
        print("EXAMPLE: First sample with all APs = 100")
        print("=" * 70)
        idx = empty_samples[0]['index']
        row = df.iloc[idx]
        print(f"\nRow index: {idx}")
        print(f"BUILDINGID: {row['BUILDINGID']}")
        print(f"FLOOR: {row['FLOOR']}")
        print(f"LONGITUDE: {row['LONGITUDE']}")
        print(f"LATITUDE: {row['LATITUDE']}")
        print(f"\nWAP values (showing first 20):")
        for i, col in enumerate(wap_cols[:20]):
            print(f"  {col}: {row[col]}")
        
        # Check if ALL are 100
        all_vals = row[wap_cols].values
        unique_vals = np.unique(all_vals)
        print(f"\nUnique RSSI values in this row: {unique_vals}")

if __name__ == "__main__":
    analyze_raw_data()
