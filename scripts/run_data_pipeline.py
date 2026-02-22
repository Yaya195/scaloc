# scripts/run_data_pipeline.py
# Full data preparation pipeline (platform-independent)

import subprocess
import sys

STEPS = [
    ("Step 0: Inspect raw data",    [sys.executable, "-m", "src.data.load_raw"]),
    ("Step 1: Clean + preprocess",  [sys.executable, "-m", "src.data.clean_preprocess"]),
    ("Step 2: Build domains",       [sys.executable, "-m", "src.data.build_domains"]),
    ("Step 3: Build RPs",           [sys.executable, "-m", "src.data.build_rps"]),
    ("Step 4: Build graphs",        [sys.executable, "-m", "src.data.build_graphs"]),
    ("Step 5: Build samples",       [sys.executable, "-m", "src.data.build_samples"]),
]


def main():
    for label, cmd in STEPS:
        print(f"\n=== {label} ===")
        result = subprocess.run(cmd, cwd=".")
        if result.returncode != 0:
            print(f"FAILED at: {label}")
            sys.exit(result.returncode)

    print("\n========================================")
    print(" Data pipeline completed successfully!")
    print("========================================")


if __name__ == "__main__":
    main()
