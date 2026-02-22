# scripts/run_evaluation.py
# Run baselines + generate plots (platform-independent)

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def main():
    print("=== Running all baselines ===")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "run_baselines.py")], cwd="."
    )
    if result.returncode != 0:
        print("Baselines failed!")
        sys.exit(result.returncode)

    print("\n=== Generating plots ===")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "plot_results.py"), "--plots", "all"],
        cwd=".",
    )
    if result.returncode != 0:
        print("Plotting failed!")
        sys.exit(result.returncode)

    print("\n========================================")
    print(" Evaluation completed!")
    print(" Results: results/")
    print(" Plots:   results/plots/")
    print("========================================")


if __name__ == "__main__":
    main()
