# scripts/run_full_pipeline.py
# Complete pipeline: data -> training -> baselines -> plots
# Platform-independent entry point.

import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def run(label, cmd):
    print(f"\n{'=' * 48}")
    print(f" {label}")
    print('=' * 48)
    result = subprocess.run(cmd, cwd=".")
    if result.returncode != 0:
        print(f"FAILED at: {label}")
        sys.exit(result.returncode)


def main():
    # Phase 1: Data pipeline
    run("PHASE 1: Data Pipeline",
        [sys.executable, str(SCRIPTS_DIR / "run_data_pipeline.py")])

    # Phase 2: Federated GNN training
    run("PHASE 2: Federated GNN Training",
        [sys.executable, "run_fl_experiment.py"])

    # Phase 3: Baselines
    run("PHASE 3: Baselines",
        [sys.executable, str(SCRIPTS_DIR / "run_baselines.py")])

    # Phase 4: Plots
    run("PHASE 4: Visualization",
        [sys.executable, str(SCRIPTS_DIR / "plot_results.py"), "--plots", "all"])

    print("\n========================================")
    print(" Full pipeline completed!")
    print(" Results: results/")
    print(" Plots:   results/plots/")
    print("========================================")


if __name__ == "__main__":
    main()
