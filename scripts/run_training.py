# scripts/run_training.py
# Run federated GNN training (platform-independent)

import subprocess
import sys


def main():
    print("=== Federated GNN Training ===")
    result = subprocess.run([sys.executable, "run_fl_experiment.py"], cwd=".")
    if result.returncode != 0:
        print("Training failed!")
        sys.exit(result.returncode)
    print("=== Training completed! ===")


if __name__ == "__main__":
    main()
