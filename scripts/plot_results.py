# scripts/plot_results.py
#
# Visualization scripts for training curves, baseline comparison,
# per-domain analysis, error CDFs, and scalability results.

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("results/plots")


def load_experiment(name):
    """Load an experiment JSON from results/."""
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        print(f"Warning: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def plot_training_curves(experiment_names=None):
    """
    Plot training loss curves for one or more experiments.
    Overlays them for comparison.
    """
    import matplotlib.pyplot as plt

    if experiment_names is None:
        experiment_names = ["fedgnn"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Loss curves
    ax = axes[0]
    for name in experiment_names:
        data = load_experiment(name)
        if data is None:
            continue
        rounds = [r["round"] for r in data["rounds"]]
        losses = [r["avg_loss"] for r in data["rounds"]]
        ax.plot(rounds, losses, label=name, marker=".", markersize=3)

    ax.set_xlabel("Round / Epoch")
    ax.set_ylabel("Average Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Evaluation metric (mean error) over time
    ax = axes[1]
    for name in experiment_names:
        data = load_experiment(name)
        if data is None:
            continue
        eval_rounds = []
        eval_errors = []
        for r in data["rounds"]:
            if r.get("eval_metrics") and "mean_error" in r["eval_metrics"]:
                eval_rounds.append(r["round"])
                eval_errors.append(r["eval_metrics"]["mean_error"])
        if eval_rounds:
            ax.plot(eval_rounds, eval_errors, label=name, marker="o", markersize=4)

    ax.set_xlabel("Round / Epoch")
    ax.set_ylabel("Mean Error (normalized)")
    ax.set_title("Validation Mean Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_client_losses(experiment_name="fedgnn"):
    """Plot per-client loss trajectories."""
    import matplotlib.pyplot as plt

    data = load_experiment(experiment_name)
    if data is None:
        return

    # Collect all client IDs
    client_ids = set()
    for r in data["rounds"]:
        client_ids.update(r.get("client_losses", {}).keys())
    client_ids = sorted(client_ids)

    fig, ax = plt.subplots(figsize=(10, 5))
    for cid in client_ids:
        rounds = []
        losses = []
        for r in data["rounds"]:
            if cid in r.get("client_losses", {}):
                rounds.append(r["round"])
                losses.append(r["client_losses"][cid])
        if rounds:
            ax.plot(rounds, losses, label=cid, alpha=0.7)

    ax.set_xlabel("Round")
    ax.set_ylabel("Loss")
    ax.set_title(f"Per-Client Losses ({experiment_name})")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / f"client_losses_{experiment_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_baseline_comparison():
    """Bar chart comparing baselines on key metrics."""
    import matplotlib.pyplot as plt

    path = RESULTS_DIR / "baseline_comparison.json"
    if not path.exists():
        print(f"No baseline comparison found at {path}. Run run_baselines.py first.")
        return

    with open(path) as f:
        comparison = json.load(f)

    methods = list(comparison.keys())
    metrics_to_plot = ["mean_error", "median_error", "p75_error", "p90_error"]
    metric_labels = ["Mean", "Median", "P75", "P90"]

    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        metrics = comparison[method]
        values = [metrics.get(m, 0) for m in metrics_to_plot]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=method)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Error (normalized)")
    ax.set_title("Baseline Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "baseline_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_error_cdf(experiment_names=None):
    """
    Plot CDF of localization errors for experiments that have
    per-sample error data. Falls back to reconstructing from metrics.
    """
    import matplotlib.pyplot as plt

    if experiment_names is None:
        experiment_names = ["fedgnn"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for name in experiment_names:
        data = load_experiment(name)
        if data is None:
            continue

        # Try to find final eval metrics
        final = data["rounds"][-1] if data["rounds"] else {}
        metrics = final.get("eval_metrics", {})

        if not metrics:
            continue

        # Approximate CDF from summary stats
        points = []
        if "median_error" in metrics:
            points.append((metrics["median_error"], 0.50))
        if "p75_error" in metrics:
            points.append((metrics["p75_error"], 0.75))
        if "p90_error" in metrics:
            points.append((metrics["p90_error"], 0.90))
        if "cep_5m" in metrics:
            points.append((5.0, metrics["cep_5m"] / 100.0))
        if "cep_10m" in metrics:
            points.append((10.0, metrics["cep_10m"] / 100.0))

        if points:
            points.sort()
            xs, ys = zip(*points)
            ax.plot(xs, ys, marker="o", label=name)

    ax.set_xlabel("Error")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative Distribution of Errors")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "error_cdf.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_scalability():
    """Plot scalability experiment results."""
    import matplotlib.pyplot as plt

    summary_path = Path("results/scalability/scalability_summary.json")
    if not summary_path.exists():
        print(f"No scalability results at {summary_path}. Run run_scalability.py first.")
        return

    with open(summary_path) as f:
        all_results = json.load(f)

    n_plots = len(all_results)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(2, n_plots, figsize=(6 * n_plots, 10))
    if n_plots == 1:
        axes = axes.reshape(-1, 1)

    plot_idx = 0
    for exp_type, results in all_results.items():
        # Determine x-axis variable
        if exp_type == "domains":
            xs = [r.get("K", 0) for r in results]
            xlabel = "Number of Domains (K)"
        elif exp_type == "rps":
            xs = [r.get("max_rps", 0) for r in results]
            xlabel = "Max RPs per Domain"
            # Convert "all" to a large number for plotting
            xs = [x if isinstance(x, (int, float)) else 999 for x in xs]
        elif exp_type == "queries":
            xs = [r.get("max_queries", 0) for r in results]
            xlabel = "Max Queries per Domain"
            xs = [x if isinstance(x, (int, float)) else 9999 for x in xs]
        elif exp_type == "rounds":
            xs = [r.get("rounds", 0) for r in results]
            xlabel = "FL Rounds"
        else:
            continue

        # Top: final mean error
        ax_top = axes[0, plot_idx]
        mean_errors = [
            r.get("eval_metrics", {}).get("mean_error", float("nan")) for r in results
        ]
        ax_top.plot(xs, mean_errors, marker="o", color="tab:blue")
        ax_top.set_xlabel(xlabel)
        ax_top.set_ylabel("Mean Error (normalized)")
        ax_top.set_title(f"Accuracy vs {exp_type}")
        ax_top.grid(True, alpha=0.3)

        # Bottom: training time
        ax_bot = axes[1, plot_idx]
        times = [r.get("total_time_sec", 0) for r in results]
        ax_bot.plot(xs, times, marker="s", color="tab:orange")
        ax_bot.set_xlabel(xlabel)
        ax_bot.set_ylabel("Total Time (seconds)")
        ax_bot.set_title(f"Time vs {exp_type}")
        ax_bot.grid(True, alpha=0.3)

        plot_idx += 1

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / "scalability.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_per_domain_heatmap(experiment_name="fedgnn"):
    """
    Heatmap of per-domain error metrics from the final evaluation.
    Reads per-domain results from individual scalability/baseline JSONs.
    """
    import matplotlib.pyplot as plt

    # Try to find detailed per-domain results
    data = load_experiment(experiment_name)
    if data is None:
        return

    # Check if final round has per-domain eval
    final = data["rounds"][-1] if data.get("rounds") else {}
    metrics = final.get("eval_metrics", {})

    if not metrics or "mean_error" not in metrics:
        print(f"No per-domain data available for {experiment_name}")
        return

    print(f"Global metrics for {experiment_name}: {metrics}")


def main():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument(
        "--plots", nargs="+",
        choices=["training", "clients", "baselines", "cdf", "scalability", "domain", "all"],
        default=["all"],
        help="Which plots to generate",
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Experiment names for training curves (default: fedgnn)",
    )
    args = parser.parse_args()

    plots = args.plots
    if "all" in plots:
        plots = ["training", "clients", "baselines", "cdf", "scalability"]

    if "training" in plots:
        exps = args.experiments or ["fedgnn", "centralized_gnn", "centralized_mlp", "federated_mlp"]
        plot_training_curves(exps)

    if "clients" in plots:
        plot_client_losses()

    if "baselines" in plots:
        plot_baseline_comparison()

    if "cdf" in plots:
        exps = args.experiments or ["fedgnn"]
        plot_error_cdf(exps)

    if "scalability" in plots:
        plot_scalability()

    if "domain" in plots:
        name = args.experiments[0] if args.experiments else "fedgnn"
        plot_per_domain_heatmap(name)


if __name__ == "__main__":
    main()
